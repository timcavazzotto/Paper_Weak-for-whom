"""
01_prepare.py — base analítica harmonizada (ELSI W1, W2, W3)
============================================================
Lê as três ondas, aplica as regras do sr_core (harmonização, desfecho, exclusões),
ajusta a REFERÊNCIA CONDICIONAL (força esperada para idade, sexo e altura) só no
treino, define as caudas e grava a base analítica + o material do suplemento.

Uso (na raiz do projeto):
    python code/01_prepare.py
    $env:SR_SCENARIO="sens_ref_weight"; python code/01_prepare.py    (PowerShell)

Saídas em outputs/v<VERSION>/<cenário>/01_prepare/
  data/analytic_all.parquet             todas as linhas, com motivo de exclusão (desenho amostral completo)
  data/analytic.parquet                 amostra analítica (incluídos)
  tables/S1_harmonisation_rules.csv     regra de cada preditor
  tables/S1_harmonisation_audit.csv     distribuição de cada preditor por onda (incluídos)
  tables/S1_unexpected_codes.csv        valores fora do esperado nos dados brutos
  tables/S1_data_corrections.csv        correções aplicadas (ex.: equilíbrio da W3 registrado ÷10)
  tables/S2_flowchart.csv               fluxograma por onda
  tables/S2_missing_by_wave.csv         % ausente por preditor e onda
  tables/S2_included_vs_excluded.csv    incluídos x excluídos (sem medidas físicas / outros)
  tables/S3_norms_used.csv              normas internacionais usadas
  tables/S3_reference_model.csv         coeficientes da referência condicional (treino)
  tables/S3_reference_calibration.csv   % ≤P10, ≤P50, ≥P90 condicional por sexo, idade e altura (treino e W3)
  tables/S3_tail_prevalence.csv         prevalência das caudas por onda
  tables/S3_quartile_distribution.csv   quartis do modelo B por onda
  tables/outcome_summary.csv            desfecho por onda e sexo
  tables/data_dictionary.json           dicionário da base analítica
"""
import numpy as np
import pandas as pd

import sr_core as core

run = core.Run("01_prepare", __file__)
log = run.log
for w, p in core.RAW_FILES.items():
    run.add_input(p, f"elsi_w{w}")
run.add_input(core.NORMS_FILE, "norms")
CFG = core.CFG

# -----------------------------------------------------------------------------
# 1. harmonização por onda
# -----------------------------------------------------------------------------
frames = []
for w in (1, 2, 3):
    log.info(f"onda {w} ({core.WAVE_YEARS[w]}): lendo {core.RAW_FILES[w].name}")
    raw = core.read_wave(w)
    f = core.design_vars(raw, w)
    for feat in core.FEATURES:
        try:
            f[feat.name] = feat.builder(raw, w)
        except KeyError as e:
            log.warning(f"  W{w} {feat.name}: {e} -> ausente nesta onda")
            f[feat.name] = np.nan
    out = core.build_outcome(raw, w, f["age"], f["male"], f["height_cm"])
    f = pd.concat([f, out], axis=1)
    f["exclusion"], order = core.exclusion_flags(raw, w, f)
    log.info(f"  n={len(f):,} | incluídos={int((f['exclusion'] == 'included').sum()):,}")
    frames.append(f)

data = pd.concat(frames, ignore_index=True)
inc = data["exclusion"] == "included"
train_inc = inc & (data["split"] == "train")

# peso normalizado (média 1 dentro da onda, só entre incluídos) para juntar ondas
data["weight_norm"] = np.nan
for w in (1, 2, 3):
    m = inc & (data["wave"] == w)
    data.loc[m, "weight_norm"] = data.loc[m, "weight"] / data.loc[m, "weight"].mean()

# -----------------------------------------------------------------------------
# 2. desfechos derivados do treino: quartis internos e referência condicional
# -----------------------------------------------------------------------------
cuts = core.internal_quartile_cuts(data.loc[train_inc, "hgs_z"])
data["quartile_internal"] = core.assign_quartile(data["hgs_z"], cuts)
if CFG.noncompleters == "as_lowest":
    data.loc[data["grip_status"].isin(["unable", "risky"]), "quartile_internal"] = 0.0
data["quartile"] = data["quartile_intl"] if CFG.quartile_reference == "international" else data["quartile_internal"]
log.info(f"quartis internos (z, treino): {np.round(cuts, 3).tolist()} | referência do modelo B: {CFG.quartile_reference}")

ref = core.ConditionalReference()
fit_rows = train_inc & (data["grip_status"] == "measured")
ref.fit(data[fit_rows], data.loc[fit_rows, "grip_max_kg"], data.loc[fit_rows, "weight_norm"])
cond = ref.transform(data, data["grip_max_kg"])
data = pd.concat([data, cond], axis=1)
data = core.add_tails(data)
log.info(f"referência condicional: variáveis {CFG.reference_vars}, ajustada em {int(fit_rows.sum()):,} pessoas do treino; "
         f"sem z condicional entre incluídos medidos: {int((inc & (data['grip_status'] == 'measured') & data['z_cond'].isna()).sum())}")

analytic = data[inc].reset_index(drop=True)
run.save_data(data, "analytic_all", "todas as linhas das 3 ondas com motivo de exclusão")
run.save_data(analytic, "analytic", "amostra analítica (incluídos)")

# -----------------------------------------------------------------------------
# 3. fluxograma, incluídos x excluídos (S2 / Figura 1)
# -----------------------------------------------------------------------------
rows = []
for w in (1, 2, 3):
    d = data[data["wave"] == w]
    remaining = len(d)
    rows.append({"wave": f"W{w}", "step": "enrolled", "excluded": 0, "remaining": remaining})
    for r in order:
        k = int((d["exclusion"] == r).sum())
        remaining -= k
        rows.append({"wave": f"W{w}", "step": r, "excluded": k, "remaining": remaining})
run.save_table(pd.DataFrame(rows), "flowchart", "fluxograma por onda (critérios em ordem)", item="S2")

grp = np.select([inc, data["exclusion"] == "no_physical_measures", data["exclusion"].isin(["grip_refused", "grip_unable_or_risky"])],
                ["included", "excluded: no physical measures", "excluded: grip refused/unable"], "excluded: other reasons")
COMPARE = ["age", "male", "education_years", "income_quintile", "rural_childhood", "srh", "multimorbidity",
           "adl_n", "mobility_n", "memory_delayed", "bmi"]
cmp_rows = []
for (w, g), d in data.assign(group=grp).groupby(["wave", "group"]):
    r = {"wave": f"W{w}", "group": g, "n": len(d)}
    for v in COMPARE:
        r[v] = round(d[v].mean(), 2)
    r["region_northeast_pct"] = round(100 * (d["region"] == "Northeast").mean(), 1)
    r["urban_pct"] = round(100 * d["urban"].mean(), 1)
    cmp_rows.append(r)
run.save_table(pd.DataFrame(cmp_rows), "included_vs_excluded", "médias (não ponderadas) por grupo de inclusão e onda", item="S2")

# -----------------------------------------------------------------------------
# 4. regras, auditoria e ausentes (S1, S2)
# -----------------------------------------------------------------------------
rules = pd.DataFrame([{"feature": f.name, "label": f.label, "domain": f.domain, "layer": f.layer,
                       "type": f.kind, "role": f.role, "harmonisation_rule": f.rule} for f in core.FEATURES])
run.save_table(rules, "harmonisation_rules", "regra de harmonização de cada preditor", item="S1")

audit = []
for f in core.FEATURES:
    for w in (1, 2, 3):
        s = analytic.loc[analytic["wave"] == w, f.name]
        r = {"feature": f.name, "domain": f.domain, "wave": f"W{w}", "n_valid": int(s.notna().sum()),
             "pct_missing": round(100 * s.isna().mean(), 1)}
        if f.kind in ("cont", "ord"):
            r.update(mean=round(s.mean(), 2), sd=round(s.std(), 2), p50=s.median(), min=s.min(), max=s.max())
        else:
            vc = s.value_counts(normalize=True)
            r["distribution"] = "; ".join(f"{k}: {100 * v:.1f}%" for k, v in vc.items())
        audit.append(r)
audit = pd.DataFrame(audit)
run.save_table(audit, "harmonisation_audit", "distribuição de cada preditor por onda (incluídos)", item="S1")

unexpected = pd.DataFrame(core.AUDIT, columns=["wave", "variable", "unexpected_values", "n"])
unexpected["unexpected_values"] = unexpected["unexpected_values"].astype(str)
unexpected = unexpected.drop_duplicates().reset_index(drop=True)
run.save_table(unexpected, "unexpected_codes", "valores fora do esperado nos dados brutos", item="S1")
if len(unexpected):
    log.warning(f"{len(unexpected)} variável×onda com valores inesperados (ver S1_unexpected_codes.csv):\n"
                + unexpected.to_string(index=False))

corr = pd.DataFrame(core.CORRECTIONS, columns=["wave", "variable", "rule", "n_corrected"]).drop_duplicates()
run.save_table(corr, "data_corrections", "correções aplicadas aos dados brutos", item="S1")
if len(corr):
    log.warning("correções aplicadas:\n" + corr.to_string(index=False))

miss = audit.pivot(index=["feature", "domain"], columns="wave", values="pct_missing").reset_index()
run.save_table(miss, "missing_by_wave", "% ausente por preditor e onda (incluídos)", item="S2")
high_miss = miss[(miss[["W1", "W2", "W3"]] > 15).any(axis=1)]
if len(high_miss):
    log.warning("preditores com > 15% ausente em alguma onda:\n" + high_miss.to_string(index=False))

# -----------------------------------------------------------------------------
# 5. desfechos: normas, referência condicional, caudas (S3)
# -----------------------------------------------------------------------------
norms = core.load_norms(CFG.norm)
run.save_table(norms.drop(columns="age_mid"), "norms_used", f"normas Tomkinson 2025 ({CFG.norm})", item="S3")
run.save_table(ref.table(), "reference_model", "referência condicional: média e |resíduo| (kg) por grupo", item="S3")

# calibração: na referência ideal, 10% ≤P10, 50% ≤P50 e 10% ≥P90 em TODA combinação de sexo, idade e altura
cal = analytic[analytic["centile_cond"].notna()].copy()
cal["sex"] = np.where(cal["male"] == 1, "male", "female")
cal["age_band"] = pd.cut(cal["age"], [49, 59, 69, 79, 120], labels=["50–59", "60–69", "70–79", "80+"])
tert = {s: cal.loc[(cal["sex"] == s) & (cal["split"] == "train"), "height_cm"].quantile([1 / 3, 2 / 3]).to_numpy()
        for s in ("male", "female")}
cal["height_tertile"] = [("T1", "T2", "T3")[int(np.searchsorted(tert[s], h, side="right"))] for s, h in zip(cal["sex"], cal["height_cm"])]
cal_rows = []
for keys, d in cal.groupby(["split", "sex", "age_band", "height_tertile"], observed=True):
    wgt = d["weight_norm"]
    cal_rows.append({"split": keys[0], "sex": keys[1], "age_band": keys[2], "height_tertile": keys[3], "n": len(d),
                     "pct_le_P10": round(100 * np.average(d["centile_cond"] <= 10, weights=wgt), 1),
                     "pct_le_P50": round(100 * np.average(d["centile_cond"] <= 50, weights=wgt), 1),
                     "pct_ge_P90": round(100 * np.average(d["centile_cond"] >= 90, weights=wgt), 1)})
cal_t = pd.DataFrame(cal_rows)
run.save_table(cal_t, "reference_calibration", "calibração da referência condicional (ideal 10 / 50 / 10)", item="S3")
for sp in ("train", "test"):
    c = cal_t[cal_t["split"] == sp]
    log.info(f"calibração da referência ({sp}): % ≤P10 entre células {c['pct_le_P10'].min()}–{c['pct_le_P10'].max()} | "
             f"% ≥P90 {c['pct_ge_P90'].min()}–{c['pct_ge_P90'].max()} (ideal 10)")

tails = [c for c in core.TAILS]
prev = analytic.groupby("wave")[tails].agg(["mean", "sum"])
prev.columns = [f"{a}_{'pct' if b == 'mean' else 'n'}" for a, b in prev.columns]
for c in prev.columns:
    if c.endswith("_pct"):
        prev[c] = (100 * prev[c]).round(1)
prev = prev.reset_index()
run.save_table(prev, "tail_prevalence", "prevalência das caudas por onda (incluídos)", item="S3")
log.info("caudas por onda:\n" + prev.to_string(index=False))

summ = (data.assign(sex=np.where(data["male"] == 1, "male", "female"))
        .groupby(["wave", "sex"])
        .apply(lambda d: pd.Series({
            "n_total": len(d),
            "grip_measured": int((d["grip_status"] == "measured").sum()),
            "grip_unable_or_risky": int(d["grip_status"].isin(["unable", "risky"]).sum()),
            "grip_refused": int((d["grip_status"] == "refused").sum()),
            "no_physical_measures": int(d["grip_status"].isin(["na", "missing"]).sum()),
            "n_included": int((d["exclusion"] == "included").sum()),
            "grip_kg_mean": round(d.loc[d["exclusion"] == "included", "grip_max_kg"].mean(), 1),
            "z_intl_mean": round(d.loc[d["exclusion"] == "included", "hgs_z"].mean(), 2),
            "centile_intl_median": round(d.loc[d["exclusion"] == "included", "hgs_centile"].median(), 1),
            "z_cond_mean": round(d.loc[d["exclusion"] == "included", "z_cond"].mean(), 2),
        }), include_groups=False)
        .reset_index())
run.save_table(summ, "outcome_summary", "desfecho por onda e sexo (incluídos)")
for _, r in summ.iterrows():
    log.info(f"  W{r['wave']} {r['sex']:6s}: incluídos {int(r['n_included']):,} | força {r['grip_kg_mean']} kg | "
             f"z intl {r['z_intl_mean']} | centil intl mediano {r['centile_intl_median']} | z cond {r['z_cond_mean']}")

d_inc = data[inc]
quart = pd.crosstab([d_inc["wave"]], d_inc["quartile"].map(dict(enumerate(core.QUARTILE_LABELS))))
quart_pct = (100 * quart.div(quart.sum(axis=1), axis=0)).round(1).add_suffix(" %")
quart = pd.concat([quart, quart_pct], axis=1).reset_index()
run.save_table(quart, "quartile_distribution", f"distribuição dos quartis ({CFG.quartile_reference}) por onda", item="S3")

# -----------------------------------------------------------------------------
# 6. dicionário da base
# -----------------------------------------------------------------------------
dictionary = {
    "design": {"wave": "ELSI wave (1–3)", "row_id": "row id within wave (no person id across waves)",
               "cluster": "primary sampling unit, prefixed by wave", "stratum": "stratum, prefixed by wave",
               "weight": "peso_calibrado (W1 relative; W2/W3 expansion)",
               "weight_norm": "weight / mean(weight) within wave (included only)",
               "split": f"train = waves {CFG.train_waves}; test = waves {CFG.test_waves}"},
    "outcome": {"grip_max_kg": "maximum valid trial of mf27–mf29 (kg)",
                "grip_status": "measured | unable | risky | refused | na | missing",
                "hgs_hand_left": "mf26: test done with the left hand",
                "hgs_value": f"{CFG.norm} handgrip ({'kg' if CFG.norm == 'absolute' else 'kg/m²'})",
                "grip_adj_kg": "max grip × protocol factor (ELSI protocol -> norms' reference protocol)",
                "hgs_z": "z relative to Tomkinson 2025 norms (age-interpolated), after protocol adjustment",
                "hgs_centile": "100 × Φ(hgs_z)",
                "quartile_intl": "0–3: international quartile of hgs_centile (<P25, P25–49, P50–74, ≥P75)",
                "quartile_internal": "0–3: quartile of hgs_z with cut points from the training waves",
                "quartile": f"target of model B ({CFG.quartile_reference})",
                "grip_expected_kg": f"expected grip given {CFG.reference_vars} (ELSI training reference)",
                "grip_sd_kg": "conditional SD of grip (scale model)",
                "centile_cond": "conditional centile: position among people of same age, sex and height",
                "z_cond": "Φ⁻¹(centile_cond)",
                "tail_cond_low": f"centile_cond ≤ {CFG.cond_low}", "tail_cond_high": f"centile_cond ≥ {CFG.cond_high}",
                "tail_intl_low": f"hgs_centile < {CFG.intl_low}", "tail_intl_high": f"hgs_centile ≥ {CFG.intl_high}"},
    "features": {f.name: {"label": f.label, "domain": f.domain, "layer": f.layer, "type": f.kind, "role": f.role,
                          "rule": f.rule} for f in core.FEATURES},
    "exclusion": {"order": order},
}
run.save_json(dictionary, "data_dictionary", "dicionário da base analítica")

run.finish(notes=f"n analítico = {len(analytic):,} (train {int((analytic['split'] == 'train').sum()):,}; "
                 f"test {int((analytic['split'] == 'test').sum()):,})")

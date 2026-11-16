"""
08_cutoffs.py — qual critério de força baixa discrimina melhor a saúde real?
===========================================================================
Validade concorrente (transversal) de cinco critérios de força de preensão baixa contra desfechos de
saúde que NÃO entram nos modelos de ML, com o desenho amostral do ELSI.

Critérios (pontos de corte usuais):
  EWGSOP2      força máxima < 27 kg (homens) / < 16 kg (mulheres)
  FNIH         força máxima < 26 kg / < 16 kg
  FNIH-BMI     força / IMC < 1,0 / < 0,56
  Intl <P10    centil internacional (Tomkinson 2025) < 10
  Cond ≤P10    centil condicional (idade, sexo e altura; referência do ELSI) ≤ 10

Mesma prevalência: cada escore contínuo (força absoluta, força/IMC, centil internacional, centil condicional)
é cortado, dentro de cada sexo e onda, na prevalência ponderada do EWGSOP2. É a comparação justa: um corte
mais extremo sempre mostra OR maior.

Desfechos (12 meses): internação ≥ 24 h (u42) e atendimento de urgência (u49) [principais];
≥ 6 consultas médicas (u6), saúde autoavaliada ruim/muito ruim, qualquer dificuldade nas AVD,
marcha < 0,8 m/s [secundários].

Métricas: prevalência e perfil de quem cada critério classifica; sobreposição; OR ajustado por idade, idade² e
sexo (+ onda nas ondas empilhadas); ganho de AUC sobre o modelo só com idade e sexo; reclassificação líquida
(NRI) do condicional contra cada critério na mesma prevalência; discordantes (só EWGSOP2 × só condicional ×
ambos, contra ambos normais), no total e dentro de faixas etárias (50–64, 65–74, ≥75; grupo com < 30 pessoas
na faixa não é estimado). Estimativas ponderadas; IC 95% por bootstrap de conglomerados (UPA dentro do
estrato). Análise principal: W3; apoio: W1–W3 empilhadas (peso normalizado dentro da onda).

Lê só o 01 (base analítica) e os .dta (u42, u49, u6). Não depende de 02–07.
Uso (na raiz):  .venv\\Scripts\\python code\\08_cutoffs.py         (B_CUT=500 reamostras; SR_FAST=1 -> 30)
"""
import os
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import sr_core as core

warnings.filterwarnings("ignore")
run = core.Run("08_cutoffs", __file__)
log = run.log
up1 = run.require("01_prepare")
B = int(os.environ.get("B_CUT", 30 if core.FAST else 500))

# -----------------------------------------------------------------------------
# dados
# -----------------------------------------------------------------------------
ALL = pd.read_parquet(up1 / "data" / "analytic_all.parquet")
D = ALL[(ALL["exclusion"] == "included") & ALL["grip_max_kg"].notna() & ALL["centile_cond"].notna()
        & ALL["hgs_centile"].notna() & ALL["bmi"].notna()].copy()
parts = []
for w in (1, 2, 3):
    raw = core.read_wave(w)
    run.add_input(core.RAW_FILES[w], f"elsi_w{w}")
    o = raw[["u42", "u49", "u6"]].copy()
    o["row_id"] = [f"W{w}_{i}" for i in range(len(raw))]
    parts.append(o)
D = D.merge(pd.concat(parts), on="row_id", how="left", validate="1:1")

OUTCOMES = {  # nome: (rótulo, principal?)
    "hosp": ("Hospitalisation ≥24 h, past 12 months", True),
    "urgent": ("Urgent care, past 12 months", True),
    "visits6": ("≥6 medical visits, past 12 months", False),
    "poor_srh": ("Poor or very poor self-rated health", False),
    "adl_any": ("Any ADL difficulty", False),
    "slow_gait": ("Gait speed <0.8 m/s", False),
}
D["hosp"] = D["u42"].where(D["u42"].isin([0, 1]))
D["urgent"] = D["u49"].where(D["u49"].isin([0, 1]))
D["visits6"] = (D["u6"] >= 6).astype(float).where(D["u6"].notna() & (D["u6"] < 88))   # 88/99/888/999 = NS/NR
D["poor_srh"] = (D["srh"] >= 4).astype(float).where(D["srh"].notna())
D["adl_any"] = (D["adl_n"] > 0).astype(float).where(D["adl_n"].notna())
D["slow_gait"] = (D["gait_speed"] < 0.8).astype(float).where(D["gait_speed"].notna())

# -----------------------------------------------------------------------------
# critérios
# -----------------------------------------------------------------------------
men = D["male"] == 1
D["grip_bmi"] = D["grip_max_kg"] / D["bmi"]
USUAL = {
    "EWGSOP2": ("Max grip <27 kg men / <16 kg women", np.where(men, D["grip_max_kg"] < 27, D["grip_max_kg"] < 16)),
    "FNIH": ("Max grip <26 kg men / <16 kg women", np.where(men, D["grip_max_kg"] < 26, D["grip_max_kg"] < 16)),
    "FNIH-BMI": ("Grip/BMI <1.0 men / <0.56 women", np.where(men, D["grip_bmi"] < 1.0, D["grip_bmi"] < 0.56)),
    "Intl <P10": ("International centile <10 (Tomkinson 2025)", D["hgs_centile"] < 10),
    "Cond ≤P10": ("Conditional centile ≤10 (age, sex, height)", D["centile_cond"] <= 10),
}
for k, (_, v) in USUAL.items():
    D[f"u:{k}"] = np.asarray(v, float)

SCORES = {"Absolute grip": "grip_max_kg", "Grip/BMI": "grip_bmi", "International centile": "hgs_centile",
          "Conditional centile": "centile_cond"}
EQ_REF = "Absolute grip"            # na mesma prevalência, EWGSOP2 = força absoluta
for k, col in SCORES.items():
    D[f"e:{k}"] = np.nan
for (w, m), g in D.groupby(["wave", "male"]):
    p = np.average(g["u:EWGSOP2"], weights=g["weight"])
    for k, col in SCORES.items():
        thr = core.weighted_quantile(g[col], g["weight"], [p])[0]
        D.loc[g.index, f"e:{k}"] = (g[col] <= thr).astype(float) if k != EQ_REF else g["u:EWGSOP2"]
CRIT = [f"u:{k}" for k in USUAL] + [f"e:{k}" for k in SCORES]
CRIT_LAB = {**{f"u:{k}": k for k in USUAL}, **{f"e:{k}": k for k in SCORES}}
COND_EQ = "e:Conditional centile"
D["disc"] = np.select([(D["u:EWGSOP2"] == 1) & (D[COND_EQ] == 1), (D["u:EWGSOP2"] == 1) & (D[COND_EQ] == 0),
                       (D["u:EWGSOP2"] == 0) & (D[COND_EQ] == 1)], ["Both", "EWGSOP2 only", "Conditional only"], "Neither")
DISC_GROUPS = ["EWGSOP2 only", "Conditional only", "Both"]
# discordantes dentro de faixas etárias: os grupos "só EWGSOP2" (mais velhos) e "só condicional" (mais jovens) quase não
# se sobrepõem em idade; dentro de cada faixa o ajuste por idade não depende de extrapolação
AGE_BANDS = [("50–64", 50, 65), ("65–74", 65, 75), ("≥75", 75, 200)]
MIN_BAND_N = 30                     # grupo com menos pessoas na faixa não é estimado
D["age80"] = (D["age"] >= 80).astype(float)
D["band"] = pd.cut(D["age"], [b[1] for b in AGE_BANDS] + [200], right=False, labels=[b[0] for b in AGE_BANDS]).astype(str)
log.info(f"amostra: {len(D):,} (W1 {int((D.wave == 1).sum()):,} | W2 {int((D.wave == 2).sum()):,} | W3 {int((D.wave == 3).sum()):,}) | B = {B}")


# -----------------------------------------------------------------------------
# ferramentas
# -----------------------------------------------------------------------------
def logit(X, y, w, iters=30):
    """Regressão logística ponderada (IRLS). Devolve coeficientes e probabilidades ajustadas."""
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        eta = np.clip(X @ b, -30, 30)
        p = 1 / (1 + np.exp(-eta))
        v = w * p * (1 - p) + 1e-10
        step = np.linalg.solve((X * v[:, None]).T @ X + 1e-8 * np.eye(X.shape[1]), X.T @ (w * (y - p)))
        b += step
        if np.max(np.abs(step)) < 1e-8:
            break
    return b, 1 / (1 + np.exp(-np.clip(X @ b, -30, 30)))


def wauc(y, p, w):
    return roc_auc_score(y, p, sample_weight=w)


def base_design(g, pooled):
    a = (g["age"].to_numpy() - 70) / 10
    cols = [np.ones(len(g)), a, a ** 2, g["male"].to_numpy(float)]
    if pooled:
        cols += [(g["wave"].to_numpy() == w).astype(float) for w in (2, 3)]
    return np.column_stack(cols)


def compute(g, pooled):
    """Todas as métricas num conjunto de dados (amostra original ou reamostra). Devolve dict plano."""
    out = {}
    w_all = g["weight_norm"].to_numpy(float)
    Xb_all = base_design(g, pooled)
    for o in OUTCOMES:
        y_all = g[o].to_numpy(float)
        ok = ~np.isnan(y_all)
        y, w, Xb = y_all[ok], w_all[ok], Xb_all[ok]
        if y.sum() < 5 or (1 - y).sum() < 5:
            continue
        _, pb = logit(Xb, y, w)
        auc0 = wauc(y, pb, w)
        Xc = Xb[:, [0] + ([4, 5] if pooled else [])]                          # sem ajuste (só intercepto; empilhado: + onda)
        out[(o, "base", "auc")] = auc0
        ev, ne = y == 1, y == 0
        for c in CRIT:
            f = g[c].to_numpy(float)[ok]
            b, pf = logit(np.column_stack([Xb, f]), y, w)
            out[(o, c, "or")] = np.exp(b[-1])
            bc, _ = logit(np.column_stack([Xc, f]), y, w)
            out[(o, c, "or_crude")] = np.exp(bc[-1])
            out[(o, c, "dauc")] = wauc(y, pf, w) - auc0
            out[(o, c, "p_event")] = np.average(f[ev], weights=w[ev])          # P(classificado | desfecho)
            out[(o, c, "p_nonevent")] = np.average(f[ne], weights=w[ne])
        for c in CRIT:
            if c.startswith("e:") and c != COND_EQ:                            # NRI e ΔAUC pareados: condicional − outro
                out[(o, c, "nri_event")] = out[(o, COND_EQ, "p_event")] - out[(o, c, "p_event")]
                out[(o, c, "nri_nonevent")] = out[(o, c, "p_nonevent")] - out[(o, COND_EQ, "p_nonevent")]
                out[(o, c, "nri")] = out[(o, c, "nri_event")] + out[(o, c, "nri_nonevent")]
                out[(o, c, "dauc_diff")] = out[(o, COND_EQ, "dauc")] - out[(o, c, "dauc")]
        grp = g["disc"].to_numpy()[ok]
        dum = np.column_stack([(grp == k).astype(float) for k in DISC_GROUPS])
        b, _ = logit(np.column_stack([Xb, dum]), y, w)
        bc, _ = logit(np.column_stack([Xc, dum]), y, w)
        for j, k in enumerate(DISC_GROUPS):
            out[(o, f"disc:{k}", "or")] = np.exp(b[Xb.shape[1] + j])
            out[(o, f"disc:{k}", "or_crude")] = np.exp(bc[Xc.shape[1] + j])
        band = g["band"].to_numpy()[ok]
        for bl, _, _ in AGE_BANDS:
            mb = band == bl
            ks = [k for k in DISC_GROUPS if (ELIG[pooled].get((bl, k), 0) >= MIN_BAND_N) and np.sum(mb & (grp == k)) > 0
                  and y[mb & (grp == k)].sum() > 0]
            if not ks or y[mb].sum() < 5:
                continue
            dumb = np.column_stack([(grp[mb] == k).astype(float) for k in ks])
            b, _ = logit(np.column_stack([Xb[mb], dumb]), y[mb], w[mb])
            for j, k in enumerate(ks):
                out[(o, f"band:{bl}:{k}", "or")] = np.exp(b[Xb.shape[1] + j])
    return out


def with_ci(sample_name, g, pooled):
    t0 = time.time()
    est = compute(g, pooled)
    idx = core.cluster_boot_indices(g["stratum"], g["cluster"], B, seed=core.SEED)
    boots = []
    for i, ix in enumerate(idx):
        try:
            boots.append(compute(g.iloc[ix], pooled))
        except (ValueError, np.linalg.LinAlgError):
            continue
        if (i + 1) % 100 == 0:
            log.info(f"  [{sample_name}] bootstrap {i + 1}/{B} ({(time.time() - t0) / 60:.1f} min)")
    bt = pd.DataFrame(boots)
    rows = []
    for k, v in est.items():
        s = bt[k].dropna() if k in bt else pd.Series(dtype=float)
        lo, hi = (s.quantile(0.025), s.quantile(0.975)) if len(s) else (np.nan, np.nan)
        rows.append({"sample": sample_name, "outcome": k[0], "criterion": k[1], "metric": k[2], "estimate": v,
                     "ci_low": lo, "ci_high": hi, "n_boot": len(s)})
    log.info(f"  [{sample_name}] {len(bt)} reamostras válidas em {(time.time() - t0) / 60:.1f} min")
    return pd.DataFrame(rows)


def svy(y, dom, prop=True):
    return core.svy_mean(y, D["weight"], D["stratum"], D["cluster"], domain=dom, proportion=prop)


def fmt(r, k=100, d=1):
    return f"{k * r['est']:.{d}f} ({k * r['ci_low']:.{d}f}–{k * r['ci_high']:.{d}f})" if r["n"] else "—"


# -----------------------------------------------------------------------------
# 1. desfechos por onda e definição/prevalência dos critérios
# -----------------------------------------------------------------------------
rows = []
for o, (lab, prim) in OUTCOMES.items():
    r = {"outcome": o, "label": lab, "primary": prim}
    for w in (1, 2, 3):
        r[f"W{w} %, survey-weighted (95% CI)"] = fmt(svy(D[o], D["wave"] == w))
        r[f"W{w} n"] = int(D.loc[D["wave"] == w, o].notna().sum())
    rows.append(r)
run.save_table(pd.DataFrame(rows), "C1_outcomes_by_wave", "desfechos de saúde por onda (desenho amostral)")

rows = []
for c in CRIT:
    kind = "usual cut-off" if c.startswith("u:") else "equal prevalence (EWGSOP2, within sex and wave)"
    rule = USUAL[CRIT_LAB[c]][0] if c.startswith("u:") else f"lowest {SCORES[CRIT_LAB[c]]} at the EWGSOP2 prevalence"
    r = {"criterion": CRIT_LAB[c], "set": kind, "rule": rule}
    for w in (1, 2, 3):
        r[f"W{w} prevalence, % (95% CI)"] = fmt(svy(D[c], D["wave"] == w))
    w3 = (D["wave"] == 3) & (D[c] == 1)
    for v, lab, k, d in (("age", "age, mean", 1, 1), ("height_cm", "height cm, mean", 1, 1), ("male", "men, %", 100, 1),
                         ("age80", "aged ≥80, %", 100, 1), ("bmi", "BMI, mean", 1, 1)):
        r[f"W3 flagged: {lab}"] = fmt(svy(D[v], w3, prop=False), k=k, d=d)
    r["W3 flagged: n"] = int(w3.sum())
    rows.append(r)
run.save_table(pd.DataFrame(rows), "C2_criteria_profile", "critérios: regra, prevalência por onda e perfil de quem é classificado (W3)")

# sobreposição (W3, ponderada): Jaccard e kappa
w3 = D[D["wave"] == 3]
rows = []
for a in [f"u:{k}" for k in USUAL] + [COND_EQ]:
    for b_ in [f"u:{k}" for k in USUAL] + [COND_EQ]:
        x, y, ww = w3[a].to_numpy(), w3[b_].to_numpy(), w3["weight"].to_numpy()
        both, either = np.sum(ww * (x * y)), np.sum(ww * np.maximum(x, y))
        po = np.sum(ww * (x == y)) / ww.sum()
        px, py = np.average(x, weights=ww), np.average(y, weights=ww)
        pe = px * py + (1 - px) * (1 - py)
        rows.append({"a": CRIT_LAB[a] + (" (equal prev.)" if a == COND_EQ else ""), "b": CRIT_LAB[b_] + (" (equal prev.)" if b_ == COND_EQ else ""),
                     "jaccard": both / either if either else np.nan, "kappa": (po - pe) / (1 - pe) if pe < 1 else np.nan})
run.save_table(pd.DataFrame(rows), "C3_overlap", "sobreposição entre critérios na W3 (Jaccard e kappa, ponderados)")

# discordantes: perfil
rows = []
for sname, dom in (("W3", D["wave"] == 3), ("W1–W3", pd.Series(True, index=D.index))):
    for k in ["Neither"] + DISC_GROUPS:
        m = dom & (D["disc"] == k)
        r = {"sample": sname, "group": k, "n": int(m.sum()), "% of sample (weighted)": 100 * np.average(D.loc[dom, "disc"] == k, weights=D.loc[dom, "weight_norm"])}
        for v in ("age", "height_cm", "weight_kg", "bmi", "male", "grip_max_kg", "centile_cond", "hgs_centile"):
            r[v] = np.average(D.loc[m, v], weights=D.loc[m, "weight_norm"]) if m.any() else np.nan
        for o in OUTCOMES:
            mm = m & D[o].notna()
            r[f"{o} %"] = 100 * np.average(D.loc[mm, o], weights=D.loc[mm, "weight_norm"]) if mm.any() else np.nan
        rows.append(r)
run.save_table(pd.DataFrame(rows).round(3), "C6_discordant_profile",
               "discordantes EWGSOP2 × condicional na mesma prevalência: perfil e % de cada desfecho (ponderado)")

# -----------------------------------------------------------------------------
# 2. associações com IC (bootstrap de conglomerados)
# -----------------------------------------------------------------------------
ELIG = {False: D[D["wave"] == 3].groupby(["band", "disc"]).size().to_dict(), True: D.groupby(["band", "disc"]).size().to_dict()}
rows = []
for sname, pooled in (("W3", False), ("W1–W3", True)):
    dom = D if pooled else D[D["wave"] == 3]
    for bl, _, _ in AGE_BANDS:
        for k in ["Neither"] + DISC_GROUPS:
            m = (dom["band"] == bl) & (dom["disc"] == k)
            r = {"sample": sname, "age band": bl, "group": k, "n": int(m.sum()), "estimated": ELIG[pooled].get((bl, k), 0) >= MIN_BAND_N or k == "Neither"}
            for v in ("age", "height_cm"):
                r[v] = np.average(dom.loc[m, v], weights=dom.loc[m, "weight_norm"]) if m.any() else np.nan
            for o in OUTCOMES:
                mm = m & dom[o].notna()
                r[f"{o} %"] = 100 * np.average(dom.loc[mm, o], weights=dom.loc[mm, "weight_norm"]) if mm.any() else np.nan
            rows.append(r)
run.save_table(pd.DataFrame(rows).round(2), "C7_discordant_by_age_profile", "discordantes por faixa etária: n, idade, altura e % de cada desfecho")
res = pd.concat([with_ci("W3", D[D["wave"] == 3].reset_index(drop=True), pooled=False),
                 with_ci("W1–W3", D.reset_index(drop=True), pooled=True)], ignore_index=True)
res["outcome_label"] = res["outcome"].map(lambda o: OUTCOMES[o][0])
res["primary_outcome"] = res["outcome"].map(lambda o: OUTCOMES[o][1])
res["set"] = res["criterion"].map(lambda c: "usual" if c.startswith("u:") else "equal prevalence" if c.startswith("e:")
                                  else "discordant" if c.startswith("disc:") else "discordant by age" if c.startswith("band:") else "base")
res["criterion_label"] = res["criterion"].map(lambda c: CRIT_LAB.get(c, c.replace("disc:", "").split(":")[-1]))
res["age_band"] = res["criterion"].map(lambda c: c.split(":")[1] if c.startswith("band:") else "")
run.save_table(res, "C_all_long", "todas as estimativas (formato longo) com IC 95% por bootstrap de conglomerados")


def ci_txt(r, d=2, sign=False):
    f = f"{{:{'+' if sign else ''}.{d}f}}"
    return f"{f.format(r['estimate'])} ({r['ci_low']:.{d}f} to {r['ci_high']:.{d}f})"


assoc = []
for (s, o, c), g in res[res["set"].isin(["usual", "equal prevalence"])].groupby(["sample", "outcome", "criterion"], sort=False):
    m = g.set_index("metric")
    assoc.append({"sample": s, "outcome": OUTCOMES[o][0], "set": g["set"].iloc[0], "criterion": CRIT_LAB[c],
                  "OR crude (95% CI)": ci_txt(m.loc["or_crude"]), "OR adjusted (95% CI)": ci_txt(m.loc["or"]), "ΔAUC vs age+sex (95% CI)": ci_txt(m.loc["dauc"], 3, True),
                  "% flagged among events": round(100 * m.loc["p_event", "estimate"], 1),
                  "% flagged among non-events": round(100 * m.loc["p_nonevent", "estimate"], 1)})
run.save_table(pd.DataFrame(assoc), "C4_association", "OR ajustado e ganho de AUC de cada critério, por desfecho (W3 e W1–W3)")

nri = []
for (s, o, c), g in res[res["metric"].isin(["nri", "nri_event", "nri_nonevent", "dauc_diff"])].groupby(["sample", "outcome", "criterion"], sort=False):
    m = g.set_index("metric")
    nri.append({"sample": s, "outcome": OUTCOMES[o][0], "comparator": CRIT_LAB[c],
                "NRI events (95% CI)": ci_txt(m.loc["nri_event"], 3, True), "NRI non-events (95% CI)": ci_txt(m.loc["nri_nonevent"], 3, True),
                "NRI total (95% CI)": ci_txt(m.loc["nri"], 3, True), "ΔAUC conditional − comparator (95% CI)": ci_txt(m.loc["dauc_diff"], 3, True)})
run.save_table(pd.DataFrame(nri), "C5_nri", "reclassificação do centil condicional contra cada critério, na mesma prevalência")

disc = []
for (s, o), g in res[res["set"] == "discordant"].groupby(["sample", "outcome"], sort=False):
    r = {"sample": s, "outcome": OUTCOMES[o][0]}
    for _, x in g[g["metric"] == "or"].iterrows():
        r[f"{x['criterion_label']}: OR vs neither (95% CI)"] = ci_txt(x)
    for _, x in g[g["metric"] == "or_crude"].iterrows():
        r[f"{x['criterion_label']}: crude OR vs neither (95% CI)"] = ci_txt(x)
    disc.append(r)
run.save_table(pd.DataFrame(disc), "C6_discordant_or", "discordantes: OR ajustado contra quem é normal pelos dois critérios")

disc_b = []
for (s_, o, bl), g in res[res["set"] == "discordant by age"].groupby(["sample", "outcome", "age_band"], sort=False):
    r = {"sample": s_, "outcome": OUTCOMES[o][0], "age band": bl}
    for k in DISC_GROUPS:
        x = g[g["criterion_label"] == k]
        r[f"{k}: OR vs neither (95% CI)"] = ci_txt(x.iloc[0]) if len(x) else "not estimated (n < %d)" % MIN_BAND_N
    disc_b.append(r)
run.save_table(pd.DataFrame(disc_b), "C7_discordant_by_age_or",
               "discordantes dentro de faixas etárias: OR ajustado (idade, idade², sexo[, onda]) contra quem é normal pelos dois critérios")
for o in ("hosp", "urgent", "visits6"):
    for bl, _, _ in AGE_BANDS:
        x = res[(res["sample"] == "W1–W3") & (res["outcome"] == o) & (res["age_band"] == bl)]
        log.info(f"W1–W3 · {OUTCOMES[o][0][:28]:28s} · {bl:6s} " + " | ".join(f"{r['criterion_label']}: {ci_txt(r)}" for _, r in x.iterrows()))

# resumo no log
for o in ("hosp", "urgent"):
    log.info(f"W3 · {OUTCOMES[o][0]}")
    for c in CRIT:
        m = res[(res["sample"] == "W3") & (res["outcome"] == o) & (res["criterion"] == c)].set_index("metric")
        if len(m):
            log.info(f"   {CRIT_LAB[c]:22s} {'(eq)' if c.startswith('e:') else '    '} OR {ci_txt(m.loc['or'])} | ΔAUC {ci_txt(m.loc['dauc'], 3, True)}")
    for k in DISC_GROUPS:
        m = res[(res["sample"] == "W3") & (res["outcome"] == o) & (res["criterion"] == f"disc:{k}")]
        if len(m):
            log.info(f"   discordante {k:18s} OR {ci_txt(m.iloc[0])}")

run.finish(notes=f"B={B}")

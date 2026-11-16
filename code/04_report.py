"""
04_report.py — tabelas (.csv + .docx) e figuras (.pdf + .png) do artigo
======================================================================
Lê 01, 02 e 03 (assinaturas conferidas) e produz:
  T1  características por onda, com o desenho amostral (peso, estrato, UPA)
  T2  posição nacional frente às normas mundiais por sexo, idade, região, raça/cor,
      escolaridade e renda (média do z, % <P10, % <P25, % ≥P75; desenho amostral)
  T3  desempenho dos modelos (primários) — do 02
  T4  assimetria topo × base por domínio — do 03
  F2  centis da força no ELSI sobre as faixas das normas internacionais, por sexo
  F3  decomposição Shapley do R² (4 blocos de domínios) e a fatia não explicada
  F4  participação de cada domínio nos modelos da base e do topo
  F5  regressão quantílica: efeito ao longo da distribuição condicional
  S3  calibração da referência condicional

Uso (na raiz):  python code/04_report.py        ($env:SR_FAST="1" se 02/03 foram rápidos)
Saídas em outputs/v<VERSION>/<cenário>/04_report/  (tables/, figures/, docx/)
"""
import json
import warnings

import matplotlib
import numpy as np
import pandas as pd

import sr_core as core

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

warnings.filterwarnings("ignore")
run = core.Run("04_report", __file__)
log = run.log
up1, up2, up3 = run.require("01_prepare"), run.require("02_model"), run.require("03_explain")
P = core.PALETTE
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": P["ink2"], "axes.labelcolor": P["ink"], "xtick.color": P["ink2"], "ytick.color": P["ink2"],
                     "axes.grid": True, "grid.color": "#ecebe7", "grid.linewidth": 0.6, "legend.frameon": False})

ALL = pd.read_parquet(up1 / "data" / "analytic_all.parquet")
INC = ALL["exclusion"] == "included"
primary = json.loads((up2 / "tables" / "primary_models.json").read_text(encoding="utf-8"))
T3_DIR = up3 / "tables"


def svy(y, dom, prop=False):
    return core.svy_mean(y, ALL["weight"], ALL["stratum"], ALL["cluster"], domain=dom, proportion=prop)


def fmt(r, pct=False, d=1):
    k = 100 if pct else 1
    return f"{k * r['est']:.{d}f} ({k * r['ci_low']:.{d}f}–{k * r['ci_high']:.{d}f})" if r["n"] else "—"


def to_docx(tables: dict, name: str):
    """Grava várias tabelas num .docx (uma por seção, com título)."""
    from docx import Document
    from docx.shared import Pt
    doc = Document()
    st = doc.styles["Normal"]
    st.font.name, st.font.size = "Arial", Pt(8)
    for title, df in tables.items():
        doc.add_paragraph().add_run(title).bold = True
        t = doc.add_table(rows=1, cols=len(df.columns))
        t.style = "Table Grid"
        for j, c in enumerate(df.columns):
            t.rows[0].cells[j].text = str(c)
        for _, row in df.iterrows():
            cells = t.add_row().cells
            for j, v in enumerate(row):
                cells[j].text = "" if pd.isna(v) else str(v)
        doc.add_paragraph()
    p = run.dir / "docx" / f"{name}.docx"
    doc.save(p)
    run.save_file(p, "docx", f"tabelas {', '.join(tables)}")


# -----------------------------------------------------------------------------
# T1 — características por onda (desenho amostral)
# -----------------------------------------------------------------------------
ALL["age_band"] = pd.cut(ALL["age"], [49, 59, 69, 79, 120], labels=["50–59", "60–69", "70–79", "80+"])
ALL["edu_group"] = pd.cut(ALL["education_years"], [-1, 0, 4, 8, 11, 30], labels=["None", "1–4", "5–8", "9–11", "≥12"])
ALL["sex"] = np.where(ALL["male"] == 1, "Men", "Women")
CONT = [("age", "Age, years"), ("education_years", "Education, years"), ("grip_max_kg", "Handgrip, kg"),
        ("hgs_centile", "International centile (Tomkinson 2025)"), ("bmi", "BMI, kg/m²"), ("gait_speed", "Gait speed, m/s"),
        ("multimorbidity", "Chronic conditions, n")]
BINS = [("male", "Men"), ("partner", "Living with partner"), ("urban", "Urban"), ("rural_childhood", "Rural childhood"),
        ("childhood_hunger", "Childhood hunger"), ("heavy_work", "Heavy physical work most of life"),
        ("hypertension", "Hypertension"), ("diabetes", "Diabetes"), ("falls", "Fall in past year"), ("current_smoking", "Current smoking")]
CATS = [("region", "Region"), ("race", "Race/skin colour")]
t1 = []
for w in (1, 2, 3):
    dom = INC & (ALL["wave"] == w)
    col = f"W{w} {core.WAVE_YEARS[w]} (n={int(dom.sum()):,})"
    rows = [("n (unweighted)", f"{int(dom.sum()):,}")]
    rows += [(lab + ", mean (95% CI)", fmt(svy(ALL[v], dom))) for v, lab in CONT]
    rows += [(lab + ", % (95% CI)", fmt(svy(ALL[v], dom, True), pct=True)) for v, lab in BINS]
    for v, lab in CATS:
        for lv in sorted(ALL.loc[dom, v].dropna().unique()):
            rows.append((f"{lab}: {lv}, %", fmt(svy((ALL[v] == lv).astype(float).where(ALL[v].notna()), dom, True), pct=True)))
    t1.append(pd.Series(dict(rows), name=col))
t1 = pd.concat(t1, axis=1).reset_index().rename(columns={"index": "Characteristic"})
run.save_table(t1, "characteristics", "características por onda, ponderadas pelo desenho amostral", item="T1")

# -----------------------------------------------------------------------------
# T2 — posição nacional frente às normas (desenho amostral)
# -----------------------------------------------------------------------------
GROUPS = [("All", None), ("Sex", "sex"), ("Age", "age_band"), ("Region", "region"), ("Race/skin colour", "race"),
          ("Education", "edu_group"), ("Income quintile", "income_quintile")]
lt10 = (ALL["hgs_centile"] < 10).astype(float).where(ALL["hgs_centile"].notna())
lt25 = (ALL["hgs_centile"] < 25).astype(float).where(ALL["hgs_centile"].notna())
ge75 = (ALL["hgs_centile"] >= 75).astype(float).where(ALL["hgs_centile"].notna())
t2 = []
for w in (1, 2, 3):
    base = INC & (ALL["wave"] == w)
    for gname, gvar in GROUPS:
        levels = [None] if gvar is None else sorted(ALL.loc[base, gvar].dropna().unique(), key=str)
        for lv in levels:
            dom = base if gvar is None else base & (ALL[gvar] == lv)
            med = core.weighted_quantile(ALL.loc[dom, "hgs_centile"], ALL.loc[dom, "weight"], [0.5])[0]
            t2.append({"wave": f"W{w}", "group": gname, "level": "All" if lv is None else str(lv), "n": int(dom.sum()),
                       "mean_z_intl (95% CI)": fmt(svy(ALL["hgs_z"], dom), d=2), "median_centile": round(med, 1),
                       "% <P10 (95% CI)": fmt(svy(lt10, dom, True), pct=True),
                       "% <P25 (95% CI)": fmt(svy(lt25, dom, True), pct=True),
                       "% ≥P75 (95% CI)": fmt(svy(ge75, dom, True), pct=True),
                       "mean_z_cond (95% CI)": fmt(svy(ALL["z_cond"], dom), d=2)})
t2 = pd.DataFrame(t2)
run.save_table(t2, "national_position", "posição frente às normas mundiais por grupo e onda (desenho amostral)", item="T2")
for w in ("W1", "W2", "W3"):
    a = t2[(t2["wave"] == w) & (t2["group"] == "All")].iloc[0]
    log.info(f"{w}: centil mediano {a['median_centile']} | <P25 {a['% <P25 (95% CI)']} | ≥P75 {a['% ≥P75 (95% CI)']}")

# checagem sexo × idade (z internacional e condicional)
sa = []
for w in (1, 2, 3):
    for sx in ("Women", "Men"):
        for ab in ["50–59", "60–69", "70–79", "80+"]:
            dom = INC & (ALL["wave"] == w) & (ALL["sex"] == sx) & (ALL["age_band"] == ab)
            r, rc = svy(ALL["hgs_z"], dom), svy(ALL["z_cond"], dom)
            sa.append({"wave": f"W{w}", "sex": sx, "age_band": ab, "n": r["n"], "z_intl": r["est"], "z_intl_low": r["ci_low"],
                       "z_intl_high": r["ci_high"], "z_cond": rc["est"]})
run.save_table(pd.DataFrame(sa), "z_by_sex_age", "z internacional e condicional por sexo, idade e onda (desenho amostral)", item="T2")

# -----------------------------------------------------------------------------
# F2 — centis do ELSI sobre as normas internacionais
# -----------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3), sharey=False)
ages = np.arange(50, 91)
bands = np.arange(50, 95, 5)
GKG = "grip_adj_kg" if "grip_adj_kg" in ALL.columns else "grip_max_kg"   # força no protocolo das normas
for ax, (sex, is_m, title) in zip(axes, (("female", 0, "Women"), ("male", 1, "Men"))):
    cur = core.norm_curves(ages.astype(float), sex, core.CFG.norm)
    iP = {c: i for i, c in enumerate(core.PCT_COLS)}
    ax.fill_between(ages, cur[:, iP["P10"]], cur[:, iP["P90"]], color=P["norm_band"], lw=0, label="International P10–P90")
    ax.fill_between(ages, cur[:, iP["P30"]], cur[:, iP["P70"]], color="#d9d8d3", lw=0, label="International P30–P70")
    ax.plot(ages, cur[:, iP["P50"]], color=P["norm_line"], lw=1.5, ls="--", label="International median")
    d = ALL[INC & (ALL["male"] == is_m) & ALL["grip_max_kg"].notna()]
    mid, q10, q50, q90 = [], [], [], []
    for lo in bands:
        g = d[(d["age"] >= lo) & (d["age"] < lo + 5)] if lo < 90 else d[d["age"] >= 90]
        if len(g) < 30:
            continue
        q = core.weighted_quantile(g[GKG], g["weight_norm"], [0.1, 0.5, 0.9])
        mid.append(lo + 2.5); q10.append(q[0]); q50.append(q[1]); q90.append(q[2])
    ax.fill_between(mid, q10, q90, color=P["elsi"], alpha=0.18, lw=0, label="ELSI P10–P90")
    ax.plot(mid, q50, color=P["elsi"], lw=2, marker="o", ms=4, label="ELSI median")
    ax.set_title(title, loc="left", fontsize=9.5, color=P["ink"])
    ax.set_xlabel("Age, years")
    ax.set_xlim(50, 92)
axes[0].set_ylabel("Handgrip strength, kg (norms' protocol)")
axes[1].legend(loc="upper right", fontsize=7)
fig.suptitle("Older Brazilians against international handgrip norms (ELSI 2015–2024, survey-weighted)",
             x=0.01, ha="left", fontsize=10, color=P["ink"])
fig.tight_layout()
run.save_figure(fig, "elsi_vs_norms", "centis do ELSI sobre as normas internacionais", item="F2")
plt.close(fig)

# -----------------------------------------------------------------------------
# F3 — decomposição Shapley do R² em 4 blocos + não explicado
# -----------------------------------------------------------------------------
sh_path = T3_DIR / "F3_shapley_r2.csv"
if sh_path.exists():
    sh = pd.read_csv(sh_path)
    grp_of = {d: g for g, ds in core.DOMAIN_GROUPS.items() for d in ds}
    rows = []
    for task, lab in (("Acond", "Strength for body size\n(conditional)"), ("Aintl", "Strength vs world norms\n(international)")):
        s = sh[(sh["task"] == task) & sh["domain"].isin(core.DOMAINS)]
        if s.empty:
            continue
        g = s.assign(group=s["domain"].map(grp_of)).groupby("group")["shapley_r2_train"].sum()
        tot = sh[(sh["task"] == task) & (sh["domain"] == "TOTAL explained (R²)")]
        rows.append((lab, g, float(tot["shapley_r2_test"].iloc[0])))
    fig, ax = plt.subplots(figsize=(7.2, 2.5))
    for i, (lab, g, r2_test) in enumerate(rows):
        left = 0.0
        for j, grp in enumerate(core.DOMAIN_GROUPS):
            v = 100 * max(g.get(grp, 0), 0)
            ax.barh(i, v, left=left, color=P["groups"][j], edgecolor="white", linewidth=1.5, height=0.55,
                    label=grp if i == 0 else None)
            if v >= 2.5:
                ax.text(left + v / 2, i, f"{v:.0f}", ha="center", va="center", fontsize=7.5, color="white")
            left += v
        ax.barh(i, 100 - left, left=left, color=P["unexplained"], edgecolor="white", linewidth=1.5, height=0.55,
                label="Unexplained" if i == 0 else None)
        ax.text(left + (100 - left) / 2, i, f"Unexplained {100 - left:.0f}%  (W3 R² {100 * r2_test:.0f}%)",
                ha="center", va="center", fontsize=7.5, color=P["ink2"])
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.set_xlim(0, 100); ax.set_xlabel("% of variance (Shapley decomposition of R², training waves)")
    ax.invert_yaxis(); ax.grid(axis="y", visible=False)
    ax.legend(ncol=3, loc="lower left", bbox_to_anchor=(0, 1.02), fontsize=7)
    fig.tight_layout()
    run.save_figure(fig, "variance_decomposition", "decomposição Shapley do R² por bloco de domínios", item="F3")
    plt.close(fig)

# -----------------------------------------------------------------------------
# F4 / T4 — assimetria topo × base
# -----------------------------------------------------------------------------
asym_path = T3_DIR / "asymmetry_domain.csv"
refit_path = T3_DIR / "F4_asymmetry_refit.csv"
asy = None
if refit_path.exists():                      # IC com reajuste dos modelos (o do artigo)
    asy = pd.read_csv(refit_path).query("level == 'domain'").rename(columns={"key": "domain"})
    ci_note = "95% CI: cluster bootstrap with model refit"
elif asym_path.exists():                     # versões sem reajuste: IC só das pessoas da W3
    asy = pd.read_csv(asym_path)
    for side in ("low", "high"):
        ci = asy[f"{side}_ci"].astype(str).str.split("–", n=1, expand=True)
        asy[f"{side}_ci_low"], asy[f"{side}_ci_high"] = pd.to_numeric(ci[0], errors="coerce"), pd.to_numeric(ci[1], errors="coerce")
    ci_note = "95% CI: W3 resampling, fixed model"
if asy is not None and not ({"conditional", "international"} & set(asy["pair"])):
    asy = None                                   # cenário sem as caudas (ex.: só o modelo ordinal B)
if asy is not None:
    from scipy.stats import spearmanr
    pairs = [p for p in ("conditional", "international") if p in set(asy["pair"])]
    fig, axes = plt.subplots(1, len(pairs), figsize=(3.7 * len(pairs), 3.6), sharey=False, squeeze=False)
    for ax, pair in zip(axes[0], pairs):
        a = asy[asy["pair"] == pair].sort_values("diff_high_minus_low_pp")
        y = np.arange(len(a))
        for side, colr, off, lab in (("low", P["low"], -0.12, "Bottom-tail model"), ("high", P["high"], 0.12, "Top-tail model")):
            est = a[f"share_{side}_pct"]
            # barras nunca negativas (o ponto pode cair fora do IC percentil do bootstrap)
            err = np.vstack([(est - a[f"{side}_ci_low"]).clip(lower=0).fillna(0), (a[f"{side}_ci_high"] - est).clip(lower=0).fillna(0)])
            outside = ((est < a[f"{side}_ci_low"]) | (est > a[f"{side}_ci_high"])).sum()
            if outside:
                log.warning(f"F4 [{pair}, {side}]: {outside} domínio(s) com estimativa fora do IC bootstrap")
            ax.errorbar(est, y + off, xerr=err, fmt="o", ms=5, color=colr, ecolor=colr, elinewidth=1, capsize=0, label=lab)
        ax.set_yticks(y, a["domain"])
        ax.set_xlabel("Share of |SHAP|, %")
        title = {"conditional": "For body size (≤P10 vs ≥P90 conditional)",
                 "international": "Against world norms (<P10 vs ≥P75)"}[pair]
        ax.set_title(title, loc="left", fontsize=8.5, color=P["ink"])
        ax.grid(axis="y", visible=False)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, fontsize=7.5, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(f"Top and bottom tails are explained by different domains (W3; {ci_note})", x=0.01, ha="left",
                 fontsize=9.5, color=P["ink"])
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    run.save_figure(fig, "tail_asymmetry", "participação dos domínios nos modelos da base e do topo", item="F4")
    plt.close(fig)
    rho = {p: spearmanr(g["share_low_pct"], g["share_high_pct"]).statistic for p, g in asy.groupby("pair")}
    t4 = asy.assign(spearman_low_high=asy["pair"].map(rho), ci=ci_note)[
        ["pair", "domain", "share_low_pct", "low_ci_low", "low_ci_high", "share_high_pct", "high_ci_low", "high_ci_high",
         "diff_high_minus_low_pp", "diff_ci_low", "diff_ci_high", "spearman_low_high", "ci"]].round(2)
    run.save_table(t4, "tail_asymmetry", "participação por domínio na base e no topo, com diferença e IC", item="T4")

# -----------------------------------------------------------------------------
# F5 — regressão quantílica (termos com maior mudança entre τ = 0,10 e 0,90)
# -----------------------------------------------------------------------------
qr_path, qa_path = T3_DIR / "F5_qr_coefficients.csv", T3_DIR / "qr_asymmetry.csv"
if qr_path.exists():
    qr, qa = pd.read_csv(qr_path), pd.read_csv(qa_path)
    # termos interpretáveis (sem indicadores de ausência e sem categorias "Missing"/"Other"),
    # ordenados pela evidência de mudança entre τ = 0,10 e 0,90 (diferença / largura do IC)
    qa = qa[~qa["term"].str.endswith("__missing") & ~qa["term"].str.contains("=Missing|=Other", regex=True)].copy()
    qa["evidence"] = qa["diff_p90_minus_p10"].abs() / (qa["diff_ci_high"] - qa["diff_ci_low"]).replace(0, np.nan)
    top = qa.sort_values("evidence", ascending=False).head(8)["term"].tolist()
    fig, axes = plt.subplots(2, 4, figsize=(7.2, 3.8), sharex=True)
    for ax, term in zip(axes.flat, top):
        q = qr[(qr["term"] == term) & (qr["tau"] != "OLS (mean)")].copy()
        q["tau"] = q["tau"].astype(float)
        ols = qr[(qr["term"] == term) & (qr["tau"] == "OLS (mean)")]["coef"].iloc[0]
        ax.fill_between(q["tau"], q["ci_low"], q["ci_high"], color=P["elsi"], alpha=0.18, lw=0)
        ax.plot(q["tau"], q["coef"], color=P["elsi"], lw=2, marker="o", ms=3.5)
        ax.axhline(ols, color=P["norm_line"], ls="--", lw=1)
        ax.axhline(0, color=P["ink2"], lw=0.6)
        feat = q["feature"].iloc[0]
        lab = core.FEATURE_BY_NAME[feat].label if feat in core.FEATURE_BY_NAME else term
        if "=" in term:
            lab = f"{lab.split(' (')[0]}: {term.split('=', 1)[1]}"
        ax.set_title(lab[:38], fontsize=6.8, loc="left", color=P["ink"])
    for ax in axes[1]:
        ax.set_xlabel("Quantile (τ)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Effect on z (conditional)")
    fig.suptitle("Effects across the conditional strength distribution (dashed: mean effect, OLS)",
                 x=0.01, ha="left", fontsize=10, color=P["ink"])
    fig.tight_layout()
    run.save_figure(fig, "quantile_effects", "regressão quantílica: 8 termos com maior mudança entre τ 0,10 e 0,90", item="F5")
    plt.close(fig)

# -----------------------------------------------------------------------------
# S3 — calibração da referência condicional
# -----------------------------------------------------------------------------
cal = pd.read_csv(up1 / "tables" / "S3_reference_calibration.csv")
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True)
for ax, sp in zip(axes, ("train", "test")):
    c = cal[cal["split"] == sp].reset_index(drop=True)
    x = np.arange(len(c))
    ax.axhline(10, color=P["norm_line"], ls="--", lw=1)
    ax.scatter(x - 0.15, c["pct_le_P10"], s=14, color=P["low"], label="% ≤P10")
    ax.scatter(x + 0.15, c["pct_ge_P90"], s=14, color=P["high"], label="% ≥P90")
    ax.set_xticks(x, [f"{s[0].upper()} {a} {h}" for s, a, h in zip(c["sex"], c["age_band"], c["height_tertile"])],
                  rotation=90, fontsize=5.5)
    ax.set_title({"train": "Training waves (W1+W2)", "test": "Temporal validation (W3)"}[sp], loc="left", fontsize=8.5)
axes[0].set_ylabel("% of people (ideal 10%)")
axes[0].legend(fontsize=7)
fig.tight_layout()
run.save_figure(fig, "reference_calibration", "calibração da referência condicional por sexo, idade e tercil de altura", item="S3")
plt.close(fig)

# -----------------------------------------------------------------------------
# T3 e docx
# -----------------------------------------------------------------------------
perf = pd.read_csv(up2 / "tables" / "T3_performance.csv")
t3 = perf[perf["Primary"] == "yes"].dropna(axis=1, how="all")
tabs = {"Table 1. Participant characteristics by wave (survey-weighted)": t1,
        "Table 2. National position against international norms (survey-weighted)": t2,
        "Table 3. Predictive performance of the primary models": t3}
if asy is not None:
    tabs["Table 4. Domain shares in models of the bottom and top tails (W3)"] = t4
to_docx(tabs, "tables_main")
run.finish(notes="T1–T4, F2–F5, S3")

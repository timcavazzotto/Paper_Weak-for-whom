"""
07_figures.py — figuras combinadas (painéis a, b, c…) e tabelas finais do artigo
===============================================================================
NÃO recalcula nada: lê o que 01–04 (e 06, se existir) já gravaram, usando a execução
concluída mais recente do cenário principal (completa antes da rápida).

Estrutura para a JSHS (≤ 6 itens no texto principal; ver BUDGET): Table 1 · Fig 1 desenho e posição nacional ·
Fig 2 força para o corpo (FigCond) · Fig 3 hipóteses base × topo · Fig 4 onde está a assimetria · Fig 5 pontos de corte.
Suplemento: FigS1–S7 e Tables S1–S5. A numeração final é automática (tables/figure_map.csv).

IDs internos das figuras (nomes das funções):
  Fig1  Design and national position: desenho, centil condicional, ELSI × normas, quartis, desigualdades
  Fig2  Model performance (W3): ROC e calibração das caudas, observado × previsto, algoritmos, camadas
  Fig3  What explains strength for body size: SHAP, permutação (3 algoritmos), Shapley R², dependência
  FigCond  fusão de Fig2 + Fig3 para o texto principal: observado × previsto, ROC base × topo, Shapley R², SHAP
  Fig4  Top and bottom are not mirror images: quadro de hipóteses (participação, diferença, sensibilidades)
  Fig5  Where the asymmetry lives: |SHAP| base × topo, regressão quantílica, região
  Fig6  Which low-strength criterion tracks health (08_cutoffs): quem cada critério classifica, OR e ganho de AUC
        na mesma prevalência, discordantes EWGSOP2 × condicional
Suplementares: FigS1 fluxograma | FigS2 ausentes | FigS3 calibração da referência |
               FigS4 permutação por variável | FigS5 UPA (ICC e efeitos fixos)
Tabelas em formato de artigo (três linhas horizontais, título em cima, notas embaixo, grupos recuados,
uma tabela por página, paisagem quando larga): Table1 características | Table2 posição nacional |
Table3 desempenho | Table4 assimetria | Table5 critérios × desfechos | Table6 discordantes -> docx/tables_main.docx e docx/tables_supplement.docx (numeradas
conforme BUDGET: Table 1, 2… no texto principal; Table S1, S2… no suplemento) + .csv de cada uma.

Uso (na raiz, depois do run_scenarios):  .venv\\Scripts\\python code\\07_figures.py
Para mudar o que vai para o texto principal, a largura ou a fonte: edite o bloco CONFIGURAÇÃO abaixo.
"""
import json
import re
import warnings
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

import sr_core as core

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402,F401
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURAÇÃO (o que muda de revista para revista)
# =============================================================================
WIDTH_MM = 180                         # largura de página dupla
FONT = ["Arial", "Helvetica", "DejaVu Sans"]
DPI_PNG = 600
BUDGET = {                             # JSHS: no máximo 6 itens (figuras + tabelas) no texto principal; o resto vai para o suplemento
    "main": ["Table1", "Fig1", "FigCond", "Fig4", "Fig5", "Fig6"],
    "supplement": ["FigS1", "FigS2", "FigS3", "Fig2", "Fig3", "FigS4", "FigS5", "Table2", "Table7", "Table3", "Table8", "Table4", "Table5", "Table6"],
}
# Os nomes acima são IDs internos (das funções). A numeração final segue a ordem do BUDGET: figuras do texto principal
# viram Fig1, Fig2…; as do suplemento, FigS1, FigS2… (o mapa fica em tables/figure_map.csv).
TABLE_FONT, TABLE_PT = "Times New Roman", 9   # tabelas do .docx (muitas revistas pedem Times 9–10 pt)
SHOW_BADGES = True                     # selos Supported / Not supported no quadro de hipóteses (Fig4)
# Okabe-Ito
C_LOW, C_HIGH = "#E69F00", "#0072B2"   # base (laranja) e topo (azul) em todas as figuras
C_WOMEN, C_MEN = "#CC79A7", "#009E73"
C_ROLE = {"deficit": "#D55E00", "capacity": "#009E73", "context": "#8C8C8C"}
INK, INK2, GREY, LGREY, FRAME = "#111111", "#4D4D4D", "#8C8C8C", "#D9D9D9", "#A6A6A6"
ALG_MK = {"XGBoost": "D", "LightGBM": "o", "ElasticNet": "s", "Logistic": "s", "OrdinalLogit": "s"}
# hipóteses do quadro (Fig4): linhas = ("domain"|"role", nome); esperado = sinal de (topo − base)
HYPOTHESES = [
    {"id": "H1", "title": "Deficits concentrate at the bottom", "expect": -1, "color": "#D55E00",
     "rows": [("domain", "Functional status"), ("domain", "Morbidity"), ("role", "deficit")]},
    {"id": "H2", "title": "Capacity drives the top", "expect": +1, "color": "#009E73",
     "rows": [("domain", "Physical performance"), ("domain", "Behavioural"), ("role", "capacity")]},
    {"id": "H3", "title": "Place and life course shape the top", "expect": +1, "color": "#56B4E9",
     "rows": [("domain", "Context"), ("domain", "Life course"), ("domain", "Socioeconomic & work")]},
    {"id": "", "title": "Body size (descriptive)", "expect": 0, "color": "#BFBFBF",
     "rows": [("domain", "Anthropometry")]},
    {"id": "", "title": "Not hypothesised", "expect": 0, "color": "#BFBFBF",
     "rows": [("domain", "Social & wellbeing")]},
]
QR_NAME = {"ffm_est": "Fat-free mass", "adl_n": "ADL difficulties", "mobility_n": "Mobility difficulty", "gait_speed": "Gait speed",
           "mvpa_min_week": "Physical activity", "heavy_work": "Heavy work", "region=North": "Region: North", "region=Northeast": "Region: Northeast"}
ROLE_NAME = {"deficit": "All deficit variables", "capacity": "All capacity variables", "context": "All context variables"}

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": FONT, "font.size": 6.5, "axes.linewidth": 0.6,
                     "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2.2, "ytick.major.size": 2.2,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
                     "legend.frameon": False, "pdf.fonttype": 42, "svg.fonttype": "none"})

run = core.Run("07_figures", __file__)
log = run.log


# =============================================================================
# localizar resultados (execução completa mais recente; se não houver, a rápida)
# =============================================================================
def versions():
    """Versão atual e anteriores da MESMA MAJOR (não mistura desfechos de MAJORs diferentes)."""
    major = core.VERSION.split(".")[0]
    return [core.VERSION] + sorted([p.name[1:] for p in core.OUTPUTS.glob("v*")
                                    if p.is_dir() and p.name[1:] != core.VERSION and p.name[1:].split(".")[0] == major],
                                   key=core._version_key, reverse=True)


def latest(stage, scenario="main"):
    for fast in (False, True):
        for v in versions():
            d = core.stage_dir(stage, v, scenario, fast=fast)
            m = d / "manifest.json"
            if m.exists() and json.loads(m.read_text(encoding="utf-8")).get("status") == "complete":
                return d
    return None


def upstream_dir(man, stage, scenario="main"):
    up = man["upstream"][stage]
    p = Path(up["path"])
    if p.exists():
        return p
    for fast in (False, True):                       # caminho de outro computador: reconstrói pela versão
        d = core.stage_dir(stage, up["version"], scenario, fast=fast)
        if (d / "manifest.json").exists():
            return d
    raise SystemExit(f"PARADO: não encontrei {stage} v{up['version']}")


D4 = latest("04_report")
if D4 is None:
    raise SystemExit("PARADO: rode 01–04 do cenário principal antes do 07.")
M4 = json.loads((D4 / "manifest.json").read_text(encoding="utf-8"))
D1, D2, D3 = (upstream_dir(M4, s) for s in ("01_prepare", "02_model", "03_explain"))
D6 = latest("06_sensitivity")
D8 = latest("08_cutoffs")
for k, d in (("01", D1), ("02", D2), ("03", D3), ("04", D4), ("06", D6), ("08", D8)):
    log.info(f"{k}: {d if d else '— (sem; colunas de sensibilidade ficam vazias)'}")
    if d:
        run.add_input(d / "manifest.json", f"manifest_{k}")
FAST_USED = any(d and d.name.endswith("_fast") for d in (D2, D3, D6, D8))
if FAST_USED:
    log.warning("ATENÇÃO: parte dos resultados vem do MODO RÁPIDO — figuras só para conferir o layout.")

ALL = pd.read_parquet(D1 / "data" / "analytic_all.parquet")
INC = ALL[ALL["exclusion"] == "included"].copy()
PRIM = json.loads((D2 / "tables" / "primary_models.json").read_text(encoding="utf-8"))
PERF = pd.read_csv(D2 / "tables" / "performance_long.csv")
FEAT = core.FEATURE_BY_NAME


def t3(*names):
    for n in names:
        p = D3 / "tables" / n
        if p.exists():
            return pd.read_csv(p)
    return None


def short(name):
    """Rótulo curto em inglês de uma variável ou termo (one-hot: 'region=North' -> 'Region: North')."""
    base, _, lv = name.partition("=")
    base = base.replace("__missing", "")
    lab = FEAT[base].label.split(" (")[0] if base in FEAT else base
    lab = lab.replace("Number of chronic conditions", "Chronic conditions")
    return f"{lab}: {lv}" if lv else lab


NUM = re.compile(r"(-?[\d.]+)\s*\((-?[\d.]+)\s*(?:–|to)\s*(-?[\d.]+)\)")


def parse_ci(s):
    m = NUM.search(str(s))
    return (float(m.group(1)), float(m.group(2)), float(m.group(3))) if m else (np.nan, np.nan, np.nan)


# =============================================================================
# canvas em milímetros (origem no canto superior esquerdo), painéis tracejados
# =============================================================================
def _labels():
    out = {}
    for part, pre in (("main", "Fig"), ("supplement", "FigS")):
        figs = [k for k in BUDGET[part] if k.startswith("Fig")]
        out.update({k: f"{pre}{i + 1}" for i, k in enumerate(figs)})
    return out


FIG_LABEL = _labels()


class Canvas:
    def __init__(self, height_mm, width_mm=WIDTH_MM):
        self.W, self.H = width_mm, height_mm
        self.fig = plt.figure(figsize=(width_mm / 25.4, height_mm / 25.4))

    def ax(self, x, y, w, h, **kw):
        return self.fig.add_axes([x / self.W, 1 - (y + h) / self.H, w / self.W, h / self.H], **kw)

    def text(self, x, y, s, **kw):
        kw.setdefault("ha", "left"); kw.setdefault("va", "baseline")
        return self.fig.text(x / self.W, 1 - y / self.H, s, **kw)

    def panel(self, x, y, w, h, letter, title=None, sub=None):
        self.fig.patches.append(FancyBboxPatch((x / self.W, 1 - (y + h) / self.H), w / self.W, h / self.H,
                                               boxstyle="round,pad=0,rounding_size=0.006", transform=self.fig.transFigure,
                                               fill=False, ec=FRAME, lw=0.6, ls=(0, (3, 2)), zorder=0))
        self.text(x + 1.2, y + 0.2, letter, fontsize=9, fontweight="bold", va="center",
                  bbox=dict(fc="white", ec="none", pad=0.6), zorder=5)
        if title:
            self.text(x + 2.5, y + 5.2, title, fontsize=7.2, fontweight="bold")
        if sub:
            self.text(x + 2.5, y + 8.6, sub, fontsize=5.8, color=INK2)

    def save(self, name, desc):
        iid, _, rest = name.partition("_")
        name = f"{FIG_LABEL.get(iid, iid)}_{rest}"
        base = run.dir / "figures" / name
        self.fig.savefig(base.with_suffix(".pdf"))
        self.fig.savefig(base.with_suffix(".png"), dpi=DPI_PNG)
        run.save_file(base.with_suffix(".pdf"), "figure", desc, name.split("_")[0])
        run.save_file(base.with_suffix(".png"), "figure_png", desc, name.split("_")[0])
        plt.close(self.fig)


def clean(ax, grid_x=False):
    ax.tick_params(labelsize=6, pad=1.5)
    if grid_x:
        ax.grid(axis="x", color="#EEEEEE", lw=0.5)
        ax.set_axisbelow(True)


def ci_err(est, lo, hi):
    est, lo, hi = (np.asarray(v, float) for v in (est, lo, hi))
    return np.vstack([np.clip(est - lo, 0, None), np.clip(hi - est, 0, None)])


def wprop(mask, w):
    m, w = np.asarray(mask, float), np.asarray(w, float)
    ok = ~np.isnan(m) & ~np.isnan(w)
    return float(np.sum(m[ok] * w[ok]) / np.sum(w[ok]))


# =============================================================================
# Fig1 — design and national position
# =============================================================================
def fig1():
    cv = Canvas(185)
    n_tr, n_te = int((INC["split"] == "train").sum()), int((INC["split"] == "test").sum())
    # a — desenho
    cv.panel(1, 2, 178, 36, "a", "Study design", "ELSI-Brazil, community-dwelling adults ≥50 years with measured handgrip")
    boxes = [(4, 13, 48, 21, "Training · waves 1+2 (2015–21)", [f"n = {n_tr:,}", "5×3 nested cross-validation", "Unweighted (weights: sensitivity)"], "#F2F2F2"),
             (59, 13, 50, 21, "Outcomes", ["Conditional centile: strength for age,", "sex and height (ELSI reference, train only)",
                                         "International centile (Tomkinson 2025)"], "white"),
             (116, 13, 60, 21, "Temporal validation · wave 3 (2023–24)", [f"n = {n_te:,} · models frozen", "Bottom (≤P10) and top (≥P90) tails",
                                                                                 "SHAP, permutation, quantile regression"], "#F2F2F2")]
    for x, y, w, h, ttl, lines, fc in boxes:
        cv.fig.patches.append(FancyBboxPatch((x / cv.W, 1 - (y + h) / cv.H), w / cv.W, h / cv.H, boxstyle="round,pad=0,rounding_size=0.004",
                                             transform=cv.fig.transFigure, fc=fc, ec=GREY, lw=0.6))
        cv.text(x + 2, y + 5, ttl, fontsize=6.4, fontweight="bold")
        for i, ln in enumerate(lines):
            cv.text(x + 2, y + 10 + 4.2 * i, ln, fontsize=6)
    for x0, x1 in ((52, 59), (109, 116)):
        cv.fig.add_artist(matplotlib.patches.FancyArrowPatch((x0 / cv.W, 1 - 23.5 / cv.H), (x1 / cv.W, 1 - 23.5 / cv.H),
                                                             transform=cv.fig.transFigure, arrowstyle="-|>", mutation_scale=7, lw=0.7, color=INK2))
    # b — o centil condicional (mulheres no tercil médio de altura, W1+W2)
    cv.panel(1, 41, 58, 62, "b", "Strength for body size", "Women, middle height tertile · training waves")
    ax = cv.ax(11, 55, 45, 40)
    d = INC[(INC["male"] == 0) & (INC["split"] == "train") & INC["centile_cond"].notna()]
    t = d["height_cm"].quantile([1 / 3, 2 / 3]).to_numpy()
    d = d[d["height_cm"].between(*t)].sample(frac=1, random_state=core.SEED)
    mid = d[(d["tail_cond_low"] == 0) & (d["tail_cond_high"] == 0)]
    ax.scatter(mid["age"], mid["grip_max_kg"], s=1.2, c=LGREY, lw=0, rasterized=True)
    ax.scatter(d.loc[d["tail_cond_low"] == 1, "age"], d.loc[d["tail_cond_low"] == 1, "grip_max_kg"], s=2, c=C_LOW, lw=0, rasterized=True)
    ax.scatter(d.loc[d["tail_cond_high"] == 1, "age"], d.loc[d["tail_cond_high"] == 1, "grip_max_kg"], s=2, c=C_HIGH, lw=0, rasterized=True)
    s = d.sort_values("age")
    ax.plot(s["age"], s["grip_expected_kg"].rolling(200, center=True, min_periods=50).mean(), color=INK, lw=1)
    ax.set_xlabel("Age, years", fontsize=6.4, labelpad=1); ax.set_ylabel("Handgrip, kg", fontsize=6.4, labelpad=1)
    ax.text(0.98, 0.97, "Top ≥P90", color=C_HIGH, transform=ax.transAxes, ha="right", va="top", fontsize=6, fontweight="bold")
    ax.text(0.98, 0.88, "Bottom ≤P10", color=C_LOW, transform=ax.transAxes, ha="right", va="top", fontsize=6, fontweight="bold")
    ax.text(0.02, 0.03, "line: expected for age, sex, height", transform=ax.transAxes, fontsize=5.4, color=INK2)
    clean(ax)
    # c, d — ELSI × normas
    ages = np.arange(50, 91)
    iP = {c: i for i, c in enumerate(core.PCT_COLS)}
    for k, (letter, sex, is_m, ttl, x0) in enumerate((("c", "female", 0, "Women", 61), ("d", "male", 1, "Men", 120))):
        cv.panel(x0, 41, 58 if k == 0 else 59, 62, letter, f"{ttl} vs international norms", f"ELSI P10–P90 (weighted, ×{core.PROTOCOL_FACTORS[core.CFG.protocol_adjustment]['absolute']:.3f} protocol) vs Tomkinson 2025")
        ax = cv.ax(x0 + 10, 55, 45, 40)
        cur = core.norm_curves(ages.astype(float), sex, core.CFG.norm)
        ax.fill_between(ages, cur[:, iP["P10"]], cur[:, iP["P90"]], color="#EDEDED", lw=0)
        ax.plot(ages, cur[:, iP["P50"]], color=GREY, lw=1, ls="--")
        GKG = "grip_adj_kg" if "grip_adj_kg" in INC.columns else "grip_max_kg"   # força no protocolo das normas
        g = INC[(INC["male"] == is_m) & INC[GKG].notna()]
        rows = []
        for lo in range(50, 90, 5):
            gg = g[(g["age"] >= lo) & (g["age"] < lo + 5)]
            if len(gg) >= 40:
                rows.append([lo + 2.5, *core.weighted_quantile(gg[GKG], gg["weight_norm"], [0.1, 0.5, 0.9])])
        r = np.array(rows)
        col = C_WOMEN if is_m == 0 else C_MEN
        ax.fill_between(r[:, 0], r[:, 1], r[:, 3], color=col, alpha=0.18, lw=0)
        ax.plot(r[:, 0], r[:, 2], color=col, lw=1.4, marker="o", ms=2.2)
        ax.set_xlim(50, 90); ax.set_xlabel("Age, years", fontsize=6.4, labelpad=1)
        if k == 0:
            ax.set_ylabel("Handgrip, kg", fontsize=6.4, labelpad=1)
        ax.text(0.98, 0.97, "ELSI median", color=col, transform=ax.transAxes, ha="right", va="top", fontsize=6, fontweight="bold")
        ax.text(0.98, 0.88, "World median (dashed)", color=GREY, transform=ax.transAxes, ha="right", va="top", fontsize=6)
        clean(ax)
    # e — quartis internacionais por onda
    cv.panel(1, 106, 72, 77, "e", "Position against world norms", "International quartile, % (survey-weighted)")
    ax = cv.ax(21, 122, 48, 40)
    qcols = ["#7F7F7F", "#BFBFBF", "#9ECAE1", C_HIGH]
    for i, w in enumerate((1, 2, 3)):
        g = INC[INC["wave"] == w]
        left = 0
        for q in range(4):
            p = 100 * wprop(g["quartile_intl"] == q, g["weight"])
            ax.barh(i, p, left=left, color=qcols[q], edgecolor="white", lw=0.8, height=0.62)
            if p > 6:
                ax.text(left + p / 2, i, f"{p:.0f}", ha="center", va="center", fontsize=5.8, color="white" if q in (0, 3) else INK)
            left += p
    ax.set_yticks([0, 1, 2], [f"W{w} {core.WAVE_YEARS[w]}" for w in (1, 2, 3)]); ax.invert_yaxis()
    ax.set_xlim(0, 100); ax.set_xlabel("% of older adults", fontsize=6.4, labelpad=1)
    clean(ax)
    hl = [Rectangle((0, 0), 1, 1, color=c) for c in qcols]
    ax.legend(hl, ["<P25", "P25–49", "P50–74", "≥P75"], ncol=4, fontsize=5.8, loc="upper center", bbox_to_anchor=(0.45, -0.32),
              handlelength=1, columnspacing=0.8)
    # f — desigualdades (% < P25, W3)
    t2all = pd.read_csv(D4 / "tables" / "T2_national_position.csv", keep_default_na=False)
    allv = parse_ci(t2all.query("wave == 'W3' and group == 'All'")["% <P25 (95% CI)"].iloc[0])
    cv.panel(75, 106, 104, 77, "f", "Inequalities in low strength",
             f"% below the international P25 · wave 3 · weighted (95% CI); dashed: national {allv[0]:.0f}%")
    t2 = pd.read_csv(D4 / "tables" / "T2_national_position.csv", keep_default_na=False)
    t2 = t2[(t2["wave"] == "W3") & (t2["group"] != "All") & (t2["level"].astype(str) != "")].copy()
    t2["level"] = t2["level"].replace({"None": "No schooling"})
    t2[["est", "lo", "hi"]] = t2["% <P25 (95% CI)"].apply(lambda s: pd.Series(parse_ci(s)))
    lab = {"Income quintile": "Income Q", "Race/skin colour": "Race"}
    t2["level"] = t2["level"].astype(str).str.replace(r"\.0$", "", regex=True)
    t2["lab"] = [f"{lab.get(g, g)}: {lv}" for g, lv in zip(t2["group"], t2["level"])]
    half = int(np.ceil(len(t2) / 2))
    for j, part in enumerate((t2.iloc[:half], t2.iloc[half:])):
        ax = cv.ax(100 + j * 41, 117, 34 if j == 0 else 35, 58)
        y = np.arange(len(part))
        ax.axvline(allv[0], color=GREY, lw=0.6, ls="--")
        ax.errorbar(part["est"], y, xerr=ci_err(part["est"], part["lo"], part["hi"]), fmt="o", ms=2.6, color=INK, ecolor=INK2, elinewidth=0.7)
        ax.set_yticks(y, part["lab"], fontsize=5.6); ax.invert_yaxis()
        ax.set_xlim(30, 85); ax.set_xlabel("% <P25", fontsize=6.2, labelpad=1)
        clean(ax, grid_x=True)
    cv.save("Fig1_design_national", "desenho, centil condicional, ELSI × normas, quartis e desigualdades")


# =============================================================================
# Fig2 — model performance
# =============================================================================
def pred(task, algo=None, layer="M3", kind="test"):
    algo = algo or PRIM[task]["algorithm"]
    p = D2 / "predictions" / f"{kind}_{task}_{algo}_{layer}.parquet"
    return pd.read_parquet(p) if p.exists() else None


def perf(task, metric, algo=None, layer="M3", setname="test"):
    algo = algo or PRIM[task]["algorithm"]
    g = PERF[(PERF["task"] == task) & (PERF["algorithm"] == algo) & (PERF["layer"] == layer) & (PERF["metric"] == metric)]
    if setname == "cv":
        v = g[g["set"] == "cv_fold"]["value"]
        return (v.mean(), v.mean() - 1.96 * v.std() / np.sqrt(len(v)), v.mean() + 1.96 * v.std() / np.sqrt(len(v))) if len(v) else (np.nan,) * 3
    g = g[g["set"] == setname]
    return (g["value"].iloc[0], g["ci_low"].iloc[0], g["ci_high"].iloc[0]) if len(g) else (np.nan,) * 3


def fig2():
    from sklearn.metrics import roc_curve
    cv = Canvas(150)
    pairs = [("a", "c", "Conditional tails", "Tcond_low", "Tcond_high", "≤P10", "≥P90", 1),
             ("b", "d", "International tails", "Tintl_low", "Tintl_high", "<P10", "≥P75", 61)]
    for la, lc, ttl, lo, hi, lo_lab, hi_lab, x0 in pairs:
        if lo not in PRIM or hi not in PRIM:
            continue
        cv.panel(x0, 2, 58, 62, la, f"ROC · {ttl}", "ELSI wave 3 (temporal validation)")
        ax = cv.ax(x0 + 10, 15, 44, 42)
        ax.plot([0, 1], [0, 1], color=GREY, lw=0.6, ls="--")
        for k, (task, col, nm) in enumerate(((lo, C_LOW, f"Bottom {lo_lab}"), (hi, C_HIGH, f"Top {hi_lab}"))):
            p = pred(task)
            f, tp, _ = roc_curve(p["observed"], p["predicted"])
            ax.plot(f, tp, color=col, lw=1.2)
            a = perf(task, "auc")
            ax.text(0.97, 0.05 + 0.09 * (1 - k), f"{nm} · AUC {a[0]:.2f} ({a[1]:.2f}–{a[2]:.2f})", color=col, transform=ax.transAxes,
                    ha="right", fontsize=5.4, fontweight="bold", bbox=dict(fc="white", ec="none", pad=0.3, alpha=0.85))
        ax.set_xlabel("1 − specificity", fontsize=6.4, labelpad=1); ax.set_ylabel("Sensitivity", fontsize=6.4, labelpad=1)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); clean(ax)
        cv.panel(x0, 67, 58, 50, lc, f"Calibration · {ttl}", "Deciles of predicted probability, wave 3")
        ax = cv.ax(x0 + 10, 80, 44, 31)
        mx = 0
        for task, col in ((lo, C_LOW), (hi, C_HIGH)):
            p = pred(task)
            b = pd.qcut(p["predicted"].rank(method="first"), 10, labels=False)
            g = p.groupby(b).agg(x=("predicted", "mean"), y=("observed", "mean"))
            ax.plot(g["x"], g["y"], color=col, lw=1, marker="o", ms=2.3)
            mx = max(mx, g["x"].max(), g["y"].max())
        mx = min(1, mx * 1.1)
        ax.plot([0, mx], [0, mx], color=GREY, lw=0.6, ls="--")
        ax.set_xlim(0, mx); ax.set_ylim(0, mx)
        ax.set_xlabel("Predicted probability", fontsize=6.4, labelpad=1); ax.set_ylabel("Observed rate", fontsize=6.4, labelpad=1)
        clean(ax)
    # e — observado × previsto (z condicional)
    cv.panel(121, 2, 58, 62, "e", "Strength for body size", "Observed vs predicted z, deciles · wave 3")
    ax = cv.ax(131, 15, 44, 42)
    p = pred("Acond")
    if p is not None:
        b = pd.qcut(p["predicted"].rank(method="first"), 10, labels=False)
        g = p.groupby(b).agg(x=("predicted", "mean"), y=("observed", "mean"), s=("observed", "std"), n=("observed", "size"))
        ax.errorbar(g["x"], g["y"], yerr=1.96 * g["s"] / np.sqrt(g["n"]), fmt="o", ms=2.5, color=INK, ecolor=INK2, elinewidth=0.7)
        lim = [min(g["x"].min(), g["y"].min()) - 0.1, max(g["x"].max(), g["y"].max()) + 0.1]
        ax.plot(lim, lim, color=GREY, lw=0.6, ls="--")
        c = perf("Acond", "c_index"); r2 = perf("Acond", "r2"); sl = perf("Acond", "cal_slope")
        ax.text(0.03, 0.97, f"C-index {c[0]:.2f} ({c[1]:.2f}–{c[2]:.2f})\nR² {r2[0]:.2f} · slope {sl[0]:.2f}", transform=ax.transAxes,
                va="top", fontsize=5.8)
        ax.set_xlabel("Predicted z (conditional)", fontsize=6.4, labelpad=1); ax.set_ylabel("Observed z", fontsize=6.4, labelpad=1)
    clean(ax)
    # f — discriminação por tarefa e algoritmo
    cv.panel(121, 67, 58, 50, "f", "Discrimination by algorithm", "W3; primary model with 95% CI")
    ax = cv.ax(146, 79, 30, 22)
    rows = [("Acond", "c_index", "Body-size z (C)"), ("Aintl", "c_index", "World z (C)"), ("Tcond_low", "auc", "Bottom ≤P10"),
            ("Tcond_high", "auc", "Top ≥P90"), ("Tintl_low", "auc", "Bottom <P10 world"), ("Tintl_high", "auc", "Top ≥P75 world")]
    rows = [r for r in rows if r[0] in PRIM]
    for i, (task, met, lab) in enumerate(rows):
        for algo in core.TASKS[task]["algorithms"]:
            e = perf(task, met, algo)
            prim = algo == PRIM[task]["algorithm"]
            col = C_LOW if "Bottom" in lab else (C_HIGH if "Top" in lab else INK)
            if prim:
                ax.errorbar(e[0], i, xerr=ci_err([e[0]], [e[1]], [e[2]]), fmt=ALG_MK[algo], ms=3, color=col, ecolor=col, elinewidth=0.8)
            else:
                ax.plot(e[0], i, ALG_MK[algo], ms=2.6, mfc="none", mec=col, mew=0.6)
    ax.set_yticks(range(len(rows)), [r[2] for r in rows], fontsize=5.6); ax.invert_yaxis()
    ax.set_xlim(0.5, 0.8); ax.set_xlabel("C-index / AUC", fontsize=6.2, labelpad=1); clean(ax, grid_x=True)
    fax = ax
    # g — camadas
    cv.panel(1, 120, 178, 28, "g", "What each layer of predictors adds", "C-index; open: cross-validation (W1+W2), filled: wave 3 with 95% CI")
    ax = cv.ax(62, 129.5, 112, 11)
    lay = [("Aintl", "M0", "World z · M0 age, sex, height, weight"), ("Aintl", "M3", "World z · M3 all 55 predictors"),
           ("Acond", "M1", "Body-size z · M1 social, life course, behaviour"), ("Acond", "M2", "Body-size z · M2 + health"),
           ("Acond", "M3", "Body-size z · M3 + body and function")]
    lay = [r for r in lay if r[0] in PRIM and pred(r[0], layer=r[1]) is not None]
    for i, (task, L, lab) in enumerate(lay):
        cvv, te = perf(task, "c_index", layer=L, setname="cv"), perf(task, "c_index", layer=L)
        ax.plot(cvv[0], i, "o", ms=3, mfc="none", mec=INK2, mew=0.7)
        ax.errorbar(te[0], i, xerr=ci_err([te[0]], [te[1]], [te[2]]), fmt="o", ms=3, color=INK, elinewidth=0.8)
    ax.set_yticks(range(len(lay)), [r[2] for r in lay], fontsize=5.6); ax.invert_yaxis()
    ax.set_xlim(0.5, 0.7); ax.set_xlabel("C-index", fontsize=6.2, labelpad=1); clean(ax, grid_x=True)
    leg = [Line2D([], [], marker=m, ls="none", color=INK, mfc="none", ms=3, label=a) for a, m in
           (("XGBoost", "D"), ("LightGBM", "o"), ("Linear", "s"))]
    fax.legend(handles=leg, loc="upper center", bbox_to_anchor=(0.2, -0.36), ncol=3, fontsize=5.4, handletextpad=0.2, columnspacing=0.6)
    cv.save("Fig2_performance", "ROC, calibração, observado × previsto, algoritmos e camadas na W3")


# =============================================================================
# Fig3 — what explains strength for body size
# =============================================================================
def beeswarm(ax, S, X, feats, cmap=plt.cm.PRGn):
    rng = np.random.default_rng(core.SEED)
    for i, f in enumerate(feats):
        v = S[f].to_numpy()
        x = X[f]
        if f in core.FEATURE_BY_NAME and core.FEATURE_BY_NAME[f].kind != "cat":
            xv = pd.to_numeric(x, errors="coerce")
            col = cmap(xv.rank(pct=True).fillna(0.5).to_numpy() * 0.8 + 0.1)
        else:
            col = np.array([matplotlib.colors.to_rgba(GREY)] * len(v))
        bins = np.digitize(v, np.linspace(v.min(), v.max() + 1e-9, 60))
        off = np.zeros(len(v))
        for b in np.unique(bins):
            idx = np.where(bins == b)[0]
            n = len(idx)
            off[idx] = (rng.permutation(n) - (n - 1) / 2) / max(n, 1) * min(0.42, n / 120)
        ax.scatter(v, i + off, s=0.8, c=col, lw=0, rasterized=True)
    ax.axvline(0, color=INK2, lw=0.5)
    ax.set_yticks(range(len(feats)), [short(f) for f in feats], fontsize=5.8); ax.invert_yaxis()


def fig3():
    cv = Canvas(160)
    S, X = pd.read_parquet(D3 / "cache" / "shap_Acond.parquet"), pd.read_parquet(D3 / "cache" / "X_shap.parquet")
    feats = S.abs().mean().sort_values(ascending=False).head(14).index.tolist()
    algo = PRIM["Acond"]["algorithm"]
    cv.panel(1, 2, 88, 92, "a", "SHAP · strength for body size", f"{algo}, wave 3 (n = {len(S):,}); z of conditional centile")
    ax = cv.ax(36, 16, 48, 62)
    beeswarm(ax, S, X, feats)
    ax.set_xlabel("SHAP value (z units)", fontsize=6.2, labelpad=1); clean(ax)
    cax = cv.ax(84, 30, 1.6, 30)
    cb = matplotlib.colorbar.ColorbarBase(cax, cmap=plt.cm.PRGn, norm=matplotlib.colors.Normalize(0, 1))
    cb.set_ticks([0.05, 0.95]); cb.set_ticklabels(["Low", "High"]); cb.ax.tick_params(labelsize=5.4, length=0); cb.outline.set_visible(False)
    cv.text(34, 91.5, "← lower than expected        higher than expected →", fontsize=5.6, color=INK2)
    # b — permutação por domínio, todos os algoritmos
    pdm = t3("S4_perm_domain.csv")
    cv.panel(91, 2, 88, 92, "b", "Permutation importance by domain", "C-index drop in wave 3 · three algorithms")
    ax = cv.ax(124, 16, 51, 62)
    g = pdm[pdm["task"] == "Acond"]
    order = g[g["primary"]].sort_values("drop", ascending=False)["domain"].tolist()
    algos = core.TASKS["Acond"]["algorithms"]
    for k, a in enumerate(algos):
        ga = g[g["algorithm"] == a].set_index("domain").reindex(order)
        y = np.arange(len(order)) + (k - 1) * 0.22
        prim = a == algo
        ax.errorbar(1000 * ga["drop"], y, xerr=1000 * ci_err(ga["drop"], ga["ci_low"], ga["ci_high"]), fmt=ALG_MK[a], ms=2.8 if prim else 2.4,
                    color=INK if prim else GREY, mfc=INK if prim else "white", ecolor=INK if prim else LGREY, elinewidth=0.7, mew=0.6, label=a)
    ax.axvline(0, color=INK2, lw=0.5)
    ax.set_yticks(range(len(order)), order, fontsize=5.8); ax.invert_yaxis()
    ax.set_xlabel("C-index drop (×10⁻³)", fontsize=6.2, labelpad=1); clean(ax, grid_x=True)
    ax.legend(fontsize=5.6, loc="lower right", handletextpad=0.2)
    conc = t3("S4_concordance_algorithms.csv")
    if conc is not None:
        c = conc[conc["task"] == "Acond"]["spearman_domains"]
        cv.text(93, 90, f"Agreement between algorithms: Spearman ρ {c.min():.2f}–{c.max():.2f} (domain rankings)", fontsize=5.8, color=INK2)
    # c — Shapley R²
    sh = t3("F3_shapley_r2.csv")
    cv.panel(1, 97, 88, 61, "c", "How much is explained", "Shapley decomposition of R² by domain (linear model)")
    if sh is not None:
        s = sh[(sh["task"] == "Acond") & sh["domain"].isin(core.DOMAINS)].sort_values("shapley_r2_train", ascending=False)
        tot = sh[(sh["task"] == "Acond") & (sh["domain"] == "TOTAL explained (R²)")].iloc[0]
        ax = cv.ax(36, 115, 48, 36)
        y = np.arange(len(s))
        ax.barh(y, 100 * s["shapley_r2_train"], color="#9ECAE1", height=0.62, label="Training (in-sample)")
        ax.plot(100 * s["shapley_r2_test"], y, "D", ms=2.6, color=INK, label="Wave 3 (out-of-sample)")
        ax.axvline(0, color=INK2, lw=0.5)
        ax.set_yticks(y, s["domain"], fontsize=5.8); ax.invert_yaxis()
        ax.set_xlabel("% of variance", fontsize=6.2, labelpad=1); clean(ax, grid_x=True)
        ax.legend(fontsize=5.6, loc="lower right")
        cv.text(3.5, 111, f"Explained {100 * tot['shapley_r2_train']:.0f}% (training) and {100 * tot['shapley_r2_test']:.0f}% (wave 3) · "
                          f"unexplained ≈ {100 * (1 - tot['shapley_r2_test']):.0f}%", fontsize=6, fontweight="bold")
    # d — dependência
    cv.panel(91, 97, 88, 61, "d", "Shape of the main effects", "SHAP by value, wave 3 (dots) with binned mean (line)")
    numf = [f for f in feats if f in core.FEATURE_BY_NAME and core.FEATURE_BY_NAME[f].kind in ("cont", "ord")][:3]
    for k, f in enumerate(numf):
        ax = cv.ax(103 + k * 26.5, 112, 19, 36)
        xv = pd.to_numeric(X[f], errors="coerce")
        ax.scatter(xv, S[f], s=0.6, c=LGREY, lw=0, rasterized=True)
        ok = xv.notna()
        nb = min(12, xv[ok].nunique())
        b = pd.qcut(xv[ok].rank(method="first"), nb, labels=False)
        g = pd.DataFrame({"x": xv[ok], "s": S.loc[ok, f]}).groupby(b).mean()
        ax.plot(g["x"], g["s"], color=INK, lw=1.1)
        ax.axhline(0, color=INK2, lw=0.4)
        ax.set_xlabel(short(f).replace(" completed", ""), fontsize=5.8, labelpad=1)
        if k == 0:
            ax.set_ylabel("SHAP (z units)", fontsize=6, labelpad=1)
        lo, hi = xv.quantile([0.01, 0.99])
        ax.set_xlim(lo, hi); clean(ax)
    cv.save("Fig3_explanation", "SHAP, permutação por domínio, Shapley R² e formas dos efeitos (centil condicional)")


# =============================================================================
# FigCond — força para o corpo: desempenho e explicação (fusão das antigas Fig2 e Fig3 para o texto principal)
# =============================================================================
def figCond():
    from sklearn.metrics import roc_curve
    cv = Canvas(152)
    # a — observado × previsto
    cv.panel(1, 2, 88, 70, "a", "How well strength for body size is predicted", "Observed vs predicted conditional z, deciles · wave 3")
    ax = cv.ax(16, 17, 66, 46)
    p = pred("Acond")
    if p is not None:
        b = pd.qcut(p["predicted"].rank(method="first"), 10, labels=False)
        g = p.groupby(b).agg(x=("predicted", "mean"), y=("observed", "mean"), s=("observed", "std"), n=("observed", "size"))
        ax.errorbar(g["x"], g["y"], yerr=1.96 * g["s"] / np.sqrt(g["n"]), fmt="o", ms=2.8, color=INK, ecolor=INK2, elinewidth=0.7)
        lim = [min(g["x"].min(), g["y"].min()) - 0.1, max(g["x"].max(), g["y"].max()) + 0.1]
        ax.plot(lim, lim, color=GREY, lw=0.6, ls="--")
        c = perf("Acond", "c_index"); r2 = perf("Acond", "r2"); sl = perf("Acond", "cal_slope")
        ax.text(0.03, 0.97, f"C-index {c[0]:.2f} ({c[1]:.2f}–{c[2]:.2f})\nR² {r2[0]:.2f} ({r2[1]:.2f}–{r2[2]:.2f}) · calibration slope {sl[0]:.2f}",
                transform=ax.transAxes, va="top", fontsize=5.8)
        ax.set_xlabel("Predicted z", fontsize=6.4, labelpad=1); ax.set_ylabel("Observed z", fontsize=6.4, labelpad=1)
    clean(ax)
    # b — ROC base × topo
    cv.panel(91, 2, 88, 70, "b", "The bottom is more predictable than the top", "ROC of the conditional tails · wave 3 (temporal validation)")
    ax = cv.ax(106, 17, 66, 46)
    ax.plot([0, 1], [0, 1], color=GREY, lw=0.6, ls="--")
    for k, (task, col, nm) in enumerate((("Tcond_low", C_LOW, "Bottom ≤P10"), ("Tcond_high", C_HIGH, "Top ≥P90"))):
        if task not in PRIM:
            continue
        pp = pred(task)
        f, tp, _ = roc_curve(pp["observed"], pp["predicted"])
        ax.plot(f, tp, color=col, lw=1.3)
        a = perf(task, "auc")
        ax.text(0.97, 0.05 + 0.09 * (1 - k), f"{nm} · AUC {a[0]:.2f} ({a[1]:.2f}–{a[2]:.2f})", color=col, transform=ax.transAxes,
                ha="right", fontsize=5.8, fontweight="bold", bbox=dict(fc="white", ec="none", pad=0.3, alpha=0.85))
    ax.set_xlabel("1 − specificity", fontsize=6.4, labelpad=1); ax.set_ylabel("Sensitivity", fontsize=6.4, labelpad=1)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); clean(ax)
    # c — Shapley R²
    sh = t3("F3_shapley_r2.csv")
    cv.panel(1, 75, 88, 75, "c", "How much is explained", "Shapley decomposition of R² by domain (linear model)")
    if sh is not None:
        s_ = sh[(sh["task"] == "Acond") & sh["domain"].isin(core.DOMAINS)].sort_values("shapley_r2_train", ascending=False)
        tot = sh[(sh["task"] == "Acond") & (sh["domain"] == "TOTAL explained (R²)")].iloc[0]
        ax = cv.ax(36, 94, 48, 46)
        y = np.arange(len(s_))
        ax.barh(y, 100 * s_["shapley_r2_train"], color="#9ECAE1", height=0.62, label="Training (in-sample)")
        ax.plot(100 * s_["shapley_r2_test"], y, "D", ms=2.6, color=INK, label="Wave 3 (out-of-sample)")
        ax.axvline(0, color=INK2, lw=0.5)
        ax.set_yticks(y, s_["domain"], fontsize=5.8); ax.invert_yaxis()
        ax.set_xlabel("% of variance", fontsize=6.2, labelpad=1); clean(ax, grid_x=True)
        ax.legend(fontsize=5.6, loc="lower right")
        cv.text(3.5, 89.5, f"Explained {100 * tot['shapley_r2_train']:.0f}% (training) and {100 * tot['shapley_r2_test']:.0f}% (wave 3) · "
                           f"unexplained ≈ {100 * (1 - tot['shapley_r2_test']):.0f}%", fontsize=6, fontweight="bold")
    # d — SHAP
    S, X = pd.read_parquet(D3 / "cache" / "shap_Acond.parquet"), pd.read_parquet(D3 / "cache" / "X_shap.parquet")
    feats = S.abs().mean().sort_values(ascending=False).head(12).index.tolist()
    cv.panel(91, 75, 88, 75, "d", "What weighs, and in which direction", f"SHAP, {PRIM['Acond']['algorithm']} · wave 3 (n = {len(S):,})")
    ax = cv.ax(129, 89, 40, 51)
    beeswarm(ax, S, X, feats)
    ax.set_xlabel("SHAP value (z units)", fontsize=6.2, labelpad=1); clean(ax)
    cax = cv.ax(171, 100, 1.6, 26)
    cb = matplotlib.colorbar.ColorbarBase(cax, cmap=plt.cm.PRGn, norm=matplotlib.colors.Normalize(0, 1))
    cb.set_ticks([0.05, 0.95]); cb.set_ticklabels(["Low", "High"]); cb.ax.tick_params(labelsize=5.2, length=0); cb.outline.set_visible(False)
    cv.text(123, 147.5, "← lower than expected    higher than expected →", fontsize=5.4, color=INK2)
    cv.save("FigCond_body_size", "força para o corpo: observado × previsto, ROC base × topo, Shapley R² e SHAP (texto principal)")


# =============================================================================
# Fig4 — quadro de hipóteses topo × base
# =============================================================================
def asymmetry_table():
    """Tabela única: pair, level, key, shares (CI), diff (CI). Usa o IC com reajuste se existir."""
    rf = t3("F4_asymmetry_refit.csv")
    if rf is not None:
        a = rf.rename(columns={"share_low_pct": "low", "share_high_pct": "high", "low_ci_low": "low_lo", "low_ci_high": "low_hi",
                               "high_ci_low": "high_lo", "high_ci_high": "high_hi", "diff_high_minus_low_pp": "diff",
                               "diff_ci_low": "d_lo", "diff_ci_high": "d_hi"})
        a["ci_kind"] = "cluster bootstrap with model refit"
        return a
    parts = []
    dom = t3("asymmetry_domain.csv", "F4_asymmetry_domain.csv")
    for level, df, keycol in (("domain", dom, "domain"), ("role", t3("asymmetry_role.csv"), "role")):
        if df is None:
            continue
        d = df.rename(columns={keycol: "key", "share_low_pct": "low", "share_high_pct": "high", "diff_high_minus_low_pp": "diff",
                               "diff_ci_low": "d_lo", "diff_ci_high": "d_hi"})
        for side in ("low", "high"):
            ci = d[f"{side}_ci"].astype(str).str.split("–", n=1, expand=True)
            d[f"{side}_lo"], d[f"{side}_hi"] = pd.to_numeric(ci[0], errors="coerce"), pd.to_numeric(ci[1], errors="coerce")
        d["level"] = level
        parts.append(d)
    a = pd.concat(parts, ignore_index=True)
    a["ci_kind"] = "W3 resampling, fixed model (no refit)"
    return a


def sensitivity_long():
    if D6 is None:
        return None
    p = D6 / "tables" / "S6_sensitivity_long.csv"
    return pd.read_csv(p) if p.exists() else None


def verdict(h, A):
    if h["expect"] == 0:
        return ""
    ok = []
    for level, key in h["rows"]:
        r = A[(A["pair"] == "conditional") & (A["level"] == level) & (A["key"] == key)]
        if not len(r):
            continue
        r = r.iloc[0]
        ok.append(r["d_hi"] < 0 if h["expect"] < 0 else r["d_lo"] > 0)
    if not ok:
        return ""
    return "Supported" if all(ok) else ("Partly supported" if any(ok) else "Not supported")


def fig4():
    A = asymmetry_table()
    S6 = sensitivity_long()
    SC_ALL, SC_MK = [], {}
    if S6 is not None:                                  # mesmo marcador para o mesmo cenário em todas as linhas e na legenda
        cond = S6[S6["block"].str.startswith("Asymmetry conditional") | S6["block"].str.startswith("Asymmetry by role conditional")]
        SC_ALL = [s for s in core.SCENARIOS if s in set(cond["scenario"])]
        SC_MK = {s: ("D" if s == "main" else "sDv^<>phX*"[(k - 1) % 10]) for k, s in enumerate(SC_ALL)}
    rows = []
    for h in HYPOTHESES:
        for level, key in h["rows"]:
            rows.append((h, level, key))
    n = len(rows)
    row_h, top = 7.2, 15
    y_of, y = [], top
    for i, (h, level, key) in enumerate(rows):
        if i > 0 and rows[i - 1][0] is not h:
            y += 3
        y_of.append(y)
        y += row_h
    y_end = y
    H = y_end + 29
    cv = Canvas(H)
    colx = {"lab": 1, "a": 64, "b": 102, "c": 141}
    colw = {"a": 37, "b": 38, "c": 38}
    heads = {"a": ("Share of |SHAP|", "conditional tails · W3"),
             "b": ("Top − bottom, pp", "black: conditional · grey: world"),
             "c": ("Sensitivity analyses", "top − bottom, conditional")}
    for k in ("a", "b", "c"):
        cv.panel(colx[k], 2, colw[k], y_end + 6, k, *heads[k])
    # faixas dos blocos e textos
    for h in HYPOTHESES:
        ys = [y_of[i] for i, r in enumerate(rows) if r[0] is h]
        y0, y1 = min(ys) - 1.2, max(ys) + row_h - 1.2
        cv.fig.patches.append(Rectangle((1 / cv.W, 1 - y1 / cv.H), 178 / cv.W, (y1 - y0) / cv.H, transform=cv.fig.transFigure,
                                        color=h["color"], alpha=0.10, lw=0, zorder=-1))
        cv.fig.patches.append(Rectangle((1 / cv.W, 1 - y1 / cv.H), 1.6 / cv.W, (y1 - y0) / cv.H, transform=cv.fig.transFigure,
                                        color=h["color"], lw=0))
        import textwrap
        ttl = textwrap.wrap(h["title"], 20)
        if h["id"]:
            cv.text(4.5, y0 + 4.2, h["id"], fontsize=7.5, fontweight="bold", color=h["color"])
        for j, ln in enumerate(ttl):
            cv.text(4.5 + (6 if h["id"] else 0), y0 + 4.2 + 3.4 * j, ln, fontsize=6.2, fontweight="bold")
        v = verdict(h, A)
        if SHOW_BADGES and v:
            dark = v == "Supported"
            cv.text(10.5, y0 + 6.5 + 3.4 * len(ttl), f" {v} ", fontsize=5.6, color="white" if dark else INK, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.25", fc=INK if dark else "white", ec=INK, lw=0.6))
    for i, (h, level, key) in enumerate(rows):
        cv.text(62, y_of[i] + 3.6, key if level == "domain" else ROLE_NAME[key], fontsize=6, ha="right", va="center",
                fontweight="bold" if level == "role" else "normal")
    # eixos das três colunas: um eixo por coluna cobrindo todas as linhas (coordenadas em mm)
    def col_ax(k):
        ax = cv.ax(colx[k] + 3, top - 1.2, colw[k] - 6, y_end - top)
        ax.set_ylim(y_end - 1.2, top - 1.2)
        ax.set_yticks([]); ax.spines["left"].set_visible(False)
        ax.patch.set_alpha(0)
        clean(ax, grid_x=True)
        return ax
    axa, axb, axc = col_ax("a"), col_ax("b"), col_ax("c")
    for i, (h, level, key) in enumerate(rows):
        yc = y_of[i] + 3.6
        for pair, mk, off in (("conditional", "o", 0), ("international", "o", 1.6)):
            r = A[(A["pair"] == pair) & (A["level"] == level) & (A["key"] == key)]
            if not len(r):
                continue
            r = r.iloc[0]
            if pair == "conditional":
                axa.errorbar(r["low"], yc - 1, xerr=ci_err([r["low"]], [r["low_lo"]], [r["low_hi"]]), fmt="o", ms=2.6, color=C_LOW, elinewidth=0.7)
                axa.errorbar(r["high"], yc + 1, xerr=ci_err([r["high"]], [r["high_lo"]], [r["high_hi"]]), fmt="o", ms=2.6, color=C_HIGH, elinewidth=0.7)
            sig = (r["d_lo"] > 0) or (r["d_hi"] < 0)
            filled = pair == "conditional"
            axb.errorbar(r["diff"], yc - 0.8 + off, xerr=ci_err([r["diff"]], [r["d_lo"]], [r["d_hi"]]), fmt="D" if sig else "d",
                         ms=2.8, color=INK if filled else GREY, mfc=(INK if filled else GREY) if sig else "white", elinewidth=0.7, mew=0.6)
        if S6 is not None:
            blk = "Asymmetry conditional" if level == "domain" else "Asymmetry by role conditional"
            s6 = S6[S6["block"].str.startswith(blk) & (S6["item"] == key)]
            for j, sc in enumerate(SC_ALL):
                r = s6[s6["scenario"] == sc]
                if not len(r):
                    continue
                r = r.iloc[0]
                yy = yc - 2.4 + 4.8 * j / max(len(SC_ALL) - 1, 1)
                main = sc == "main"
                axc.errorbar(r["estimate"], yy, xerr=ci_err([r["estimate"]], [r["ci_low"]], [r["ci_high"]]),
                             fmt=SC_MK[sc], ms=2.6 if main else 2.2, color=INK if main else GREY,
                             mfc=INK if main else "white", elinewidth=0.5, mew=0.5)
    for ax in (axb, axc):
        ax.axvline(0, color=INK2, lw=0.6)
    axa.set_xlim(0, None)
    for ax, lab in ((axa, "% of total |SHAP|"), (axb, "Difference, pp"), (axc, "Difference, pp")):
        ax.set_xlabel(lab, fontsize=6.2, labelpad=1)
    if S6 is None:
        axc.text(0.5, 0.5, "run run_scenarios.py\n(06_sensitivity)", transform=axc.transAxes, ha="center", va="center", fontsize=6, color=GREY)
    # legendas
    leg = [Line2D([], [], marker="o", ls="none", color=C_LOW, ms=3, label="Bottom ≤P10"),
           Line2D([], [], marker="o", ls="none", color=C_HIGH, ms=3, label="Top ≥P90"),
           Line2D([], [], marker="D", ls="none", color=INK, ms=3, label="CI excludes 0"),
           Line2D([], [], marker="d", ls="none", color=INK, mfc="white", ms=3, label="CI includes 0")]
    if SC_ALL:
        leg += [Line2D([], [], marker=SC_MK[s], ls="none", color=INK if s == "main" else GREY,
                       mfc=INK if s == "main" else "white", ms=3, label=s.replace("sens_", "")) for s in SC_ALL]
    cv.fig.legend(handles=leg, loc="lower center", bbox_to_anchor=(0.5, 0.005), ncol=min(len(leg), 6), fontsize=5.8, handletextpad=0.2,
                  columnspacing=0.9)
    ck = A["ci_kind"].iloc[0]
    cv.text(3, y_end + 12, f"95% CI: {ck}.", fontsize=5.6, color=INK2)
    if "sens_train_w1" in SC_ALL:
        cv.text(3, y_end + 13.5, "train_w1 (training on wave 1 only): the top-tail model credits fat-free mass (a capacity variable) instead of BMI "
                             "(context), so its marker moves to the top in the capacity row and to zero in the anthropometry row; body size still "
                             "dominates the top.", fontsize=5.4, color=INK2, wrap=True, va="top")
    cv.save("Fig4_hypotheses", "quadro de hipóteses: participação na base e no topo, diferença e sensibilidades")
    return A


# =============================================================================
# Fig5 — onde a assimetria está
# =============================================================================
def fig5():
    cv = Canvas(150)
    dr = t3("asymmetry_direction.csv")
    cv.panel(1, 2, 72, 92, "a", "Each variable in each tail", "Share of |SHAP| in the bottom vs top model · conditional · W3")
    if dr is not None:
        d = dr[dr["pair"] == "conditional"].copy()
        d["lo"] = 100 * d["mean_abs_low"] / d["mean_abs_low"].sum()
        d["hi"] = 100 * d["mean_abs_high"] / d["mean_abs_high"].sum()
        ax = cv.ax(12, 15, 56, 66)
        m = max(d["lo"].max(), d["hi"].max()) * 1.15
        ax.plot([0.05, m], [0.05, m], color=GREY, lw=0.6, ls="--")
        ax.scatter(d["lo"], d["hi"], s=8, c=[C_ROLE[r] for r in d["role"]], lw=0.3, edgecolor="white", zorder=3)
        d["dev"] = np.abs(np.log((d["hi"] + 0.2) / (d["lo"] + 0.2))) * np.sqrt(d["lo"] + d["hi"])
        for _, r in d.sort_values("dev", ascending=False).head(10).iterrows():
            ax.annotate(short(r["feature"]), (r["lo"], r["hi"]), xytext=(2, 1.5), textcoords="offset points", fontsize=5.2, color=INK2)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(0.05, m); ax.set_ylim(0.05, m)
        ax.set_xlabel("Bottom model, % of |SHAP|", fontsize=6.2, labelpad=1); ax.set_ylabel("Top model, % of |SHAP|", fontsize=6.2, labelpad=1)
        ax.text(0.04, 0.96, "matters more at the top", transform=ax.transAxes, fontsize=5.6, va="top", color=C_HIGH)
        ax.text(0.96, 0.04, "matters more at the bottom", transform=ax.transAxes, fontsize=5.6, ha="right", color=C_LOW)
        clean(ax)
        leg = [Line2D([], [], marker="o", ls="none", color=c, ms=3, label=k) for k, c in C_ROLE.items()]
        ax.legend(handles=leg, fontsize=5.6, loc="upper left", bbox_to_anchor=(0.0, 0.92), title="variable type", title_fontsize=5.6)
    # b — regressão quantílica
    qr = t3("F5_qr_coefficients.csv")
    cv.panel(75, 2, 104, 92, "b", "Effects across the conditional distribution",
             "Effect on z (per SD; binary 0/1) by quantile · training waves (95% CI); dashed: mean (OLS)")
    if qr is not None:
        terms = [t for t in ("ffm_est", "adl_n", "mobility_n", "gait_speed", "mvpa_min_week", "heavy_work", "region=North", "region=Northeast")
                 if t in set(qr["term"])][:8]
        for k, term in enumerate(terms):
            r, c = divmod(k, 4)
            ax = cv.ax(86 + c * 24, 18 + r * 38, 16.5, 25)
            q = qr[(qr["term"] == term) & (qr["tau"] != "OLS (mean)")].copy()
            q["tau"] = q["tau"].astype(float)
            ols = qr[(qr["term"] == term) & (qr["tau"] == "OLS (mean)")]["coef"]
            ax.fill_between(q["tau"], q["ci_low"], q["ci_high"], color="#BDD7E7", lw=0)
            ax.plot(q["tau"], q["coef"], color=INK, lw=1, marker="o", ms=1.8)
            if len(ols):
                ax.axhline(ols.iloc[0], color=GREY, lw=0.6, ls="--")
            ax.axhline(0, color=INK2, lw=0.4)
            ax.set_xticks([0.1, 0.5, 0.9]); ax.set_title(QR_NAME.get(term, short(term))[:20], fontsize=5.8, loc="left", pad=2)
            if r == 1:
                ax.set_xlabel("Quantile", fontsize=5.8, labelpad=1)
            clean(ax)
    # c — região
    cv.panel(1, 97, 178, 51, "c", "Where people sit in each tail", "Conditional bottom (≤P10) and top (≥P90) by region, % · wave 3 · survey-weighted (95% CI)")
    ax = cv.ax(40, 110, 132, 31)
    w3 = ALL["wave"] == 3
    regions = ["North", "Northeast", "Central-West", "Southeast", "South"]
    inc = ALL["exclusion"] == "included"
    for i, rg in enumerate(regions):
        for tail, col, off in (("tail_cond_low", C_LOW, -0.15), ("tail_cond_high", C_HIGH, 0.15)):
            r = core.svy_mean(ALL[tail], ALL["weight"], ALL["stratum"], ALL["cluster"], domain=inc & w3 & (ALL["region"] == rg), proportion=True)
            ax.errorbar(100 * r["est"], i + off, xerr=100 * ci_err([r["est"]], [r["ci_low"]], [r["ci_high"]]), fmt="o", ms=3, color=col, elinewidth=0.8)
    ax.axvline(10, color=GREY, lw=0.6, ls="--")
    ax.set_yticks(range(len(regions)), regions, fontsize=6); ax.invert_yaxis()
    ax.set_xlabel("% of older adults (10% expected in each tail)", fontsize=6.2, labelpad=1); clean(ax, grid_x=True)
    leg = [Line2D([], [], marker="o", ls="none", color=C_LOW, ms=3, label="Bottom ≤P10"), Line2D([], [], marker="o", ls="none", color=C_HIGH, ms=3, label="Top ≥P90")]
    ax.legend(handles=leg, fontsize=5.8, loc="lower right")
    cv.save("Fig5_where_asymmetry", "|SHAP| base × topo por variável, regressão quantílica e caudas por região")


# =============================================================================
# suplementares
# =============================================================================
def figS1():
    fl = pd.read_csv(D1 / "tables" / "S2_flowchart.csv")
    lab = {"age_below_min": "Age <50", "proxy_interview": "Proxy interview", "dementia_or_parkinson": "Dementia or Parkinson",
           "hand_surgery_3m": "Hand surgery (3 months)", "no_physical_measures": "No physical measures",
           "grip_refused": "Refused grip test", "grip_unable_or_risky": "Unable / risky", "outcome_not_computable": "Outcome not computable"}
    cv = Canvas(78)
    for j, (w, g) in enumerate(fl.groupby("wave", sort=True)):
        x = 2 + j * 59
        cv.panel(x, 2, 57, 74, "abc"[j], f"Wave {w[1]} · {core.WAVE_YEARS[int(w[1])]}")
        y = 12
        rows = g.to_dict("records")
        cv.text(x + 4, y + 4, f"Interviewed: {rows[0]['remaining']:,}", fontsize=6.4, fontweight="bold")
        y += 9
        for r in rows[1:]:
            if r["excluded"] == 0:
                continue
            cv.text(x + 8, y + 3, f"− {lab.get(r['step'], r['step'])}: {r['excluded']:,}", fontsize=6)
            y += 7
        cv.text(x + 4, y + 6, f"Analytic sample: {rows[-1]['remaining']:,}", fontsize=6.4, fontweight="bold", color=C_HIGH)
        cv.text(x + 4, y + 11, "training" if int(w[1]) in core.CFG.train_waves else "temporal validation", fontsize=5.8, color=INK2)
    cv.save("FigS1_flowchart", "fluxograma de atrito por onda")


def figS2():
    mw = pd.read_csv(D1 / "tables" / "S2_missing_by_wave.csv")
    mw = mw.set_index("feature")[["W1", "W2", "W3"]]
    cv = Canvas(150)
    cv.panel(1, 2, 178, 146, "a", "Missing data by predictor and wave", "% missing among included participants")
    ax = cv.ax(60, 14, 60, 128)
    im = ax.imshow(mw.to_numpy(), aspect="auto", cmap="Greys", vmin=0, vmax=max(20, mw.to_numpy().max()))
    ax.set_yticks(range(len(mw)), [short(f) for f in mw.index], fontsize=5.2); ax.set_xticks([0, 1, 2], mw.columns)
    for (i, j), v in np.ndenumerate(mw.to_numpy()):
        if v >= 1:
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=4.6, color="white" if v > 10 else INK)
    cb = cv.fig.colorbar(im, cax=cv.ax(124, 14, 2, 40)); cb.ax.tick_params(labelsize=5.4); cb.outline.set_visible(False)
    cv.save("FigS2_missing", "dados ausentes por preditor e onda")


def figS3():
    cal = pd.read_csv(D1 / "tables" / "S3_reference_calibration.csv")
    cv = Canvas(70)
    for k, sp in enumerate(("train", "test")):
        c = cal[cal["split"] == sp].reset_index(drop=True)
        cv.panel(1 + k * 89, 2, 88, 66, "ab"[k], {"train": "Training waves (W1+W2)", "test": "Wave 3"}[sp],
                 "% ≤P10 and ≥P90 of the conditional reference (ideal 10%)")
        ax = cv.ax(10 + k * 89, 16, 75, 34)
        x = np.arange(len(c))
        ax.axhline(10, color=GREY, lw=0.6, ls="--")
        ax.plot(x - 0.15, c["pct_le_P10"], "o", ms=2.4, color=C_LOW)
        ax.plot(x + 0.15, c["pct_ge_P90"], "o", ms=2.4, color=C_HIGH)
        ax.set_xticks(x, [f"{s[0].upper()} {a} {h}" for s, a, h in zip(c["sex"], c["age_band"], c["height_tertile"])], rotation=90, fontsize=4.6)
        ax.set_ylim(0, 22); ax.set_ylabel("%", fontsize=6); clean(ax)
    cv.save("FigS3_reference_calibration", "calibração da referência condicional")


def figS4():
    pv = t3("S4_perm_variable.csv")
    if pv is None:
        return
    cv = Canvas(120)
    for k, task in enumerate([t for t in ("Acond", "Aintl") if t in set(pv["task"])]):
        g = pv[pv["task"] == task].sort_values("drop", ascending=False).head(20)
        cv.panel(1 + k * 89, 2, 88, 116, "ab"[k], {"Acond": "Strength for body size", "Aintl": "Strength vs world norms"}[task],
                 f"Permutation importance by variable · {g['algorithm'].iloc[0]} · W3")
        ax = cv.ax(40 + k * 89, 15, 45, 96)
        y = np.arange(len(g))
        ax.errorbar(1000 * g["drop"], y, xerr=1000 * ci_err(g["drop"], g["ci_low"], g["ci_high"]), fmt="o", ms=2.4, color=INK, elinewidth=0.6)
        ax.axvline(0, color=INK2, lw=0.5)
        ax.set_yticks(y, [short(f) for f in g["feature"]], fontsize=5.6); ax.invert_yaxis()
        ax.set_xlabel("C-index drop (×10⁻³)", fontsize=6); clean(ax, grid_x=True)
    cv.save("FigS4_permutation_variables", "permutação por variável")


def figS5():
    icc, wu = t3("S5_icc_upa.csv"), t3("S5_within_upa.csv")
    if icc is None or wu is None:
        return
    cv = Canvas(80)
    cv.panel(1, 2, 60, 76, "a", "Between-PSU variance", "Intraclass correlation by PSU")
    ax = cv.ax(12, 16, 45, 50)
    for k, (o, col) in enumerate((("z_cond", INK), ("hgs_z", GREY))):
        g = icc[icc["outcome"] == o]
        ax.bar(np.arange(len(g)) + (k - 0.5) * 0.36, g["icc_upa"], width=0.34, color=col, label={"z_cond": "Body-size z", "hgs_z": "World z"}[o])
    ax.set_xticks(range(3), ["W1", "W2", "W3"]); ax.set_ylabel("ICC", fontsize=6); ax.legend(fontsize=5.6); clean(ax)
    cv.panel(63, 2, 116, 76, "b", "Within-PSU comparison", "ΔR² when a domain is removed · body-size z · training waves")
    ax = cv.ax(110, 16, 64, 54)
    g = wu[wu["task"] == "Acond"]
    order = g[~g["upa_fixed_effects"]].sort_values("delta_r2_drop_domain", ascending=False)["domain"].tolist()
    for k, fe in enumerate((False, True)):
        gg = g[g["upa_fixed_effects"] == fe].set_index("domain").reindex(order)
        ax.barh(np.arange(len(order)) + (k - 0.5) * 0.36, 100 * gg["delta_r2_drop_domain"], height=0.34, color=GREY if fe else INK,
                label="within PSU (fixed effects)" if fe else "all participants")
    ax.set_yticks(range(len(order)), order, fontsize=5.8); ax.invert_yaxis()
    ax.set_xlabel("ΔR², %", fontsize=6); ax.legend(fontsize=5.6, loc="lower right"); clean(ax, grid_x=True)
    cv.save("FigS5_psu", "variância entre UPAs e comparação dentro da UPA")


# =============================================================================
# tabelas finais (formato de artigo: três linhas horizontais, título em cima,
# notas embaixo, grupos com níveis recuados, uma tabela por página)
# =============================================================================
METRIC_NAME = {"c_index": "C-index", "r2": "R²", "mae_centile": "Mean absolute error, centile points", "cal_slope": "Calibration slope",
               "cal_in_large": "Calibration-in-the-large", "qwk": "Quadratic weighted kappa", "macro_auc_ovr": "Macro AUC (one-vs-rest)",
               "auc_Q4": "AUC, top quartile", "auc": "AUC", "pr_auc": "Precision–recall AUC", "brier": "Brier score"}
LEVEL_ORDER = {"Education": ["None", "1–4", "5–8", "9–11", "≥12"], "Income quintile": ["1", "2", "3", "4", "5"],
               "Age": ["50–59", "60–69", "70–79", "80+"], "Sex": ["Women", "Men"]}
WAVE_LAB = {1: "Wave 1, 2015–16", 2: "Wave 2, 2019–21", 3: "Wave 3, 2023–24"}


def fmt3(e, d=3, sep="–"):
    return "—" if np.isnan(e[0]) else f"{e[0]:.{d}f} ({e[1]:.{d}f}{sep}{e[2]:.{d}f})"


def table1():
    t1 = pd.read_csv(D4 / "tables" / "T1_characteristics.csv", keep_default_na=False)
    heads = []
    for c in t1.columns[1:]:
        m = re.match(r"W(\d)\s+(\S+)\s+\(n=([\d,]+)\)", c)
        heads.append(f"Wave {m.group(1)}, {m.group(2)}\n(n = {m.group(3)})" if m else c)
    rows, cur = [], None
    for _, r in t1.iterrows():
        ch = r["Characteristic"]
        if ch.startswith("n (unweighted)"):
            continue
        vals = [r[c] for c in t1.columns[1:]]
        ch = ch.replace(" (95% CI)", "").replace("International centile (Tomkinson 2025)", "International handgrip centile")
        if ": " in ch and ch.endswith(", %"):
            grp, lv = ch[:-3].split(": ", 1)
            if grp != cur:
                rows.append(("group", f"{grp}, %"))
                cur = grp
            rows.append(("row", lv, vals, True))
        else:
            cur = None
            rows.append(("row", ch, vals, False))
    return {"title": "Characteristics of the analytic sample by wave", "header": [[("Characteristic", 1)] + [(h, 1) for h in heads]],
            "rows": rows, "landscape": False, "first_cm": 7.6,
            "notes": ["Values are survey-weighted means or percentages (95% CI), accounting for ELSI calibrated weights, strata and primary "
                      "sampling units (Taylor linearisation); column headers give unweighted n.",
                      "International handgrip centile relative to Tomkinson et al. (2025) age- and sex-specific norms, after adjusting ELSI values to the norms' "
                      "reference protocol (×1.021: one hand instead of both). Handgrip in kg is the unadjusted maximum. BMI, body mass index.",
                      "Race/skin colour 'Other' = Asian-Brazilian or Indigenous. Only current smoking is reported: wave 1 asked about any past "
                      "smoking and waves 2–3 about ≥100 cigarettes in life, so former and never smoking are not comparable across waves."]}


def table2():
    t2 = pd.read_csv(D4 / "tables" / "T2_national_position.csv", keep_default_na=False)
    t2["level"] = t2["level"].astype(str).str.replace(r"\.0$", "", regex=True)
    rename = {("Education", "None"): "No schooling", ("Education", "1–4"): "1–4 years", ("Education", "5–8"): "5–8 years",
              ("Education", "9–11"): "9–11 years", ("Education", "≥12"): "≥12 years", ("Income quintile", "1"): "Q1 (lowest)",
              ("Income quintile", "5"): "Q5 (highest)"}
    val = {(r["wave"], r["group"], r["level"]): r for _, r in t2.iterrows()}
    rows = []
    for grp in [g for g in t2["group"].unique()]:
        levels = [lv for lv in t2.loc[(t2["group"] == grp) & (t2["wave"] == "W3"), "level"] if lv not in ("", "nan")]
        pref = LEVEL_ORDER.get(grp)
        if pref:
            levels = sorted(levels, key=lambda x: pref.index(x) if x in pref else len(pref))
        if grp == "All":
            items = [("All older adults", "All", False)]
        else:
            rows.append(("group", {"Income quintile": "Household income per capita, quintile", "Education": "Education"}.get(grp, grp)))
            items = [(rename.get((grp, lv), lv), lv, True) for lv in levels]
        for lab, lv, ind in items:
            cells = []
            for w in ("W1", "W2", "W3"):
                r = val.get((w, grp, lv))
                cells.append(r["% <P25 (95% CI)"] if r is not None else "—")
            r3 = val.get(("W3", grp, lv))
            cells += [r3["% ≥P75 (95% CI)"] if r3 is not None else "—", f"{float(r3['median_centile']):.1f}" if r3 is not None else "—"]
            rows.append(("row", lab, cells, ind))
    header = [[("", 1), ("Below the international P25, % (95% CI)", 3), ("Wave 3, 2023–24", 2)],
              [("", 1), (WAVE_LAB[1], 1), (WAVE_LAB[2], 1), (WAVE_LAB[3], 1), ("At or above P75,\n% (95% CI)", 1), ("Median\ncentile", 1)]]
    return {"title": "Position of older Brazilians against international handgrip norms, by group and wave", "header": header,
            "rows": rows, "landscape": True, "first_cm": 6.0,
            "notes": ["Survey-weighted estimates (95% CI). P25 and P75 refer to the age- and sex-specific international norms of "
                      "Tomkinson et al. (2025); under the norms, 25% of people are expected below P25 and 25% at or above P75. ELSI values were adjusted to the "
                      "norms' reference protocol (×1.021 for testing one hand instead of both; hydraulic dynamometer, seated, elbow flexed, three trials, "
                      "maximum value, as in the reference). Wave 2 of ELSI-Brazil was one of the studies pooled in the norms.",
                      "Education in completed years of schooling; income is household income per capita, quintiles within wave."]}


def table3():
    spec = [("Acond", ["c_index", "r2", "mae_centile", "cal_slope", "cal_in_large"]), ("Aintl", ["c_index", "r2", "mae_centile", "cal_slope"]),
            ("Tcond_low", ["auc", "pr_auc", "brier", "cal_slope"]), ("Tcond_high", ["auc", "pr_auc", "brier", "cal_slope"]),
            ("Tintl_low", ["auc", "pr_auc", "brier", "cal_slope"]), ("Tintl_high", ["auc", "pr_auc", "brier", "cal_slope"]),
            ("B", ["qwk", "macro_auc_ovr", "auc_Q4"])]
    rows = []
    for task, mets in spec:
        if task not in PRIM:
            continue
        rows.append(("group", f"{core.TASKS[task]['name']} · {PRIM[task]['algorithm']}"))
        for m in mets:
            d = 1 if m == "mae_centile" else 3
            rows.append(("row", METRIC_NAME.get(m, m), [fmt3(perf(task, m, setname="cv"), d), fmt3(perf(task, m), d)], True))
    return {"title": "Performance of the primary models in cross-validation and temporal validation",
            "header": [[("Model · algorithm / metric", 1), ("Cross-validation,\nwaves 1+2", 1), ("Temporal validation,\nwave 3", 1)]],
            "rows": rows, "landscape": False, "first_cm": 8.0,
            "notes": ["Cross-validation: mean across the 5 outer folds of nested cross-validation (3 inner folds for tuning), with 95% CI. "
                      "Wave 3: models trained on waves 1+2 and frozen; 95% CI by cluster bootstrap (primary sampling units within strata, "
                      f"{core.MCFG.n_boot if not D2.name.endswith('_fast') else 'n'} resamples). The primary algorithm was chosen on out-of-fold training performance.",
                      "C-index: probability that the model orders two participants correctly. Calibration slope: ideal 1. "
                      "Conditional z: handgrip relative to the value expected for age, sex and height (ELSI reference fitted on training waves)."]}


def table4(A):
    rows = []
    for pair, ttl in (("conditional", "Strength for body size: bottom (≤P10) vs top (≥P90)"),
                      ("international", "Against world norms: bottom (<P10) vs top (≥P75)")):
        a = A[A["pair"] == pair]
        if not len(a):
            continue
        rows.append(("group", ttl))
        for level in ("domain", "role"):
            g = a[a["level"] == level].copy()
            order = list(core.DOMAINS) if level == "domain" else list(ROLE_NAME)
            g = g.set_index("key").reindex([k for k in order if k in set(g["key"])]).reset_index()
            for _, r in g.iterrows():
                sig = (r["d_lo"] > 0) or (r["d_hi"] < 0)
                rows.append(("row", r["key"] if level == "domain" else ROLE_NAME[r["key"]],
                             [f"{r['low']:.1f} ({r['low_lo']:.1f}–{r['low_hi']:.1f})", f"{r['high']:.1f} ({r['high_lo']:.1f}–{r['high_hi']:.1f})",
                              (f"{r['diff']:+.1f} ({r['d_lo']:.1f} to {r['d_hi']:.1f})", sig)], True))
    return {"title": "Contribution of each domain to the models of the bottom and top tails (wave 3)",
            "header": [[("Domain", 1), ("Bottom-tail model,\n% of |SHAP| (95% CI)", 1), ("Top-tail model,\n% of |SHAP| (95% CI)", 1),
                        ("Top − bottom,\npercentage points (95% CI)", 1)]],
            "rows": rows, "landscape": False, "first_cm": 6.0,
            "notes": ["Share of the mean absolute SHAP value of each domain (net contribution of its variables) in the primary model of each tail, "
                      f"in the same wave 3 sample. 95% CI: {A['ci_kind'].iloc[0]}. Differences in bold: 95% CI excludes zero.",
                      "Variable types: deficit (present only in people with a health problem, e.g. diseases, disability), capacity "
                      "(varies on the positive side, e.g. gait speed, physical activity, fat-free mass), context (all others)."]}


def table_frame(t):
    """Versão plana da tabela (para o .csv)."""
    cols = [h[0].replace("\n", " ") for h in t["header"][-1]]
    if len(t["header"]) > 1:
        top = []
        for txt, span in t["header"][0]:
            top += [txt] * span
        cols = [f"{a} | {b}".strip(" |") if a else b for a, b in zip(top, cols)]
    out, grp = [], ""
    for r in t["rows"]:
        if r[0] == "group":
            grp = r[1]
            continue
        cells = [c[0] if isinstance(c, tuple) else c for c in r[2]]
        out.append([grp, r[1], *cells])
    return pd.DataFrame(out, columns=["Group", cols[0] or "Characteristic", *cols[1:]])


def write_article_docx(tables, name, prefix):
    from docx import Document
    from docx.enum.section import WD_ORIENT, WD_SECTION
    from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Cm, Pt

    def border(cell, top=None, bottom=None):
        tcPr = cell._tc.get_or_add_tcPr()
        old = tcPr.find(qn("w:tcBorders"))
        keep = {}
        if old is not None:
            for e in ("top", "bottom"):
                el = old.find(qn(f"w:{e}"))
                if el is not None:
                    keep[e] = el.get(qn("w:sz"))
            tcPr.remove(old)
        if top:
            keep["top"] = str(top)
        if bottom:
            keep["bottom"] = str(bottom)
        b = OxmlElement("w:tcBorders")
        for e in ("top", "bottom"):                       # ordem exigida pelo Word: top antes de bottom
            if e in keep:
                el = OxmlElement(f"w:{e}")
                for k, v in (("val", "single"), ("sz", keep[e]), ("space", "0"), ("color", "000000")):
                    el.set(qn(f"w:{k}"), v)
                b.append(el)
        anchor = None
        for tag in ("w:vMerge", "w:gridSpan", "w:tcW", "w:cnfStyle"):
            anchor = tcPr.find(qn(tag))
            if anchor is not None:
                break
        if anchor is not None:
            anchor.addnext(b)
        else:
            tcPr.insert(0, b)

    def put(cell, text, bold=False, align="center", indent=False, size=None):
        cell.text = ""
        lines = str(text).split("\n")
        p = cell.paragraphs[0]
        for i, ln in enumerate(lines):
            r = p.add_run(ln)
            r.bold = bold
            r.font.size = Pt(size or TABLE_PT)
            r.font.name = TABLE_FONT
            if i < len(lines) - 1:
                r.add_break()
        p.alignment = {"left": WD_ALIGN_PARAGRAPH.LEFT, "center": WD_ALIGN_PARAGRAPH.CENTER}[align]
        pf = p.paragraph_format
        pf.space_before, pf.space_after, pf.line_spacing = Pt(1), Pt(1), 1.0
        if indent:
            pf.left_indent = Cm(0.35)
        cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.BOTTOM

    doc = Document()
    st = doc.styles["Normal"]
    st.font.name, st.font.size = TABLE_FONT, Pt(TABLE_PT)
    st.element.rPr.rFonts.set(qn("w:eastAsia"), TABLE_FONT)
    for i, (num, t) in enumerate(tables):
        sec = doc.sections[0] if i == 0 else doc.add_section(WD_SECTION.NEW_PAGE)
        w, h = (29.7, 21.0) if t["landscape"] else (21.0, 29.7)
        sec.orientation = WD_ORIENT.LANDSCAPE if t["landscape"] else WD_ORIENT.PORTRAIT
        sec.page_width, sec.page_height = Cm(w), Cm(h)
        for m in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
            setattr(sec, m, Cm(2))
        usable = w - 4
        p = doc.add_paragraph()
        r = p.add_run(f"{prefix} {num}. "); r.bold = True; r.font.size = Pt(TABLE_PT + 1)
        r = p.add_run(t["title"]); r.font.size = Pt(TABLE_PT + 1)
        p.paragraph_format.space_after = Pt(4)
        ncol = sum(s for _, s in t["header"][-1])
        widths = [t["first_cm"]] + [(usable - t["first_cm"]) / (ncol - 1)] * (ncol - 1)
        nh = len(t["header"])
        tb = doc.add_table(rows=nh + len(t["rows"]), cols=ncol)
        tb.alignment = WD_TABLE_ALIGNMENT.CENTER
        tb.autofit = False
        for j, col in enumerate(tb.columns):                 # largura na grade (Word e LibreOffice)
            col.width = Cm(widths[j])
        for row in tb.rows:
            for j, c in enumerate(row.cells):
                c.width = Cm(widths[j])
        # cabeçalho (com células mescladas)
        for hr, hdr in enumerate(t["header"]):
            j = 0
            for txt, span in hdr:
                c = tb.cell(hr, j)
                if span > 1:
                    c = c.merge(tb.cell(hr, j + span - 1))
                put(c, txt, bold=True, align="left" if j == 0 else "center")
                if hr < nh - 1 and span > 1 and txt:
                    border(c, bottom=4)                   # linha fina sob o cabeçalho agrupado
                j += span
        for j in range(ncol):
            border(tb.cell(0, j), top=12)
            border(tb.cell(nh - 1, j), bottom=6)
        # corpo
        for i_r, r in enumerate(t["rows"]):
            ri = nh + i_r
            if r[0] == "group":
                c = tb.cell(ri, 0).merge(tb.cell(ri, ncol - 1))
                put(c, r[1], bold=True, align="left")
                continue
            put(tb.cell(ri, 0), r[1], align="left", indent=r[3])
            for j, v in enumerate(r[2]):
                txt, b = (v if isinstance(v, tuple) else (v, False))
                put(tb.cell(ri, j + 1), txt, bold=b)
        for j in range(ncol):
            border(tb.cell(nh + len(t["rows"]) - 1, j), bottom=12)
        for note in t["notes"]:
            q = doc.add_paragraph()
            rr = q.add_run(note); rr.font.size = Pt(TABLE_PT - 1)
            q.paragraph_format.space_before, q.paragraph_format.space_after = Pt(2), Pt(0)
    out = run.dir / "docx" / f"{name}.docx"
    doc.save(out)
    run.save_file(out, "docx", f"tabelas em formato de artigo ({prefix})")


# =============================================================================
# Fig6 / Table5 / Table6 — critérios de força baixa × saúde (08_cutoffs)
# =============================================================================
CUT_ROWS = [("Absolute grip, EWGSOP2", "u:EWGSOP2", "e:Absolute grip"), ("Absolute grip, FNIH", "u:FNIH", None),
            ("Grip/BMI, FNIH", "u:FNIH-BMI", "e:Grip/BMI"), ("International centile <P10", "u:Intl <P10", "e:International centile"),
            ("Conditional centile ≤P10", "u:Cond ≤P10", "e:Conditional centile")]
EQ_COL = {"e:Absolute grip": INK, "e:Grip/BMI": "#009E73", "e:International centile": "#56B4E9", "e:Conditional centile": "#D55E00"}
EQ_LAB = {"e:Absolute grip": "Absolute grip (EWGSOP2)", "e:Grip/BMI": "Grip/BMI", "e:International centile": "International centile",
          "e:Conditional centile": "Conditional centile"}
OUT_ORDER = ["hosp", "urgent", "visits6", "poor_srh", "adl_any", "slow_gait"]
OUT_SHORT = {"hosp": "Hospitalisation", "urgent": "Urgent care", "visits6": "≥6 medical visits", "poor_srh": "Poor self-rated health",
             "adl_any": "Any ADL difficulty", "slow_gait": "Gait <0.8 m/s"}


def cut_long():
    if D8 is None:
        return None
    return pd.read_csv(D8 / "tables" / "C_all_long.csv")


def cut_get(C, sample, outcome, crit, metric):
    r = C[(C["sample"] == sample) & (C["outcome"] == outcome) & (C["criterion"] == crit) & (C["metric"] == metric)]
    return (np.nan, np.nan, np.nan) if not len(r) else tuple(r.iloc[0][["estimate", "ci_low", "ci_high"]].astype(float))


def log_ticks(ax):
    lo, hi = ax.get_xlim()
    t = [v for v in (0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6) if lo <= v <= hi]
    ax.set_xticks(t, [f"{v:g}" for v in t])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())


def fig6():
    C = cut_long()
    if C is None:
        log.warning("Fig6: rode 08_cutoffs.py antes")
        return
    prof = pd.read_csv(D8 / "tables" / "C6_discordant_profile.csv")
    cp = pd.read_csv(D8 / "tables" / "C2_criteria_profile.csv")
    cv = Canvas(156)
    # a — quem cada critério classifica
    cv.panel(1, 2, 62, 70, "a", "Who each criterion flags", "Mean age and height of people flagged · wave 3")
    ax = cv.ax(13, 17, 46, 46)
    for _, r in cp.iterrows():
        a_, h_ = parse_ci(r["W3 flagged: age, mean"])[0], parse_ci(r["W3 flagged: height cm, mean"])[0]
        usual = r["set"].startswith("usual")
        name = r["criterion"]
        if not usual and name != "Conditional centile":
            continue
        col = "#D55E00" if "Cond" in name else INK
        ax.scatter(a_, h_, s=16 if usual else 12, color=col if usual else "white", edgecolor=col, lw=0.8, zorder=3)
        ax.annotate(name if usual else "Conditional (equal prev.)", (a_, h_), xytext=(3, 2), textcoords="offset points", fontsize=5.2, color=INK2)
    w3 = INC[INC["wave"] == 3]
    ax.scatter(np.average(w3["age"], weights=w3["weight"]), np.average(w3["height_cm"].dropna(), weights=w3.loc[w3["height_cm"].notna(), "weight"]),
               marker="+", s=30, color=GREY, zorder=3)
    ax.annotate("whole sample", (np.average(w3["age"], weights=w3["weight"]), np.average(w3["height_cm"].dropna(), weights=w3.loc[w3["height_cm"].notna(), "weight"])),
                xytext=(3, -6), textcoords="offset points", fontsize=5.2, color=GREY)
    ax.set_xlabel("Age, years", fontsize=6.2, labelpad=1); ax.set_ylabel("Height, cm", fontsize=6.2, labelpad=1)
    xl = ax.get_xlim(); ax.set_xlim(xl[0] - 1, xl[1] + 5)
    clean(ax)

    def forest(ax, metric, crits, colors, labels, sample="W3", ref=None):
        k = len(crits)
        for i, o in enumerate(OUT_ORDER):
            for j, c in enumerate(crits):
                e = cut_get(C, sample, o, c, metric)
                if np.isnan(e[0]):
                    continue
                yy = i + (j - (k - 1) / 2) * 0.17
                ax.errorbar(e[0], yy, xerr=ci_err([e[0]], [e[1]], [e[2]]), fmt="o", ms=2.6, color=colors[c], elinewidth=0.7,
                            mfc=colors[c], mew=0.5)
        if ref is not None:
            ax.axvline(ref, color=INK2, lw=0.6)
        ax.set_yticks(range(len(OUT_ORDER)), [OUT_SHORT[o] for o in OUT_ORDER], fontsize=5.8)
        ax.set_ylim(len(OUT_ORDER) - 0.5, -0.5)
        ax.axhline(1.5, color=LGREY, lw=0.6)
        clean(ax, grid_x=True)
        return [Line2D([], [], marker="o", ls="none", color=colors[c], ms=3, label=labels[c]) for c in crits]

    crits = list(EQ_COL)
    cv.panel(65, 2, 114, 70, "b", "Association with health at the same prevalence",
             "Odds ratio adjusted for age, age² and sex · every criterion flags the EWGSOP2 share (within sex) · wave 3")
    ax = cv.ax(98, 20, 78, 43)
    leg = forest(ax, "or", crits, EQ_COL, EQ_LAB, ref=1)
    ax.set_xscale("log"); ax.set_xlabel("Odds ratio (log scale)", fontsize=6.2, labelpad=1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    log_ticks(ax)
    ax.legend(handles=leg, fontsize=5.4, loc="lower left", bbox_to_anchor=(-0.36, 1.0), ncol=4, handletextpad=0.1, columnspacing=0.8,
              borderaxespad=0.2)
    cv.panel(1, 75, 88, 76, "c", "Added discrimination", "Gain in AUC over age and sex alone · same prevalence · colours as in b · wave 3")
    ax = cv.ax(31, 98, 55, 45)
    forest(ax, "dauc", crits, EQ_COL, EQ_LAB, ref=0)
    ax.set_xlabel("ΔAUC", fontsize=6.2, labelpad=1)
    cv.panel(91, 75, 88, 76, "d", "People the criteria disagree on",
             "OR vs flagged by neither · EWGSOP2 vs conditional (equal prevalence) · wave 3")
    ax = cv.ax(121, 98, 55, 45)
    dcol = {"disc:EWGSOP2 only": INK, "disc:Conditional only": "#D55E00", "disc:Both": GREY}
    w3p = prof[prof["sample"] == "W3"].set_index("group")
    dlab = {f"disc:{g}": f"{g} ({w3p.loc[g, 'age']:.0f} y, {w3p.loc[g, 'height_cm']:.0f} cm)" if g in w3p.index else g
            for g in ("EWGSOP2 only", "Conditional only", "Both")}
    leg = forest(ax, "or", list(dcol), dcol, dlab, ref=1)
    ax.set_xscale("log"); ax.set_xlabel("Odds ratio (log scale)", fontsize=6.2, labelpad=1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    log_ticks(ax)
    ax.legend(handles=leg, fontsize=5.2, loc="lower left", bbox_to_anchor=(-0.55, 1.0), ncol=2, handletextpad=0.1, columnspacing=0.8,
              borderaxespad=0.2)
    nb = int(C["n_boot"].max())
    cv.text(3, 154.8, f"Survey-weighted estimates; 95% CI by cluster bootstrap ({nb} resamples). Primary outcomes above the line.", fontsize=5.4, color=INK2)
    cv.save("Fig6_cutoffs", "critérios de força baixa × saúde: perfil, OR e ganho de AUC na mesma prevalência, discordantes")


def table5():
    C = cut_long()
    if C is None:
        raise RuntimeError("sem 08_cutoffs")
    rows = []
    for o in OUT_ORDER:
        rows.append(("group", OUT_SHORT[o] + (" (primary)" if o in ("hosp", "urgent") else "")))
        for lab, u, e in CUT_ROWS:
            cells = []
            for c in (u, e):
                if c is None:
                    cells += ["= EWGSOP2", "", ""]
                    continue
                orr, orc, da = cut_get(C, "W3", o, c, "or"), cut_get(C, "W3", o, c, "or_crude"), cut_get(C, "W3", o, c, "dauc")
                cells += [fmt3(orc, 2, "–") if not np.isnan(orc[0]) else "–", fmt3(orr, 2, "–"),
                          (f"{da[0]:+.3f} ({da[1]:.3f} to {da[2]:.3f})", da[1] > 0)]
            rows.append(("row", lab, cells, True))
    nb = int(C["n_boot"].max())
    return {"title": "Association of low handgrip strength criteria with health outcomes (wave 3)",
            "header": [[("", 1), ("Usual cut-off", 3), ("Equal prevalence (EWGSOP2 share)", 3)],
                       [("Outcome / criterion", 1), ("Crude OR (95% CI)", 1), ("Adjusted OR (95% CI)", 1), ("ΔAUC (95% CI)", 1),
                        ("Crude OR (95% CI)", 1), ("Adjusted OR (95% CI)", 1), ("ΔAUC (95% CI)", 1)]],
            "rows": rows, "landscape": True, "first_cm": 4.6,
            "notes": ["Survey-weighted logistic regression, crude and adjusted for age, age² and sex; ΔAUC: gain in the area under the ROC curve over a model "
                      f"with age, age² and sex only. 95% CI by cluster bootstrap ({nb} resamples, primary sampling units within strata). "
                      "ΔAUC in bold: 95% CI excludes zero.",
                      "Usual cut-offs: EWGSOP2 <27/<16 kg; FNIH <26/<16 kg and grip/BMI <1.0/<0.56 (men/women); international centile <10 "
                      "(Tomkinson et al., 2025; after protocol adjustment); conditional centile ≤10 (strength for age, sex and height, ELSI reference). Equal prevalence: each "
                      "score cut within sex at the EWGSOP2 prevalence (FNIH absolute = EWGSOP2 at equal prevalence).",
                      "Cross-sectional: outcomes in the previous 12 months may also lower handgrip strength. Outcomes are not predictors in the "
                      "machine-learning models except self-rated health, ADL difficulty and gait speed."]}


def table6():
    C = cut_long()
    if C is None:
        raise RuntimeError("sem 08_cutoffs")
    prof = pd.read_csv(D8 / "tables" / "C6_discordant_profile.csv")
    rows = []
    for s in ("W3", "W1–W3"):
        p = prof[prof["sample"] == s].set_index("group")
        rows.append(("group", "Wave 3" if s == "W3" else "Waves 1–3 pooled (adjusted also for wave)"))
        rows.append(("row", "Share of sample, %", [f"{p.loc[g, '% of sample (weighted)']:.1f}" for g in ("EWGSOP2 only", "Conditional only", "Both")], True))
        rows.append(("row", "Age, years; height, cm", [f"{p.loc[g, 'age']:.0f}; {p.loc[g, 'height_cm']:.0f}" for g in ("EWGSOP2 only", "Conditional only", "Both")], True))
        for o in OUT_ORDER:
            cells = []
            for g in ("EWGSOP2 only", "Conditional only", "Both"):
                e = cut_get(C, s, o, f"disc:{g}", "or")
                cells.append((fmt3(e, 2, "–"), e[1] > 1 or e[2] < 1))
            rows.append(("row", OUT_SHORT[o], cells, True))
    p7 = D8 / "tables" / "C7_discordant_by_age_profile.csv"
    if p7.exists():
        pb = pd.read_csv(p7)
        pb = pb[pb["sample"] == "W1–W3"]
        for bl in ("50–64", "65–74", "≥75"):
            q = pb[pb["age band"] == bl].set_index("group")
            rows.append(("group", f"Aged {bl}, waves 1–3 pooled"))
            rows.append(("row", "n", [f"{int(q.loc[g, 'n']):,}" if g in q.index else "0" for g in ("EWGSOP2 only", "Conditional only", "Both")], True))
            for o in ("hosp", "urgent", "visits6", "adl_any", "poor_srh", "slow_gait"):
                cells = []
                for g in ("EWGSOP2 only", "Conditional only", "Both"):
                    e = cut_get(C, "W1–W3", o, f"band:{bl}:{g}", "or")
                    cells.append("n < 30" if np.isnan(e[0]) else (fmt3(e, 2, "–"), e[1] > 1 or e[2] < 1))
                rows.append(("row", OUT_SHORT[o], cells, True))
    return {"title": "Health of people classified differently by EWGSOP2 and by the conditional centile",
            "header": [[("", 1), ("EWGSOP2 only", 1), ("Conditional only", 1), ("Both", 1)]],
            "rows": rows, "landscape": False, "first_cm": 6.0,
            "notes": ["Odds ratios (95% CI) against people flagged by neither criterion, from survey-weighted logistic regression adjusted for age, "
                      "age² and sex. Conditional centile cut at the EWGSOP2 prevalence within sex and wave, so both criteria flag the same share. "
                      "Bold: 95% CI excludes 1.",
                      "EWGSOP2 only: weak in absolute terms but not for age, sex and height; conditional only: weak for body size but above the "
                      "EWGSOP2 cut-off. Within age bands, the model is refitted in each band (adjusted for age, age², sex and wave); groups with "
                      "fewer than 30 people in a band are not estimated (by construction, the conditional-only group is almost absent at ≥75 years "
                      "and the EWGSOP2-only group is small at 50–64)."]}


SENS_LABEL = {
    "main": "Main analysis",
    "sens_protocol_tomkinson": "Protocol factor 1.072 (as pooled in the norms)",
    "sens_protocol_raw": "No protocol adjustment",
    "sens_ref_weight": "Weight added to the conditional reference",
    "sens_weighted": "Sampling weights in training",
    "sens_with_proxy": "Proxy interviews included",
    "sens_noncompleters": "Unable to perform the test placed at the bottom",
    "sens_mobility_all": "All mobility items",
    "sens_normalized": "Height-normalised norms",
    "sens_internal_quartiles": "Internal quartiles",
    "sens_train_w1": "Training on wave 1 only",
}
SENS_COLS = [  # (bloco, item no S6, rótulo da coluna)
    ("National position W3", "median international centile", "Median international centile"),
    ("National position W3", "% <P25 (95% CI)", "Below P25, %"),
    ("W3 performance", "Acond r2", "Conditional z, R²"),
    ("W3 performance", "Tcond_low auc", "Bottom, AUC"),
    ("W3 performance", "Tcond_high auc", "Top, AUC"),
    ("Asymmetry conditional (top − bottom, pp; CI refit)", "Anthropometry", "Anthropometry, top − bottom"),
    ("Asymmetry conditional (top − bottom, pp; CI refit)", "Functional status", "Functional status, top − bottom"),
    ("Asymmetry by role conditional (top − bottom, pp)", "capacity", "Capacity, top − bottom"),
]


def table7():
    """Table S6: cenários de sensibilidade (a partir de 06_sensitivity/S6_sensitivity_wide.csv)."""
    if D6 is None:
        raise RuntimeError("sem 06_sensitivity")
    w = pd.read_csv(D6 / "tables" / "S6_sensitivity_wide.csv")
    w["key"] = w["block"].astype(str) + "||" + w["item"].astype(str)
    w = w.set_index("key")
    scen = [c for c in SENS_LABEL if c in w.columns]

    def cell(block, item, sc):
        k = f"{block}||{item}"
        if k not in w.index:
            return "–"
        v = w.loc[k, sc]
        if pd.isna(v) or str(v).strip() in ("", "nan"):
            return "–"
        s = str(v).replace("–", " to ")
        bold = False
        num = s
        if "pp" in block or "role" in block:            # assimetria: negrito quando o IC exclui zero
            try:
                lo, hi = num.split("(")[1].rstrip(")").split(" to ")
                bold = float(lo) > 0 or float(hi) < 0
            except Exception:
                pass
        s = s.replace(" (", "\n(")                         # IC na linha de baixo
        s = re.sub(r"(?<![\w.])-(?=\d)", "\u2212", s)     # sinal de menos tipográfico
        return (s, bold)

    rows = []
    for sc in scen:
        cells = []
        for b, i, _ in SENS_COLS:
            c = cell(b, i, sc)
            if sc != "main" and b.startswith("National") and c == cell(b, i, "main"):
                c = "= main"                              # cenário não altera a comparação com as normas
            cells.append(c)
        rows.append(("row", SENS_LABEL[sc], cells, True))
    return {"title": "Sensitivity analyses: national position, model performance and bottom-top asymmetry (wave 3)",
            "header": [[("", 1), ("Against world norms", 2), ("Conditional reference", 3), ("Share of the model, top − bottom (pp)", 3)],
                       [("Scenario", 1)] + [(lab, 1) for _, _, lab in SENS_COLS]],
            "rows": rows, "landscape": True, "first_cm": 5.2,
            "notes": ["Wave 3 estimates with 95% CI. National position: survey-weighted, Taylor linearisation. Performance: cluster bootstrap "
                      "of primary sampling units within strata. Top − bottom: difference in each domain's (or role's) share of the absolute "
                      "SHAP value between the top (conditional centile ≥90) and bottom (≤10) models; 95% CI from bootstrap refits; bold: 95% CI "
                      "excludes zero.",
                      "–: not applicable (the scenario changes only the comparison with world norms, or only the conditional models); = main: same value as the main analysis. "
                      "In the training-on-wave-1 scenario, estimated fat-free mass (a capacity variable) replaced BMI at the top."]}


D9 = latest("09_subgroups")


def table8():
    """Table S7: desempenho por grupo sociodemográfico na W3 (09_subgroups)."""
    if D9 is None:
        raise RuntimeError("sem 09_subgroups")
    w = pd.read_csv(D9 / "tables" / "T9_subgroup_performance.csv", keep_default_na=False)
    cols = ["Conditional z: C-index", "Conditional z: R²", "Conditional bottom: AUC", "Conditional top: AUC",
            "International bottom: AUC", "International top: AUC"]

    def fmt(v):
        v = str(v)
        return v if v in ("–", "") else re.sub(r"(?<![\w.])-(?=\d)", "\u2212", v.replace(" (", "\n("))

    rows = []
    for g, sub in w.groupby("group", sort=False):
        if g != "All":
            rows.append(("group", g))
        for _, r in sub.iterrows():
            if int(r["n (conditional z)"]) == 0:
                continue
            lab = "All participants, wave 3" if g == "All" else str(r["level"])
            rows.append(("row", f"{lab} (n = {int(r['n (conditional z)']):,})", [fmt(r[c]) for c in cols], True))
    return {"title": "Model performance by sociodemographic group (wave 3, temporal validation)",
            "header": [[("", 1), ("Strength for body size", 4), ("Against world norms", 2)],
                       [("Group", 1), ("C-index", 1), ("R²", 1), ("Bottom, AUC", 1), ("Top, AUC", 1), ("Bottom, AUC", 1), ("Top, AUC", 1)]],
            "rows": rows, "landscape": True, "first_cm": 4.6,
            "notes": ["Primary algorithm of each task, trained on waves 1 and 2 and applied unchanged to wave 3. 95% CI by cluster "
                      "bootstrap of primary sampling units within strata, with metrics recomputed within each group. –: fewer than 100 "
                      "people or 20 events in the group.",
                      "Region and the socioeconomic variables are predictors; sex is not a predictor in the conditional models. Within a "
                      "group, the variable that defines it does not vary, so part of a lower within-group performance reflects that."]}


# =============================================================================
# execução
# =============================================================================
FIGS = {"Fig1": fig1, "Fig2": fig2, "Fig3": fig3, "FigCond": figCond, "Fig5": fig5, "Fig6": fig6, "FigS1": figS1, "FigS2": figS2, "FigS3": figS3, "FigS4": figS4, "FigS5": figS5}
A = None
for name in BUDGET["main"] + BUDGET["supplement"]:
    if not name.startswith("Fig"):
        continue
    try:
        if name == "Fig4":
            A = fig4()
        else:
            FIGS[name]()
        log.info(f"{name} ok")
    except Exception as e:                                # uma figura com problema não derruba as outras
        log.exception(f"{name} FALHOU: {e}")
if A is None:
    A = asymmetry_table()

run.save_table(pd.DataFrame([{"internal_id": k, "label": v, "part": "main" if k in BUDGET["main"] else "supplement"}
                             for k, v in FIG_LABEL.items()]), "figure_map", "ID interno -> número final das figuras")
BUILDERS = {"Table1": table1, "Table2": table2, "Table3": table3, "Table4": lambda: table4(A), "Table5": table5, "Table6": table6, "Table7": table7, "Table8": table8}
TABLES = {}
for k, fn in BUILDERS.items():
    try:
        TABLES[k] = fn()
        run.save_table(table_frame(TABLES[k]), k, TABLES[k]["title"])
    except Exception as e:
        log.exception(f"{k} FALHOU: {e}")
main_t = [(i + 1, TABLES[k]) for i, k in enumerate([k for k in BUDGET["main"] if k in TABLES])]
supp_t = [(i + 1, TABLES[k]) for i, k in enumerate([k for k in BUDGET["supplement"] if k in TABLES])]
if main_t:
    write_article_docx(main_t, "tables_main", "Table")
if supp_t:
    write_article_docx(supp_t, "tables_supplement", "Table S")
run.finish(notes=("MODO RÁPIDO em parte das entradas; " if FAST_USED else "") + f"main: {', '.join(BUDGET['main'])}")

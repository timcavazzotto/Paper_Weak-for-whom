"""
06_sensitivity.py — compara as estimativas-chave entre o cenário principal e as sensibilidades
=============================================================================================
Para cada cenário, usa a execução concluída MAIS RECENTE de 01–04 (nesta versão ou anterior) e junta:
  - tamanho da amostra analítica por onda (01)
  - desempenho na W3 dos modelos primários (02)
  - assimetria base × topo por domínio e por papel (03; IC com reajuste quando disponível)
  - regressão quantílica: β(0,90) − β(0,10) de termos-chave (03)
  - posição nacional: centil mediano e % ≥P75 na W3 (04)
Rode sem SR_SCENARIO (o run_scenarios.py chama esta etapa no fim).

Saídas em outputs/v<VERSION>/main/06_sensitivity/
  tables/S6_sensitivity_long.csv   uma linha por cenário × estimativa (com IC)
  tables/S6_sensitivity_wide.csv   estimativas nas linhas, cenários nas colunas
  figures/S6_sensitivity.pdf/.png  AUC base × topo e diferenças de assimetria por cenário
  docx/S6_sensitivity.docx
"""
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import sr_core as core

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

run = core.Run("06_sensitivity", __file__)
log = run.log
P = core.PALETTE
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#ecebe7", "legend.frameon": False})
_MAJOR = core.VERSION.split(".")[0]
# só versões com a MESMA MAJOR: uma MAJOR muda desfecho/harmonização, e misturar cenários de MAJORs diferentes
# compararia coisas diferentes (ex.: v4 sem ajuste de protocolo × v5 com ajuste)
VERSIONS = [core.VERSION] + sorted([p.name[1:] for p in core.OUTPUTS.glob("v*")
                                    if p.is_dir() and p.name[1:] != core.VERSION and p.name[1:].split(".")[0] == _MAJOR],
                                   key=core._version_key, reverse=True)
KEY_DOMAINS = ["Anthropometry", "Functional status", "Physical performance", "Context", "Morbidity",
               "Life course", "Behavioural", "Social & wellbeing"]
KEY_TERMS = ["ffm_est", "rfm", "adl_n", "mobility_n", "gait_speed", "balance_score", "mvpa_min_week", "heavy_work",
             "region=North", "region=Northeast", "childhood_hunger", "parental_education"]
PERF = {"Acond": ["c_index", "r2"], "Aintl": ["c_index", "r2"], "B": ["qwk", "auc_Q4"],
        "Tcond_low": ["auc"], "Tcond_high": ["auc"], "Tintl_low": ["auc"], "Tintl_high": ["auc"]}


def latest(stage, sc):
    for v in VERSIONS:
        for fast in ([True, False] if core.FAST else [False]):
            d = core.stage_dir(stage, v, sc, fast=fast)
            m = d / "manifest.json"
            if m.exists() and json.loads(m.read_text(encoding="utf-8")).get("status") == "complete":
                return d
    return None


def upstream_dir(man, stage, sc):
    """Pasta da etapa anterior registrada no manifesto; se o caminho é de outro computador, reconstrói pela versão."""
    up = man["upstream"][stage]
    p = Path(up["path"])
    if p.exists():
        return p
    for fast in (False, True):
        d = core.stage_dir(stage, up["version"], sc, fast=fast)
        if (d / "manifest.json").exists():
            return d
    raise SystemExit(f"PARADO: não encontrei {stage} v{up['version']} ({sc})")


def first_existing(folder, *names):
    for n in names:
        if (folder / n).exists():
            return folder / n
    return None


rows = []


def add(sc, block, item, est, lo=np.nan, hi=np.nan):
    rows.append({"scenario": sc, "block": block, "item": item, "estimate": est, "ci_low": lo, "ci_high": hi})


found = {}
for sc in core.SCENARIOS:
    d3 = latest("03_explain", sc)
    if d3 is None:
        log.info(f"{sc}: sem 03 concluído — fora da comparação")
        continue
    man3 = json.loads((d3 / "manifest.json").read_text(encoding="utf-8"))
    d2 = upstream_dir(man3, "02_model", sc)
    d1 = upstream_dir(man3, "01_prepare", sc)
    d4 = latest("04_report", sc)
    found[sc] = {"01": str(d1), "02": str(d2), "03": str(d3), "04": str(d4) if d4 else None}
    for k, folder in (("02", d2), ("03", d3)):
        run.add_input(folder / "manifest.json", f"{sc}_{k}_manifest")

    # 01 — amostra
    flow = pd.read_csv(d1 / "tables" / "S2_flowchart.csv")
    for w, g in flow.groupby("wave"):
        add(sc, "Sample", f"n analytic {w}", float(g["remaining"].iloc[-1]))

    # 02 — desempenho (primário, M3, W3)
    prim = json.loads((d2 / "tables" / "primary_models.json").read_text(encoding="utf-8"))
    perf = pd.read_csv(d2 / "tables" / "performance_long.csv")
    for task, mets in PERF.items():
        if task not in prim:
            continue
        g = perf[(perf["task"] == task) & (perf["algorithm"] == prim[task]["algorithm"]) & (perf["layer"] == "M3") & (perf["set"] == "test")]
        for m in mets:
            r = g[g["metric"] == m]
            if len(r):
                add(sc, "W3 performance", f"{task} {m}", r["value"].iloc[0], r["ci_low"].iloc[0], r["ci_high"].iloc[0])

    # 03 — assimetria (IC com reajuste se houver)
    t3 = d3 / "tables"
    rf = first_existing(t3, "F4_asymmetry_refit.csv")
    if rf is not None:
        a = pd.read_csv(rf)
        a_dom, a_role = a[a["level"] == "domain"].rename(columns={"key": "domain"}), a[a["level"] == "role"].rename(columns={"key": "role"})
        ci_kind = "refit"
    else:
        dom_p = first_existing(t3, "asymmetry_domain.csv", "F4_asymmetry_domain.csv")
        a_dom = pd.read_csv(dom_p) if dom_p else pd.DataFrame()
        a_role = pd.read_csv(t3 / "asymmetry_role.csv") if (t3 / "asymmetry_role.csv").exists() else pd.DataFrame()
        ci_kind = "W3 only"
    for _, r in a_dom.iterrows():
        if r["domain"] in KEY_DOMAINS and r["pair"] in ("conditional", "international"):
            add(sc, f"Asymmetry {r['pair']} (top − bottom, pp; CI {ci_kind})", r["domain"], r["diff_high_minus_low_pp"],
                r["diff_ci_low"], r["diff_ci_high"])
    for _, r in a_role.iterrows():
        if r["pair"] in ("conditional", "international"):
            add(sc, f"Asymmetry by role {r['pair']} (top − bottom, pp)", r["role"], r["diff_high_minus_low_pp"],
                r["diff_ci_low"], r["diff_ci_high"])
    for pair, g in (a_dom.groupby("pair") if len(a_dom) else []):
        add(sc, "Asymmetry: rank agreement", f"Spearman bottom×top ({pair})", spearmanr(g["share_low_pct"], g["share_high_pct"]).statistic)

    # 03 — regressão quantílica
    qa_p = first_existing(t3, "qr_asymmetry.csv")
    if qa_p is not None:
        qa = pd.read_csv(qa_p)
        for _, r in qa[qa["term"].isin(KEY_TERMS)].iterrows():
            add(sc, "Quantile regression β(0.90) − β(0.10)", r["term"], r["diff_p90_minus_p10"], r["diff_ci_low"], r["diff_ci_high"])

    # 04 — posição nacional na W3
    if d4 is not None and (d4 / "tables" / "T2_national_position.csv").exists():
        t2 = pd.read_csv(d4 / "tables" / "T2_national_position.csv")
        a = t2[(t2["wave"] == "W3") & (t2["group"] == "All")]
        if len(a):
            add(sc, "National position W3", "median international centile", a["median_centile"].iloc[0])
            for col in ("% <P10 (95% CI)", "% <P25 (95% CI)", "% ≥P75 (95% CI)"):
                if col in a.columns:
                    add(sc, "National position W3", col, float(str(a[col].iloc[0]).split(" ")[0]))
                    rows[-1]["text"] = a[col].iloc[0]
    log.info(f"{sc}: 01 {d1.parent.parent.name} | 02 {d2.parent.parent.name} | 03 {d3.parent.parent.name} | 04 {'—' if d4 is None else d4.parent.parent.name}")

if not rows:
    raise SystemExit("PARADO: nenhum cenário com 03 concluído.")
long = pd.DataFrame(rows)
run.save_table(long, "sensitivity_long", "estimativas-chave por cenário (IC 95%)", item="S6")
run.save_json(found, "sensitivity_sources", "pastas de origem de cada cenário")


def cell(r):
    if isinstance(r.get("text"), str):
        return r["text"]
    if pd.isna(r["estimate"]):
        return ""
    d = 0 if r["item"].startswith("n analytic") else (1 if "pp" in r["block"] or "centile" in r["item"] else 3)
    s = f"{r['estimate']:.{d}f}"
    if not pd.isna(r["ci_low"]):
        s += f" ({r['ci_low']:.{d}f} to {r['ci_high']:.{d}f})"
    return s


long["cell"] = long.apply(cell, axis=1)
order = [s for s in core.SCENARIOS if s in set(long["scenario"])]
wide = long.pivot_table(index=["block", "item"], columns="scenario", values="cell", aggfunc="first", sort=False)[order].reset_index()
run.save_table(wide, "sensitivity_wide", "estimativas-chave (linhas) por cenário (colunas)", item="S6")

# figura: AUC base vs topo (condicional) e diferenças de assimetria por cenário
fig, axes = plt.subplots(1, 2, figsize=(7.4, 4.0), gridspec_kw={"width_ratios": [1, 1.6]})
ax = axes[0]
sc_auc = [s for s in order if {"Tcond_low auc", "Tcond_high auc"} <= set(long.loc[long["scenario"] == s, "item"])]
for i, s in enumerate(sc_auc):
    for item, colr, off in (("Tcond_low auc", P["low"], -0.12), ("Tcond_high auc", P["high"], 0.12)):
        r = long[(long["scenario"] == s) & (long["item"] == item)].iloc[0]
        ax.errorbar(r["estimate"], i + off, xerr=[[max(r["estimate"] - r["ci_low"], 0)], [max(r["ci_high"] - r["estimate"], 0)]],
                    fmt="o", ms=4.5, color=colr, ecolor=colr, elinewidth=1,
                    label=("Bottom (≤P10)" if "low" in item else "Top (≥P90)") if i == 0 else None)
ax.set_yticks(range(len(sc_auc)), sc_auc)
ax.invert_yaxis()
ax.set_xlabel("AUC in W3 (conditional tails)")
ax.set_title("Predictability of each tail", loc="left", fontsize=8.5)
ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
ax = axes[1]
blk = [b for b in long["block"].unique() if b.startswith("Asymmetry conditional")]
sub = long[long["block"].isin(blk) & long["item"].isin(KEY_DOMAINS[:5])]
doms = [d for d in KEY_DOMAINS[:5] if d in set(sub["item"])]
scs = [s for s in order if s in set(sub["scenario"])]
for j, s in enumerate(scs):
    g = sub[sub["scenario"] == s].set_index("item").reindex(doms)
    y = np.arange(len(doms)) + (j - (len(scs) - 1) / 2) * 0.11
    # principal em preto; sensibilidades em cinza com marcadores distintos (laranja/azul ficam para base/topo)
    colr = P["ink"] if s == "main" else P["norm_line"]
    mk = "o" if s == "main" else "sDv^<>p"[(j - 1) % 7]
    ax.errorbar(g["estimate"], y, xerr=[(g["estimate"] - g["ci_low"]).clip(lower=0).fillna(0), (g["ci_high"] - g["estimate"]).clip(lower=0).fillna(0)],
                fmt=mk, ms=3.8 if s != "main" else 5, color=colr, ecolor=colr, elinewidth=0.8, label=s)
ax.axvline(0, color=P["ink2"], lw=0.7)
ax.set_yticks(range(len(doms)), doms)
ax.invert_yaxis()
ax.set_xlabel("Top − bottom share of |SHAP|, percentage points")
ax.set_title("Asymmetry by domain (conditional tails)", loc="left", fontsize=8.5)
ax.legend(fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=4)
fig.tight_layout()
run.save_figure(fig, "sensitivity", "AUC base × topo e assimetria por domínio entre cenários", item="S6")
plt.close(fig)

try:
    from docx import Document
    from docx.shared import Pt
    doc = Document()
    doc.styles["Normal"].font.name, doc.styles["Normal"].font.size = "Arial", Pt(7)
    doc.add_paragraph().add_run("Supplementary Table S6. Key estimates across sensitivity scenarios").bold = True
    t = doc.add_table(rows=1, cols=len(wide.columns))
    t.style = "Table Grid"
    for j, c in enumerate(wide.columns):
        t.rows[0].cells[j].text = str(c)
    for _, row in wide.iterrows():
        cells = t.add_row().cells
        for j, v in enumerate(row):
            cells[j].text = "" if pd.isna(v) else str(v)
    p = run.dir / "docx" / "S6_sensitivity.docx"
    doc.save(p)
    run.save_file(p, "docx", "tabela S6")
except Exception as e:  # o .docx é conveniência; as tabelas .csv são a fonte
    log.warning(f"docx não gerado: {e}")

run.finish(notes=f"cenários: {', '.join(order)}")

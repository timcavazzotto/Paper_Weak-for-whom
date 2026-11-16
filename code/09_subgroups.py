"""
09_subgroups — desempenho dos modelos por grupo sociodemográfico na W3 (justiça do modelo; TRIPOD+AI itens 14 e 23a).

O que faz
  Lê as predições da validação temporal (02_model, algoritmo primário, camada M3) e calcula, dentro de cada grupo da
  W3 (sexo, faixa etária, região, escolaridade, raça/cor, quintil de renda), as mesmas métricas do 02:
    - força para o corpo (Acond) e frente às normas (Aintl): C-index, R², inclinação e calibração na média;
    - caudas (Tcond_low/high, Tintl_low/high): AUC, inclinação de calibração e razão observado/previsto.
  IC 95% por bootstrap de conglomerados (UPAs dentro dos estratos) da W3 inteira; em cada reamostra as métricas são
  recalculadas dentro de cada grupo.

Observação
  Sexo e região podem ser preditores (região é; sexo não entra nas tarefas condicionais, porque define a referência).
  Avaliar o desempenho dentro de um grupo continua válido: a pergunta é se o modelo ordena e calibra igualmente bem as
  pessoas de cada grupo. Dentro de um grupo, a variável que o define não varia, então o desempenho pode cair só por isso.

Saídas (tables/): T9_subgroup_long.csv · T9_subgroup_performance.csv (formatada)
Uso (na raiz):  .venv\\Scripts\\python code\\09_subgroups.py      (B_SUB=500; SR_FAST=1 -> 50)
"""
import json
import os
import time
import warnings

import numpy as np
import pandas as pd

import sr_core as core

warnings.filterwarnings("ignore")
run = core.Run("09_subgroups", __file__)
log = run.log
up1 = run.require("01_prepare")
up2 = run.require("02_model")
B = int(os.environ.get("B_SUB", 50 if core.FAST else 500))
MIN_N, MIN_EVENTS = 100, 20

primary = json.loads((up2 / "tables" / "primary_models.json").read_text(encoding="utf-8"))
ALL = pd.read_parquet(up1 / "data" / "analytic_all.parquet")
W3 = ALL[ALL["wave"] == 3].copy()

TASKS = ["Acond", "Aintl", "Tcond_low", "Tcond_high", "Tintl_low", "Tintl_high"]
TASK_LAB = {"Acond": "Strength for body size (conditional z)", "Aintl": "Strength vs world norms (international z)",
            "Tcond_low": "Conditional bottom (≤P10)", "Tcond_high": "Conditional top (≥P90)",
            "Tintl_low": "International bottom (<P10)", "Tintl_high": "International top (≥P75)"}


def edu_group(y):
    return pd.cut(y, [-0.5, 0.5, 4.5, 8.5, 11.5, 99], labels=["None", "1–4 years", "5–8 years", "9–11 years", "≥12 years"]).astype(str)


GROUPS = {
    "Sex": W3["male"].map({1: "Men", 0: "Women"}),
    "Age": pd.cut(W3["age"], [0, 65, 75, 200], right=False, labels=["50–64", "65–74", "≥75"]).astype(str),
    "Region": W3["region"].astype(str),
    "Education": edu_group(W3["education_years"]),
    "Race/skin colour": W3["race"].astype(str),
    "Income quintile": W3["income_quintile"].map(lambda q: f"Q{int(q)}" if pd.notna(q) else np.nan),
}
for k, s in GROUPS.items():
    W3[f"g:{k}"] = s.astype(object).where(s.astype(str) != "nan", np.nan)

# predições do algoritmo primário (camada M3), unidas aos grupos pelo row_id
P = {}
for t in TASKS:
    algo = primary[t]["algorithm"]
    f = up2 / "predictions" / f"test_{t}_{algo}_M3.parquet"
    run.add_input(f, f"test_{t}")
    p = pd.read_parquet(f)
    P[t] = p.merge(W3[["row_id"] + [f"g:{k}" for k in GROUPS]], on="row_id", how="left", validate="1:1")
    log.info(f"{t}: {algo}, n = {len(p):,}")


def metrics(t, y, p):
    if core.TASKS[t]["kind"] == "regression":
        if len(y) < MIN_N:
            return {}
        m = core.metrics_reg(y, p)
        return {"c_index": m["c_index"], "r2": m["r2"], "cal_slope": m["cal_slope"], "cal_in_large": m["cal_in_large"]}
    if len(y) < MIN_N or y.sum() < MIN_EVENTS or (len(y) - y.sum()) < MIN_EVENTS:
        return {}
    m = core.metrics_bin(y, p)
    return {"auc": m["auc"], "cal_slope": m["cal_slope"], "oe_ratio": float(np.mean(y) / np.mean(p))}


def compute(idx_by_task):
    out = {}
    for t in TASKS:
        d = P[t].iloc[idx_by_task[t]] if idx_by_task is not None else P[t]
        y_all, p_all = d["observed"].to_numpy(float), d["predicted"].to_numpy(float)
        for k, v in metrics(t, y_all, p_all).items():
            out[(t, "All", "All", k)] = v
        for g in GROUPS:
            lab = d[f"g:{g}"].to_numpy(object)
            for lv in pd.unique(lab[pd.notna(lab)]):
                m = lab == lv
                for k, v in metrics(t, y_all[m], p_all[m]).items():
                    out[(t, g, lv, k)] = v
    return out


t0 = time.time()
est = compute(None)
# mesmas UPAs reamostradas para todas as tarefas: índices por tarefa a partir do sorteio de UPAs
ref = W3[["row_id", "stratum", "cluster"]].reset_index(drop=True)
boot_idx = core.cluster_boot_indices(ref["stratum"], ref["cluster"], B, seed=core.SEED)
pos = {t: pd.Series(np.arange(len(P[t])), index=P[t]["row_id"]) for t in TASKS}
boots = []
for i, ix in enumerate(boot_idx):
    rid = ref["row_id"].to_numpy()[ix]
    idx = {t: pos[t].reindex(rid).dropna().astype(int).to_numpy() for t in TASKS}
    try:
        boots.append(compute(idx))
    except (ValueError, np.linalg.LinAlgError):
        continue
    if (i + 1) % 100 == 0:
        log.info(f"  bootstrap {i + 1}/{B} ({(time.time() - t0) / 60:.1f} min)")
bt = pd.DataFrame(boots)

rows = []
for key, v in est.items():
    t, g, lv, k = key
    s = bt[key].dropna() if key in bt else pd.Series(dtype=float)
    d = P[t] if g == "All" else P[t][P[t][f"g:{g}"] == lv]
    rows.append({"task": t, "task_label": TASK_LAB[t], "group": g, "level": lv, "metric": k, "estimate": v,
                 "ci_low": s.quantile(0.025) if len(s) else np.nan, "ci_high": s.quantile(0.975) if len(s) else np.nan,
                 "n": len(d), "events": int(d["observed"].sum()) if core.TASKS[t]["kind"] == "binary" else np.nan,
                 "n_boot": len(s)})
L = pd.DataFrame(rows)
run.save_table(L, "T9_subgroup_long", "desempenho na W3 por grupo sociodemográfico (formato longo), IC 95% bootstrap de conglomerados")

LEVEL_ORDER = {"Sex": ["Women", "Men"], "Age": ["50–64", "65–74", "≥75"],
               "Region": ["North", "Northeast", "Southeast", "South", "Central-West"],
               "Education": ["None", "1–4 years", "5–8 years", "9–11 years", "≥12 years"],
               "Race/skin colour": ["White", "Brown", "Black", "Other"], "Income quintile": [f"Q{q}" for q in range(1, 6)]}


def cell(t, g, lv, k, d=2):
    r = L[(L.task == t) & (L.group == g) & (L.level == lv) & (L.metric == k)]
    if not len(r):
        return "–"
    r = r.iloc[0]
    return f"{r['estimate']:.{d}f} ({r['ci_low']:.{d}f} to {r['ci_high']:.{d}f})"


wide = []
for g in ["All"] + list(GROUPS):
    for lv in (["All"] if g == "All" else LEVEL_ORDER[g]):
        nA = len(P["Acond"]) if g == "All" else int((P["Acond"][f"g:{g}"] == lv).sum())
        wide.append({"group": g, "level": lv, "n (conditional z)": nA,
                     "Conditional z: C-index": cell("Acond", g, lv, "c_index"),
                     "Conditional z: R²": cell("Acond", g, lv, "r2"),
                     "Conditional z: calibration slope": cell("Acond", g, lv, "cal_slope"),
                     "Conditional bottom: AUC": cell("Tcond_low", g, lv, "auc"),
                     "Conditional top: AUC": cell("Tcond_high", g, lv, "auc"),
                     "International z: R²": cell("Aintl", g, lv, "r2"),
                     "International bottom: AUC": cell("Tintl_low", g, lv, "auc"),
                     "International top: AUC": cell("Tintl_high", g, lv, "auc")})
run.save_table(pd.DataFrame(wide), "T9_subgroup_performance",
               f"desempenho na W3 por grupo (IC 95%: bootstrap de conglomerados, {len(bt)} reamostras; mínimo {MIN_N} pessoas e {MIN_EVENTS} eventos)")
for t in ("Acond", "Tcond_low", "Tcond_high"):
    k = "c_index" if t == "Acond" else "auc"
    x = L[(L.task == t) & (L.metric == k)]
    log.info(f"{t} {k}: faixa entre grupos {x['estimate'].min():.3f}–{x['estimate'].max():.3f}")
run.finish(notes=f"B={len(bt)}")

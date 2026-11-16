"""
02_model.py — treino (W1+W2), CV aninhada, tuning e validação temporal (W3)
==========================================================================
Tarefas (definidas em sr_core.TASKS):
  Acond       regressão do z CONDICIONAL (força para idade, sexo e altura)      [principal]
  B           classificação ordinal do quartil internacional
  Aintl       regressão do z internacional (posição frente às normas mundiais)
  Tcond_low   base condicional (≤P10)   | Tcond_high  topo condicional (≥P90)
  Tintl_low   base mundial (<P10)       | Tintl_high  topo mundial (≥P75)
Nas tarefas condicionais, idade, sexo e altura NÃO entram como preditores (já estão no esperado).

Para cada tarefa:
  1. CV aninhada no treino (externa = desempenho honesto, interna = tuning) para cada
     algoritmo com TODOS os preditores (camada M3) -> predições fora da dobra (OOF).
  2. Algoritmo primário = maior métrica principal nas OOF (pré-especificado).
  3. Ajuste final de cada algoritmo no treino inteiro -> predição na W3, IC 95% por
     bootstrap de conglomerados (UPA dentro do estrato).
  4. Camadas adicionais da tarefa (M0, M1, M2) com o algoritmo primário.
A W3 só é usada para AVALIAR modelos já escolhidos e ajustados.

Uso (na raiz do projeto):
    python code/02_model.py                          # completo (várias horas; ver README)
    $env:SR_FAST="1"; python code/02_model.py        # teste rápido (grava em 02_model_fast)
    $env:SR_TASKS="Acond,Tcond_low,Tcond_high"; python code/02_model.py   # só algumas tarefas

Saídas em outputs/v<VERSION>/<cenário>/02_model/
  models/<tarefa>_<algoritmo>_<camada>.joblib     modelos finais
  predictions/oof_*.parquet | test_*.parquet      predições (row_id, wave, observado, previsto)
  tables/performance_long.csv                     todas as métricas (CV, OOF, teste com IC 95%)
  tables/T3_performance.csv                       tabela principal de desempenho do artigo
  tables/best_params.json | primary_models.json   hiperparâmetros e algoritmo primário por tarefa
  tables/coherence_Aintl_B.csv                    concordância Aintl x B no teste
"""
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

import sr_core as core

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

run = core.Run("02_model", __file__)
log = run.log
up = run.require("01_prepare")
data = pd.read_parquet(up / "data" / "analytic.parquet")
run.add_input(up / "data" / "analytic.parquet", "analytic")
M = core.MCFG

train = data[data["split"] == "train"].reset_index(drop=True)
test = data[data["split"] == "test"].reset_index(drop=True)
log.info(f"treino (W{list(core.CFG.train_waves)}): {len(train):,} | teste (W{list(core.CFG.test_waves)}): {len(test):,}")
log.info(f"tarefas: {', '.join(M.tasks)}")

# dobras externas fixas e iguais para todas as tarefas (estratificadas pelo quartil internacional)
outer = list(StratifiedKFold(M.outer_folds, shuffle=True, random_state=core.SEED)
             .split(train, train["quartile_intl"].astype(int)))
KIND = {t: core.TASKS[t]["kind"] for t in core.TASKS}


def xyw(df, task, names):
    tgt = core.TASKS[task]["target"]
    d = df[df[tgt].notna()]
    y = d[tgt].to_numpy(float) if KIND[task] == "regression" else d[tgt].astype(int).to_numpy()
    w = d["weight_norm"].to_numpy() if M.weighted_training else None
    return d, d[names], y, w


def search(task, algo, names, X, y, w, seed):
    pipe, grid = core.make_pipeline(task, algo, names)
    strat = pd.cut(y, [-np.inf, *core.Z_CUTS.values(), np.inf], labels=False) if KIND[task] == "regression" else y
    inner = list(StratifiedKFold(M.inner_folds, shuffle=True, random_state=seed).split(X, strat))
    n_grid = int(np.prod([len(v) for v in grid.values()]))
    rs = RandomizedSearchCV(pipe, grid, n_iter=min(M.n_iter, n_grid), scoring=core.scorer(task), cv=inner,
                            refit=True, n_jobs=-1, random_state=seed, error_score="raise")
    rs.fit(X, y, **({"clf__sample_weight": w} if w is not None else {}))
    return rs


def pred_frame(task, d, y, p, fold=None):
    out = d[["row_id", "wave", "stratum", "cluster"]].reset_index(drop=True).copy()
    out["observed"] = y
    if KIND[task] == "ordinal":
        for k in range(p.shape[1]):
            out[f"p_Q{k + 1}"] = p[:, k]
        out["predicted_class"] = p.argmax(axis=1)
    else:
        out["predicted"] = p
    if fold is not None:
        out["fold"] = fold
    return out


rows, best_params, primary = [], {}, {}
SHOW = ("c_index", "r2", "mae_centile", "qwk", "macro_auc_ovr", "auc_Q4", "auc", "pr_auc", "prevalence")


def record(task, algo, layer, setname, met, ci=None, fold=None):
    for k, v in met.items():
        r = {"task": task, "algorithm": algo, "layer": layer, "set": setname, "fold": fold, "metric": k, "value": v}
        if ci and k in ci:
            r["ci_low"], r["ci_high"] = ci[k]
        rows.append(r)


def fit_evaluate(task, algo, layer):
    """CV aninhada (OOF) + ajuste final + avaliação na W3 para uma combinação."""
    t0 = time.time()
    names = core.task_features(task, layer)
    tag = f"{task}_{algo}_{layer}"
    log.info(f"[{tag}] {len(names)} preditores | CV aninhada {M.outer_folds}x{M.inner_folds}, {M.n_iter} combinações")
    d_tr, X, y, w = xyw(train, task, names)
    pos = {rid: i for i, rid in enumerate(train["row_id"])}
    idx = d_tr["row_id"].map(pos).to_numpy()

    # 1. CV aninhada -> OOF
    oof = np.full((len(y), 4) if KIND[task] == "ordinal" else (len(y),), np.nan)
    fold_id = np.full(len(y), -1)
    params = []
    for f, (tr_i, va_i) in enumerate(outer):
        tr_mask, va_mask = np.isin(idx, tr_i), np.isin(idx, va_i)
        rs = search(task, algo, names, X[tr_mask], y[tr_mask], None if w is None else w[tr_mask], core.SEED + f)
        oof[va_mask] = core.predict(task, rs.best_estimator_, X[va_mask])
        fold_id[va_mask] = f
        record(task, algo, layer, "cv_fold", core.metrics(task, y[va_mask], oof[va_mask]), fold=f)
        params.append(rs.best_params_)
    m_oof = core.metrics(task, y, oof)
    record(task, algo, layer, "oof_pooled", m_oof)
    run.save_data(pred_frame(task, d_tr, y, oof, fold_id), f"oof_{tag}", "predições fora da dobra", sub="predictions")
    log.info(f"[{tag}] OOF: " + " | ".join(f"{k}={v:.3f}" for k, v in m_oof.items() if k in SHOW))

    # 2. ajuste final no treino inteiro -> W3
    rs = search(task, algo, names, X, y, w, core.SEED + 100)
    model = rs.best_estimator_
    joblib.dump(model, run.dir / "models" / f"{tag}.joblib")
    run.save_file(run.dir / "models" / f"{tag}.joblib", "model", f"modelo final {tag}")
    best_params[tag] = {"final": rs.best_params_, "outer_folds": params}

    d_te, Xt, yt, _ = xyw(test, task, names)
    pt = core.predict(task, model, Xt)
    m_te = core.metrics(task, yt, pt)
    ci = core.bootstrap_ci(lambda a, b: core.metrics(task, a, b), yt, pt, d_te["stratum"], d_te["cluster"])
    record(task, algo, layer, "test", m_te, ci)
    run.save_data(pred_frame(task, d_te, yt, pt), f"test_{tag}", "predições na validação temporal", sub="predictions")
    log.info(f"[{tag}] W3 : " + " | ".join(f"{k}={v:.3f}" for k, v in m_te.items() if k in SHOW)
             + f"  ({(time.time() - t0) / 60:.1f} min)")
    return m_oof, pt


test_preds = {}
for task in M.tasks:
    cfg = core.TASKS[task]
    key = cfg["metric"]
    oof_scores = {}
    for algo in cfg["algorithms"]:
        m_oof, pt = fit_evaluate(task, algo, "M3")
        oof_scores[algo] = m_oof[key]
        test_preds[(task, algo, "M3")] = pt
    best = max(oof_scores, key=oof_scores.get)
    primary[task] = {"algorithm": best, "selection_metric": key, "oof_scores": oof_scores,
                     "target": cfg["target"], "kind": cfg["kind"], "features": core.task_features(task, "M3"),
                     "rule": "maior métrica principal nas predições OOF do treino (pré-especificado)"}
    log.info(f"tarefa {task}: algoritmo primário = {best} ({key} OOF = {oof_scores[best]:.3f})")
    for layer in [L for L in cfg["layers"] if L != "M3"]:
        _, pt = fit_evaluate(task, best, layer)
        test_preds[(task, best, layer)] = pt

# ---------------------------------------------------------------------------
# coerência Aintl x B no teste (algoritmos primários, camada M3)
# ---------------------------------------------------------------------------
if {"Aintl", "B"} <= set(M.tasks):
    from sklearn.metrics import cohen_kappa_score

    pA = pd.Series(test_preds[("Aintl", primary["Aintl"]["algorithm"], "M3")], index=test.index[test["hgs_z"].notna()])
    if core.CFG.quartile_reference == "international":
        qA = pd.cut(pA, [-np.inf, core.Z_CUTS["P25"], core.Z_CUTS["P50"], core.Z_CUTS["P75"], np.inf], labels=False)
    else:
        qA = core.assign_quartile(pA, core.internal_quartile_cuts(train.loc[train["hgs_z"].notna(), "hgs_z"]))
    qB = pd.Series(test_preds[("B", primary["B"]["algorithm"], "M3")].argmax(axis=1), index=test.index[test["quartile"].notna()])
    both = pd.concat([qA.rename("from_A"), qB.rename("from_B")], axis=1).dropna().astype(int)
    coh = pd.crosstab(both["from_A"].map(dict(enumerate(core.QUARTILE_LABELS))),
                      both["from_B"].map(dict(enumerate(core.QUARTILE_LABELS)))).reset_index()
    run.save_table(coh, "coherence_Aintl_B", "quartil derivado do Aintl (linhas) x quartil previsto pelo B (colunas), W3")
    k_ab = cohen_kappa_score(both["from_A"], both["from_B"], weights="quadratic")
    primary["coherence_Aintl_B"] = {"weighted_kappa": k_ab, "exact_agreement": float((both["from_A"] == both["from_B"]).mean()),
                                    "n": len(both)}
    log.info(f"coerência Aintl x B (W3): kappa ponderado = {k_ab:.3f}")

# ---------------------------------------------------------------------------
# tabelas
# ---------------------------------------------------------------------------
perf = pd.DataFrame(rows)
run.save_table(perf, "performance_long", "todas as métricas: dobras da CV, OOF e teste (IC 95% bootstrap de conglomerados)")

KEY = {"regression": ["c_index", "r2", "mae_centile", "cal_slope", "cal_in_large", "auc_lt_P10", "auc_ge_P90"],
       "ordinal": ["qwk", "macro_auc_ovr", "auc_Q1", "auc_Q4", "auc_ge_Q2", "balanced_accuracy", "sens_Q4"],
       "binary": ["auc", "pr_auc", "prevalence", "brier", "cal_slope", "cal_in_large"]}
t3 = []
for (task, algo, layer), g in perf.groupby(["task", "algorithm", "layer"], sort=False):
    r = {"Task": task, "Model": core.TASKS[task]["name"], "Algorithm": algo, "Layer": layer,
         "Primary": "yes" if primary[task]["algorithm"] == algo else ""}
    for m in KEY[KIND[task]]:
        cv = g[(g["set"] == "cv_fold") & (g["metric"] == m)]["value"]
        te = g[(g["set"] == "test") & (g["metric"] == m)].iloc[0]
        r[f"{m} | CV W1+W2, mean (SD)"] = f"{cv.mean():.3f} ({cv.std():.3f})"
        r[f"{m} | W3, estimate (95% CI)"] = f"{te['value']:.3f} ({te['ci_low']:.3f}–{te['ci_high']:.3f})"
    t3.append(r)
run.save_table(pd.DataFrame(t3), "performance", "desempenho por tarefa, algoritmo e camada", item="T3")
run.save_json(best_params, "best_params", "hiperparâmetros escolhidos (final e por dobra externa)")
run.save_json(primary, "primary_models", "algoritmos primários (regra OOF), preditores e coerência Aintl x B")

run.finish(notes=" | ".join(f"{t}={primary[t]['algorithm']}" for t in M.tasks))

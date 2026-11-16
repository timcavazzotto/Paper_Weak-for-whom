"""
03_explain.py — explicação e testes das hipóteses (usa os modelos do 02; nada é reajustado,
exceto os modelos lineares das seções 6–8, que são análises inferenciais à parte)
===========================================================================================
  1. Permutação em BLOCO por domínio, todos os algoritmos (M3) de todas as tarefas (W3;
     IC 95% por bootstrap de conglomerados).
  2. Concordância entre algoritmos (Spearman dos rankings de domínio).
  3. Permutação por variável (primário) nas tarefas Acond, Aintl e B.
  4. SHAP do primário de cada tarefa, na MESMA amostra da W3 (pareada entre tarefas).
     B: submodelos P(≥P25) "sair da base" e P(≥P75) "chegar ao topo".
  5. ASSIMETRIA topo × base: participação de cada domínio (e de variáveis de déficit vs
     capacidade) no |SHAP| do modelo da base e do topo; diferença com IC por bootstrap.
     Pares: condicional (≤P10 vs ≥P90), mundial (<P10 vs ≥P75), ordinal B (sub0 vs sub2).
  6. Decomposição Shapley do R² por domínio (MQO; treino e W3) para Acond e Aintl:
     as fatias somam o R² e o resto é o não explicado.
  7. Regressão quantílica do z condicional (treino), τ = 0,10…0,90; teste β(0,90) − β(0,10)
     por bootstrap de conglomerados.
  8. Sensibilidade dentro da UPA: correlação intraclasse por UPA e R² por domínio com
     efeitos fixos de UPA (tira equipe, aparelho, época e contexto local).

Uso (na raiz):  python code/03_explain.py        ($env:SR_FAST="1" para teste rápido)
Variáveis opcionais: K_PERM (10), B_PERM (200), SHAP_N (4000), B_QR (bootstrap QR, 100), B_REFIT (reajustes da assimetria, 50)

Saídas em outputs/v<VERSION>/<cenário>/03_explain/tables/
  perm_domain.csv | concordance_algorithms.csv | perm_variable.csv
  shap_variable.csv | shap_domain.csv | cache/shap_<modelo>.parquet, cache/X_shap.parquet
  asymmetry_domain.csv | asymmetry_role.csv | asymmetry_direction.csv | asymmetry_permutation.csv
  F4_asymmetry_refit.csv (IC com reajuste dos modelos — o que vai para o artigo)
  shapley_r2.csv | qr_coefficients.csv | qr_asymmetry.csv | within_upa.csv | icc_upa.csv
"""
import json
import math
import statsmodels.api  # noqa: F401  (usado na seção 7; importado aqui para falhar logo se faltar)
import os
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

import sr_core as core

warnings.filterwarnings("ignore")

run = core.Run("03_explain", __file__)
log = run.log
up1 = run.require("01_prepare")
up2 = run.require("02_model")
M = core.MCFG
K = int(os.environ.get("K_PERM", 2 if core.FAST else 10))
B = int(os.environ.get("B_PERM", 30 if core.FAST else 200))
SHAP_N = int(os.environ.get("SHAP_N", 500 if core.FAST else 4000))
B_QR = int(os.environ.get("B_QR", 5 if core.FAST else 100))
log.info(f"K_PERM={K} | B_PERM={B} | SHAP_N={SHAP_N} | B_QR={B_QR}")

data = pd.read_parquet(up1 / "data" / "analytic.parquet")
train = data[data["split"] == "train"].reset_index(drop=True)
test = data[data["split"] == "test"].reset_index(drop=True)
primary = json.loads((up2 / "tables" / "primary_models.json").read_text(encoding="utf-8"))
TASKS = [t for t in M.tasks if t in primary]
domain_of = {f.name: f.domain for f in core.FEATURES}
role_of = {f.name: f.role for f in core.FEATURES}
label_of = {f.name: f.label for f in core.FEATURES}
DOMAINS = list(core.DOMAINS)
rng = np.random.default_rng(core.SEED)
log.info(f"tarefas com modelos no 02: {', '.join(TASKS)}")


def auc_fast(y, p):
    y = np.asarray(y).astype(bool)
    n1, n0 = y.sum(), (~y).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(p)
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def metric_fn(task):
    return auc_fast if core.TASKS[task]["kind"] == "binary" else core.fast_metric(task)


def load(task, algo, layer="M3"):
    return joblib.load(up2 / "models" / f"{task}_{algo}_{layer}.joblib")


def task_xy(task, df=test):
    tgt = core.TASKS[task]["target"]
    d = df[df[tgt].notna()].reset_index(drop=True)
    y = d[tgt].to_numpy(float) if core.TASKS[task]["kind"] == "regression" else d[tgt].astype(int).to_numpy()
    return d, d[core.task_features(task)], y


def perm_drops(task, model, X, y, groups: dict, boots):
    """Para cada grupo de colunas: K permutações conjuntas. IC: as mesmas reamostras de
    conglomerados comparam a previsão original e a permutada."""
    f = metric_fn(task)
    base_pred = core.predict(task, model, X)
    base = f(y, base_pred)
    base_boot = np.array([f(y[i], base_pred[i]) for i in boots])
    out = []
    for g, cols in groups.items():
        point, acc = [], np.zeros(len(boots))
        for _ in range(K):
            Xp = X.copy()
            Xp[cols] = X[cols].to_numpy()[rng.permutation(len(X))]
            pp = core.predict(task, model, Xp)
            point.append(base - f(y, pp))
            acc += base_boot - np.array([f(y[i], pp[i]) for i in boots])
        dist = acc / K
        out.append({"group": g, "n_vars": len(cols), "drop": float(np.mean(point)),
                    "ci_low": float(np.nanquantile(dist, 0.025)), "ci_high": float(np.nanquantile(dist, 0.975))})
    df = pd.DataFrame(out)
    df["baseline"] = base
    df["rank"] = df["drop"].rank(ascending=False, method="min").astype(int)
    return df


# -----------------------------------------------------------------------------
# 1–2. permutação por domínio (todos os algoritmos) + concordância
# -----------------------------------------------------------------------------
t0 = time.time()
dom_rows, conc_rows = [], []
for task in TASKS:
    d, X, y = task_xy(task)
    names = list(X.columns)
    groups = {dm: [n for n in names if domain_of[n] == dm] for dm in DOMAINS}
    groups = {k: v for k, v in groups.items() if v}
    boots = core.cluster_boot_indices(d["stratum"], d["cluster"], B)
    algos = core.TASKS[task]["algorithms"]
    ranks = {}
    for algo in algos:
        df = perm_drops(task, load(task, algo), X, y, groups, boots)
        df.insert(0, "task", task); df.insert(1, "algorithm", algo)
        df["primary"] = algo == primary[task]["algorithm"]
        df["layer"] = df["group"].map(core.DOMAINS)
        dom_rows.append(df)
        ranks[algo] = df.set_index("group")["drop"]
    for i, a in enumerate(algos):
        for b in algos[i + 1:]:
            conc_rows.append({"task": task, "algorithm_1": a, "algorithm_2": b,
                              "spearman_domains": spearmanr(ranks[a], ranks[b]).statistic,
                              "top3_overlap": len(set(ranks[a].nlargest(3).index) & set(ranks[b].nlargest(3).index))})
    p = dom_rows[-len(algos):]
    prim = [x for x in p if x["primary"].iloc[0]][0].sort_values("drop", ascending=False)
    log.info(f"[{task}] permutação por domínio ({primary[task]['algorithm']}): "
             + " > ".join(f"{r.group} {r.drop:.3f}" for r in prim.head(5).itertuples()) + f"  ({(time.time() - t0) / 60:.1f} min)")
perm_domain = pd.concat(dom_rows, ignore_index=True).rename(columns={"group": "domain"})
run.save_table(perm_domain, "perm_domain", "queda da métrica principal ao permutar cada domínio (W3), por tarefa e algoritmo", item="S4")
run.save_table(pd.DataFrame(conc_rows), "concordance_algorithms", "concordância dos rankings de domínio entre algoritmos", item="S4")

# -----------------------------------------------------------------------------
# 3. permutação por variável (primário; tarefas principais)
# -----------------------------------------------------------------------------
var_rows = []
for task in [t for t in ("Acond", "Aintl", "B") if t in TASKS]:
    algo = primary[task]["algorithm"]
    d, X, y = task_xy(task)
    boots = core.cluster_boot_indices(d["stratum"], d["cluster"], B)
    df = perm_drops(task, load(task, algo), X, y, {n: [n] for n in X.columns}, boots)
    df.insert(0, "task", task); df.insert(1, "algorithm", algo)
    df["domain"] = df["group"].map(domain_of)
    var_rows.append(df)
    log.info(f"[{task}] permutação por variável concluída ({(time.time() - t0) / 60:.1f} min)")
if var_rows:
    perm_var = pd.concat(var_rows, ignore_index=True).rename(columns={"group": "feature"})
    perm_var["label"] = perm_var["feature"].map(label_of)
    run.save_table(perm_var, "perm_variable", "queda da métrica principal ao permutar cada variável (W3, primário)", item="S4")

# -----------------------------------------------------------------------------
# 4. SHAP (primário de cada tarefa) na mesma amostra da W3
# -----------------------------------------------------------------------------
import shap  # noqa: E402

pool = test[test["z_cond"].notna() & test["hgs_z"].notna()]
S_ROWS = pool.sample(min(SHAP_N, len(pool)), random_state=core.SEED).reset_index(drop=True)
X_all = S_ROWS[core.FEATURE_NAMES]
X_all.to_parquet(run.dir / "cache" / "X_shap.parquet", index=False)
S_ROWS[["row_id", "stratum", "cluster"]].to_parquet(run.dir / "cache" / "rows_shap.parquet", index=False)
run.save_file(run.dir / "cache" / "X_shap.parquet", "cache", "valores das variáveis na amostra SHAP (W3)")

SHAP_MODELS = {}      # nome -> (tarefa, submodelo)
for task in TASKS:
    if task == "B":
        SHAP_MODELS["B_ge_P25"] = ("B", 0)
        SHAP_MODELS["B_ge_P75"] = ("B", -1)
    else:
        SHAP_MODELS[task] = (task, None)


def shap_from_pipeline(model, algo, names, sub=None):
    """SHAP na amostra fixa da W3, agregado de volta às variáveis originais."""
    Xs = S_ROWS[names]
    Xt = model["pre"].transform(Xs)
    cols = list(model["pre"].get_feature_names_out())
    est = model["clf"] if sub is None else model["clf"].models_[sub]
    if algo in ("ElasticNet", "OrdinalLogit", "Logistic"):
        sv = shap.LinearExplainer(est, Xt).shap_values(Xt)
    else:
        sv = shap.TreeExplainer(est).shap_values(Xt)
    sv = np.asarray(sv[1] if isinstance(sv, list) else sv)
    if sv.ndim == 3:
        sv = sv[:, :, 1]
    S = pd.DataFrame(sv, columns=cols).T.groupby(lambda c: core.original_feature(c, names)).sum().T
    return S[[n for n in names if n in S.columns]]


def shap_matrix(task, sub):
    algo = primary[task]["algorithm"]
    return shap_from_pipeline(load(task, algo), algo, core.task_features(task), sub), algo


SH = {}
shap_rows, dom_share = [], []
for name, (task, sub) in SHAP_MODELS.items():
    S, algo = shap_matrix(task, sub)
    SH[name] = S
    S.to_parquet(run.dir / "cache" / f"shap_{name}.parquet", index=False)
    run.save_file(run.dir / "cache" / f"shap_{name}.parquet", "cache", f"SHAP por variável original, {name}")
    mabs = S.abs().mean()
    t = pd.DataFrame({"model": name, "task": task, "algorithm": algo, "feature": mabs.index, "mean_abs_shap": mabs.values})
    t["share_pct"] = 100 * t["mean_abs_shap"] / t["mean_abs_shap"].sum()
    t["domain"] = t["feature"].map(domain_of)
    t["role"] = t["feature"].map(role_of)
    t["label"] = t["feature"].map(label_of)
    t["direction_spearman"] = [spearmanr(S_ROWS[c].astype(float), S[c], nan_policy="omit").statistic
                               if core.feature_lists([c])[0] else np.nan for c in t["feature"]]
    shap_rows.append(t.sort_values("mean_abs_shap", ascending=False))
    dm = S.T.groupby(domain_of).sum().T.abs().mean()          # contribuição líquida do domínio por pessoa
    dom_share.append(pd.DataFrame({"model": name, "task": task, "domain": dm.index, "share_pct": 100 * dm.values / dm.sum()})
                     .sort_values("share_pct", ascending=False))
    log.info(f"SHAP {name} ({algo}): top 5 = " + ", ".join(t.sort_values('mean_abs_shap', ascending=False)['feature'].head(5)))
run.save_table(pd.concat(shap_rows, ignore_index=True), "shap_variable", "média do |SHAP| por variável (primário, W3)", item="S4")
run.save_table(pd.concat(dom_share, ignore_index=True), "shap_domain", "participação (%) de cada domínio no |SHAP| (contribuição líquida do domínio)", item="S4")

# -----------------------------------------------------------------------------
# 5. assimetria topo × base
# -----------------------------------------------------------------------------
PAIRS = {k: v for k, v in core.ASYMMETRY_PAIRS.items() if all(t in SH for t in v)}
if {"B_ge_P25", "B_ge_P75"} <= set(SH):
    PAIRS["ordinal_B"] = ("B_ge_P25", "B_ge_P75")
boots_rows = core.cluster_boot_indices(S_ROWS["stratum"], S_ROWS["cluster"], B)


def shares(S, by: dict, idx=None):
    A = S if idx is None else S.iloc[idx]
    g = A.T.groupby(by).sum().T.abs().mean()
    return g / g.sum()


asym_rows, role_rows, dir_rows, perm_asym = [], [], [], []
for pair, (lo, hi) in PAIRS.items():
    for level, by_map, out in (("domain", domain_of, asym_rows), ("role", role_of, role_rows)):
        s_lo, s_hi = shares(SH[lo], by_map), shares(SH[hi], by_map)
        keys = sorted(set(s_lo.index) | set(s_hi.index))
        bl, bh = [], []
        for i in boots_rows:
            bl.append(shares(SH[lo], by_map, i).reindex(keys)); bh.append(shares(SH[hi], by_map, i).reindex(keys))
        bl, bh = pd.DataFrame(bl), pd.DataFrame(bh)
        diff = bh - bl
        for k in keys:
            out.append({"pair": pair, "model_low": lo, "model_high": hi, level: k,
                        "share_low_pct": 100 * s_lo.get(k, np.nan), "share_high_pct": 100 * s_hi.get(k, np.nan),
                        "low_ci": f"{100 * bl[k].quantile(.025):.1f}–{100 * bl[k].quantile(.975):.1f}",
                        "high_ci": f"{100 * bh[k].quantile(.025):.1f}–{100 * bh[k].quantile(.975):.1f}",
                        "diff_high_minus_low_pp": 100 * (s_hi.get(k, np.nan) - s_lo.get(k, np.nan)),
                        "diff_ci_low": 100 * diff[k].quantile(.025), "diff_ci_high": 100 * diff[k].quantile(.975)})
        if level == "domain":
            rho = spearmanr(s_lo.reindex(keys).fillna(0), s_hi.reindex(keys).fillna(0)).statistic
            log.info(f"assimetria [{pair}] Spearman dos domínios base×topo = {rho:.2f}")
            for r in out[-len(keys):]:
                r["spearman_low_high"] = rho
    # direção de cada variável: SHAP do modelo da base é risco de ESTAR na base (sinal invertido para comparar)
    common = [c for c in SH[lo].columns if c in SH[hi].columns]
    for c in common:
        if not core.feature_lists([c])[0]:
            continue
        x = S_ROWS[c].astype(float)
        dir_rows.append({"pair": pair, "feature": c, "label": label_of[c], "domain": domain_of[c], "role": role_of[c],
                         "mean_abs_low": SH[lo][c].abs().mean(), "mean_abs_high": SH[hi][c].abs().mean(),
                         "direction_low_model": spearmanr(x, SH[lo][c], nan_policy="omit").statistic,
                         "direction_high_model": spearmanr(x, SH[hi][c], nan_policy="omit").statistic})
    # versão por permutação (sem IC; checagem de robustez)
    if pair in core.ASYMMETRY_PAIRS:
        for side, t in (("low", lo), ("high", hi)):
            p = perm_domain[(perm_domain["task"] == t) & perm_domain["primary"]].copy()
            p["share_pct"] = 100 * p["drop"].clip(lower=0) / p["drop"].clip(lower=0).sum()
            p["pair"], p["side"] = pair, side
            perm_asym.append(p[["pair", "side", "task", "algorithm", "domain", "drop", "ci_low", "ci_high", "share_pct"]])

run.save_table(pd.DataFrame(asym_rows), "asymmetry_domain", "participação de cada domínio na base e no topo (IC: só pessoas da W3, modelo fixo)")
run.save_table(pd.DataFrame(role_rows), "asymmetry_role", "participação de variáveis de déficit, capacidade e contexto na base e no topo")
run.save_table(pd.DataFrame(dir_rows), "asymmetry_direction", "magnitude e direção do SHAP por variável na base e no topo")
if perm_asym:
    run.save_table(pd.concat(perm_asym, ignore_index=True), "asymmetry_permutation", "participação por domínio via permutação (checagem)")
for r in [x for x in asym_rows if abs(x["diff_high_minus_low_pp"]) >= 3]:
    log.info(f"  [{r['pair']}] {r['domain']}: base {r['share_low_pct']:.1f}% | topo {r['share_high_pct']:.1f}% "
             f"| dif {r['diff_high_minus_low_pp']:+.1f} pp ({r['diff_ci_low']:.1f} a {r['diff_ci_high']:.1f})")

# -----------------------------------------------------------------------------
# 5b. IC da assimetria com a incerteza do AJUSTE (bootstrap com reajuste)
#     Reamostra UPAs do TREINO, reajusta os modelos da base e do topo com os mesmos
#     hiperparâmetros do 02 e recalcula o SHAP na mesma amostra da W3. É este IC que
#     vai para o artigo (o da seção 5 só reamostra as pessoas da W3, com o modelo fixo).
# -----------------------------------------------------------------------------
from joblib import Parallel, delayed  # noqa: E402

B_REFIT = int(os.environ.get("B_REFIT", 3 if core.FAST else 50))
# Reajustes em fios paralelos (LightGBM/XGBoost + SHAP ao mesmo tempo) derrubaram o processo no Windows
# (código 3221225477 = 0xC0000005, acesso inválido à memória). No Windows, o padrão passa a ser SEQUENCIAL, com cada
# modelo usando todos os núcleos; SR_REFIT_PARALLEL=1 volta ao modo antigo (fios), SR_REFIT_PARALLEL=0 força sequencial.
REFIT_PARALLEL = os.environ.get("SR_REFIT_PARALLEL", "0" if os.name == "nt" else "1") == "1"
best_params = json.loads((up2 / "tables" / "best_params.json").read_text(encoding="utf-8"))


def refit_one(task, d, idx, seed):
    algo = primary[task]["algorithm"]
    names = core.task_features(task)
    pipe, _ = core.make_pipeline(task, algo, names)
    pipe.set_params(**best_params[f"{task}_{algo}_M3"]["final"])
    if "random_state" in pipe["clf"].get_params():
        pipe.set_params(clf__random_state=seed)
    if not REFIT_PARALLEL and "n_jobs" in pipe["clf"].get_params():
        pipe.set_params(clf__n_jobs=-1)          # sequencial: o próprio modelo usa os núcleos
    tgt = core.TASKS[task]["target"]
    pipe.fit(d.iloc[idx][names], d.iloc[idx][tgt].astype(int).to_numpy())
    S = shap_from_pipeline(pipe, algo, names)
    return shares(S, domain_of), shares(S, role_of)


refit_rows = []
for pair, (lo, hi) in core.ASYMMETRY_PAIRS.items():
    if not (lo in TASKS and hi in TASKS):
        continue
    tgt_lo, tgt_hi = core.TASKS[lo]["target"], core.TASKS[hi]["target"]
    d = train[train[tgt_lo].notna() & train[tgt_hi].notna()].reset_index(drop=True)
    idxs = core.cluster_boot_indices(d["stratum"], d["cluster"], B_REFIT, seed=core.SEED + 11)
    t1 = time.time()
    if REFIT_PARALLEL:
        res = Parallel(n_jobs=-1, prefer="threads")(delayed(lambda b, i: (refit_one(lo, d, i, core.SEED + b), refit_one(hi, d, i, core.SEED + b)))(b, i)
                                  for b, i in enumerate(idxs))
    else:
        res = []
        for b, i in enumerate(idxs):
            res.append((refit_one(lo, d, i, core.SEED + b), refit_one(hi, d, i, core.SEED + b)))
            if (b + 1) % 10 == 0:
                log.info(f"  [{pair}] reajuste {b + 1}/{len(idxs)} ({(time.time() - t1) / 60:.1f} min)")
    for level, k_idx, by_map in (("domain", 0, domain_of), ("role", 1, role_of)):
        L = pd.DataFrame([r[0][k_idx] for r in res]).fillna(0)
        H = pd.DataFrame([r[1][k_idx] for r in res]).fillna(0)
        p_lo, p_hi = shares(SH[lo], by_map), shares(SH[hi], by_map)
        for k in sorted(set(L.columns) | set(H.columns)):
            lo_b, hi_b = L.get(k, pd.Series(0, index=L.index)), H.get(k, pd.Series(0, index=H.index))
            dif = hi_b - lo_b
            refit_rows.append({"pair": pair, "level": level, "key": k,
                               "share_low_pct": 100 * p_lo.get(k, 0), "low_ci_low": 100 * lo_b.quantile(.025), "low_ci_high": 100 * lo_b.quantile(.975),
                               "share_high_pct": 100 * p_hi.get(k, 0), "high_ci_low": 100 * hi_b.quantile(.025), "high_ci_high": 100 * hi_b.quantile(.975),
                               "diff_high_minus_low_pp": 100 * (p_hi.get(k, 0) - p_lo.get(k, 0)),
                               "diff_ci_low": 100 * dif.quantile(.025), "diff_ci_high": 100 * dif.quantile(.975),
                               "p_boot_two_sided": float(min(1.0, 2 * min((dif <= 0).mean(), (dif >= 0).mean()))),
                               "n_refit": len(res)})
    log.info(f"assimetria [{pair}] com reajuste: {len(res)} reamostras ({(time.time() - t1) / 60:.1f} min)")
if refit_rows:
    rf = pd.DataFrame(refit_rows)
    run.save_table(rf, "asymmetry_refit", "assimetria base × topo com IC por bootstrap com reajuste dos modelos (treino)", item="F4")
    for r in rf[(rf["level"] == "domain") & ((rf["diff_ci_low"] > 0) | (rf["diff_ci_high"] < 0))].itertuples():
        log.info(f"  [{r.pair}, reajuste] {r.key}: dif {r.diff_high_minus_low_pp:+.1f} pp ({r.diff_ci_low:.1f} a {r.diff_ci_high:.1f})")


# -----------------------------------------------------------------------------
# desenho linear comum às seções 6–8
# -----------------------------------------------------------------------------
def linear_design(names, tr, te=None):
    """Contínuas/ordinais: imputação pela mediana do treino e padronização (efeito por DP).
    Binárias: 0/1 (moda). Categóricas: indicadoras, referência = categoria mais frequente.
    Indicador de ausência para variáveis com > 1% ausente no treino."""
    num, cat = core.feature_lists(names)
    kinds = {f.name: f.kind for f in core.FEATURES}

    def build(df, fit):
        cols, owner = {}, {}
        for n in num:
            s = df[n].astype(float)
            if fit:
                st[n] = {"med": tr[n].median(), "miss": tr[n].isna().mean() > 0.01,
                         "mu": tr[n].fillna(tr[n].median()).mean(), "sd": tr[n].fillna(tr[n].median()).std() or 1.0}
            p = st[n]
            v = s.fillna(p["med"])
            cols[n] = v if kinds[n] == "bin" else (v - p["mu"]) / p["sd"]
            owner[n] = n
            if p["miss"]:
                cols[f"{n}__missing"] = s.isna().astype(float)
                owner[f"{n}__missing"] = n
        for n in cat:
            s = df[n].fillna("Missing").astype(str)
            if fit:
                st[n] = {"levels": [lv for lv in tr[n].fillna("Missing").astype(str).value_counts().index[1:]]}
            for lv in st[n]["levels"]:
                cols[f"{n}={lv}"] = (s == lv).astype(float)
                owner[f"{n}={lv}"] = n
        return pd.DataFrame(cols, index=df.index), owner

    st = {}
    Xtr, owner = build(tr, True)
    Xte = build(te, False)[0] if te is not None else None
    keep = Xtr.columns[Xtr.std() > 0]
    # indicadores de ausência idênticos (ex.: cintura, RFM, massa magra) ficam só uma vez
    keep = [c for c, dup in zip(keep, Xtr[keep].T.duplicated()) if not dup]
    return Xtr[keep], (Xte[keep] if Xte is not None else None), {c: owner[c] for c in keep}


# -----------------------------------------------------------------------------
# 6. decomposição Shapley do R² por domínio (MQO, todas as 2^D combinações)
# -----------------------------------------------------------------------------
def shapley_r2(task):
    tgt = core.TASKS[task]["target"]
    names = core.task_features(task)
    tr, te = train[train[tgt].notna()], test[test[tgt].notna()]
    Xtr, Xte, owner = linear_design(names, tr, te)
    ytr, yte = tr[tgt].to_numpy(float), te[tgt].to_numpy(float)
    doms = [dm for dm in DOMAINS if any(domain_of[owner[c]] == dm for c in Xtr.columns)]
    col_dom = np.array([doms.index(domain_of[owner[c]]) for c in Xtr.columns])
    A = np.column_stack([np.ones(len(Xtr)), Xtr.to_numpy()])
    At = np.column_stack([np.ones(len(Xte)), Xte.to_numpy()])
    G, g = A.T @ A, A.T @ ytr
    sst_tr, sst_te = ((ytr - ytr.mean()) ** 2).sum(), ((yte - yte.mean()) ** 2).sum()
    D = len(doms)
    v_tr, v_te = np.zeros(2 ** D), np.zeros(2 ** D)
    for mask in range(2 ** D):
        sel = [0] + [j + 1 for j in range(len(col_dom)) if (mask >> col_dom[j]) & 1]
        b = np.linalg.lstsq(G[np.ix_(sel, sel)], g[sel], rcond=None)[0]
        v_tr[mask] = 1 - ((ytr - A[:, sel] @ b) ** 2).sum() / sst_tr
        v_te[mask] = 1 - ((yte - At[:, sel] @ b) ** 2).sum() / sst_te
    phi_tr, phi_te = np.zeros(D), np.zeros(D)
    for j in range(D):
        for mask in range(2 ** D):
            if (mask >> j) & 1:
                continue
            s = bin(mask).count("1")
            wgt = math.factorial(s) * math.factorial(D - s - 1) / math.factorial(D)
            phi_tr[j] += wgt * (v_tr[mask | (1 << j)] - v_tr[mask])
            phi_te[j] += wgt * (v_te[mask | (1 << j)] - v_te[mask])
    full = 2 ** D - 1
    out = pd.DataFrame({"task": task, "domain": doms, "layer": [core.DOMAINS[d] for d in doms],
                        "shapley_r2_train": phi_tr, "shapley_r2_test": phi_te})
    out["share_of_explained_train_pct"] = 100 * out["shapley_r2_train"] / v_tr[full]
    out["share_of_explained_test_pct"] = 100 * out["shapley_r2_test"] / (v_te[full] - v_te[0])
    tot = pd.DataFrame([{"task": task, "domain": "TOTAL explained (R²)", "shapley_r2_train": v_tr[full],
                         "shapley_r2_test": v_te[full]},
                        {"task": task, "domain": "Unexplained (1 − R²)", "shapley_r2_train": 1 - v_tr[full],
                         "shapley_r2_test": 1 - v_te[full]},
                        {"task": task, "domain": "intercept-only R² in W3 (period shift)", "shapley_r2_train": 0.0,
                         "shapley_r2_test": v_te[0]}])
    log.info(f"[{task}] Shapley R² (MQO): treino {v_tr[full]:.3f} | W3 {v_te[full]:.3f}; maiores: "
             + ", ".join(f"{r.domain} {r.shapley_r2_test:.3f}" for r in out.sort_values('shapley_r2_test', ascending=False).head(4).itertuples()))
    return pd.concat([out, tot], ignore_index=True)


sh_tabs = [shapley_r2(t) for t in ("Acond", "Aintl") if t in TASKS]
if sh_tabs:
    run.save_table(pd.concat(sh_tabs, ignore_index=True), "shapley_r2", "decomposição Shapley do R² por domínio (MQO; treino e W3)", item="F3")

# -----------------------------------------------------------------------------
# 7. regressão quantílica do z condicional (treino)
# -----------------------------------------------------------------------------
QR_DROP = {"bmi", "waist_cm", "whtr", "weight_kg", "multimorbidity"}   # colineares com rfm/ffm e com as doenças
TAUS = (0.10, 0.25, 0.50, 0.75, 0.90)
if "Acond" in TASKS:
    import statsmodels.api as sm

    qr_names = [n for n in core.task_features("Acond") if n not in QR_DROP]
    tr = train[train["z_cond"].notna()].reset_index(drop=True)
    Xq, _, owner = linear_design(qr_names, tr)
    Xq = sm.add_constant(Xq)
    yq = tr["z_cond"].to_numpy(float)
    coef_rows = []
    ols = sm.OLS(yq, Xq).fit(cov_type="cluster", cov_kwds={"groups": tr["cluster"].astype("category").cat.codes})
    for c in Xq.columns[1:]:
        coef_rows.append({"term": c, "feature": owner[c], "tau": "OLS (mean)", "coef": ols.params[c], "se": ols.bse[c]})
    fits = {}
    for tau in TAUS:
        r = sm.QuantReg(yq, Xq).fit(q=tau, max_iter=5000)
        fits[tau] = r.params
        for c in Xq.columns[1:]:
            coef_rows.append({"term": c, "feature": owner[c], "tau": tau, "coef": r.params[c], "se": r.bse[c]})
        log.info(f"QR τ={tau:.2f} ajustada ({(time.time() - t0) / 60:.1f} min)")
    qr = pd.DataFrame(coef_rows)
    qr["ci_low"], qr["ci_high"] = qr["coef"] - 1.96 * qr["se"], qr["coef"] + 1.96 * qr["se"]
    qr["domain"], qr["role"], qr["label"] = qr["feature"].map(domain_of), qr["feature"].map(role_of), qr["feature"].map(label_of)
    run.save_table(qr, "qr_coefficients", "regressão quantílica do z condicional (por DP; binárias 0/1), treino", item="F5")

    diffs = []
    for i in core.cluster_boot_indices(tr["stratum"], tr["cluster"], B_QR, seed=core.SEED + 7):
        try:
            hi = sm.QuantReg(yq[i], Xq.iloc[i]).fit(q=0.90, max_iter=5000).params
            lo = sm.QuantReg(yq[i], Xq.iloc[i]).fit(q=0.10, max_iter=5000).params
            diffs.append(hi - lo)
        except Exception:
            continue
    diffs = pd.DataFrame(diffs)
    point = fits[0.90] - fits[0.10]
    qa = pd.DataFrame({"term": Xq.columns[1:], "feature": [owner[c] for c in Xq.columns[1:]],
                       "beta_p10": fits[0.10][1:].values, "beta_p90": fits[0.90][1:].values, "diff_p90_minus_p10": point[1:].values,
                       "diff_ci_low": diffs[Xq.columns[1:]].quantile(0.025).values,
                       "diff_ci_high": diffs[Xq.columns[1:]].quantile(0.975).values,
                       "p_boot_two_sided": [2 * min((diffs[c] <= 0).mean(), (diffs[c] >= 0).mean()) for c in Xq.columns[1:]],
                       "n_boot_ok": len(diffs)})
    qa["domain"], qa["role"], qa["label"] = qa["feature"].map(domain_of), qa["feature"].map(role_of), qa["feature"].map(label_of)
    run.save_table(qa, "qr_asymmetry", "β(τ=0,90) − β(τ=0,10) com IC por bootstrap de conglomerados (treino)")
    sig = qa[(qa["diff_ci_low"] > 0) | (qa["diff_ci_high"] < 0)]
    log.info(f"QR: {len(sig)} de {len(qa)} termos com β(0,90) ≠ β(0,10) (IC 95% exclui 0): " + ", ".join(sig["term"].head(10)))

# -----------------------------------------------------------------------------
# 8. sensibilidade dentro da UPA
# -----------------------------------------------------------------------------
def icc_oneway(y, g):
    df = pd.DataFrame({"y": y, "g": g}).dropna()
    k = df.groupby("g")["y"].agg(["mean", "count"])
    n, a = len(df), len(k)
    n0 = (n - (k["count"] ** 2).sum() / n) / (a - 1)
    msb = (k["count"] * (k["mean"] - df["y"].mean()) ** 2).sum() / (a - 1)
    msw = ((df["y"] - df["g"].map(k["mean"])) ** 2).sum() / (n - a)
    s2b = max((msb - msw) / n0, 0)
    return s2b / (s2b + msw)


icc = [{"wave": f"W{w}", "outcome": o, "icc_upa": icc_oneway(g[o], g["cluster"]), "n_upa": g["cluster"].nunique()}
       for w, g in data.groupby("wave") for o in ("z_cond", "hgs_z")]
run.save_table(pd.DataFrame(icc), "icc_upa", "correlação intraclasse por UPA (equipe, aparelho, época e contexto local)", item="S5")
log.info("ICC por UPA: " + "; ".join(f"{r['wave']} {r['outcome']} {r['icc_upa']:.3f}" for r in icc))

within = []
for task in [t for t in ("Acond", "Aintl") if t in TASKS]:
    tgt = core.TASKS[task]["target"]
    tr = train[train[tgt].notna()].reset_index(drop=True)
    X, _, owner = linear_design(core.task_features(task), tr)
    y = tr[tgt].to_numpy(float)
    for fe in (False, True):
        Xm, ym = X.copy(), y.copy()
        if fe:                                               # efeitos fixos de UPA: centra dentro da UPA
            Xm = Xm - Xm.groupby(tr["cluster"]).transform("mean")
            ym = ym - pd.Series(ym).groupby(tr["cluster"]).transform("mean").to_numpy()
            Xm = Xm.loc[:, Xm.std() > 1e-9]
        A = np.column_stack([np.ones(len(Xm)), Xm.to_numpy()])
        sst = ((ym - ym.mean()) ** 2).sum()
        r2_full = 1 - ((ym - A @ np.linalg.lstsq(A, ym, rcond=None)[0]) ** 2).sum() / sst
        for dm in DOMAINS:
            keep = [0] + [j + 1 for j, c in enumerate(Xm.columns) if domain_of[owner[c]] != dm]
            if len(keep) == A.shape[1]:
                continue
            r2_wo = 1 - ((ym - A[:, keep] @ np.linalg.lstsq(A[:, keep], ym, rcond=None)[0]) ** 2).sum() / sst
            within.append({"task": task, "upa_fixed_effects": fe, "domain": dm, "r2_full": r2_full,
                           "delta_r2_drop_domain": r2_full - r2_wo})
if within:
    run.save_table(pd.DataFrame(within), "within_upa", "ΔR² ao retirar cada domínio, com e sem efeitos fixos de UPA (treino)", item="S5")

run.finish(notes=f"K_PERM={K}, B_PERM={B}, SHAP_N={SHAP_N}, B_QR={B_QR}, B_REFIT={B_REFIT}; pares de assimetria: {', '.join(PAIRS)}")

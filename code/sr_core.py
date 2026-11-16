"""
sr_core.py — núcleo do projeto Strength Reserve (ELSI-Brasil, ondas 1–3)
=========================================================================
TODA regra de codificação, todo caminho e toda convenção de arquivo vive aqui.
Os scripts de etapa (01_prepare, 02_model, 03_explain, 04_report, 05_external)
fazem `import sr_core as core` e NUNCA reescrevem uma regra.

PERGUNTAS DO ARTIGO (v3)
------------------------
1. Onde os idosos brasileiros estão em relação às normas mundiais (Tomkinson 2025)?
   -> z/centil internacional, com o desenho amostral (04).
2. O que explica a força PARA O PRÓPRIO CORPO?
   -> centil condicional: força em relação ao esperado para idade, sexo e altura,
      por um modelo de localização-escala ajustado no treino (referência ELSI).
3. Topo e base são espelho?
   -> caudas condicionais (≤P10 / ≥P90 condicional) e mundiais (<P10 / ≥P75
      internacional), modelos separados e comparação dos determinantes.

VERSIONAMENTO
-------------
VERSION = "MAJOR.MINOR.PATCH" (altere aqui e registre em CHANGELOG.md)
  MAJOR  muda definição de desfecho, exclusões ou harmonização  -> refaz tudo
  MINOR  muda modelagem ou explicabilidade                        -> refaz 02+
  PATCH  muda só tabelas/figuras                                  -> refaz 04
Cada etapa grava em  outputs/v<VERSION>/<CENÁRIO>/<ETAPA>/  e escreve um
manifest.json com: versão, cenário, data, assinatura da etapa (hash do código
e da configuração que a afetam), hash dos arquivos de entrada e de cada saída.
Uma etapa só lê a etapa anterior se a assinatura bater. Se a etapa anterior
não existir na versão atual, ela é reaproveitada de uma versão anterior com a
MESMA assinatura. Todas as execuções ficam registradas em outputs/RUNS.csv.

VARIÁVEIS DE AMBIENTE
---------------------
SR_SCENARIO       cenário (padrão "main"; ver SCENARIOS)
SR_FAST=1         teste rápido (grades mínimas; grava em <etapa>_fast)
SR_TASKS          subconjunto de tarefas do 02/03, separadas por vírgula (ex.: "Acond,B")
SR_OVERWRITE=1    permite sobrescrever uma etapa já concluída nesta versão/cenário
SR_ALLOW_STALE=1  permite usar etapa anterior com assinatura diferente (só para teste)
SR_SKIP_IF_DONE=1 a etapa termina sem fazer nada se já existe execução compatível (usado pelo run_scenarios.py)

FORMATOS
--------
dados intermediários .parquet | tabelas .csv (UTF-8 com BOM, abre no Excel) e
.docx no 04 | figuras .pdf (vetorial) + .png (300 dpi) | modelos .joblib |
metadados .json | log .log.  Arquivos que vão para o artigo levam o código do
item (T1_, F2_, S3_ …) definido em PAPER_ITEMS.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import re
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass, asdict, replace
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# =============================================================================
# >>> SECTION: config
# =============================================================================
PROJECT = "strength_reserve"
VERSION = "5.1.1"
PRESPECIFIED_ON = "2026-11-04"      # data em que domínios, desfechos e hipóteses foram fixados

CODE_DIR = Path(__file__).resolve().parent
ROOT = CODE_DIR.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_REF = ROOT / "data" / "reference"
OUTPUTS = ROOT / "outputs"

RAW_FILES = {1: DATA_RAW / "elsi_w1.dta", 2: DATA_RAW / "elsi_w2.dta", 3: DATA_RAW / "elsi_w3.dta"}
WAVE_YEARS = {1: "2015–16", 2: "2019–21", 3: "2023–24"}
NORMS_FILE = DATA_REF / "tomkinson2025_hgs_norms.csv"
SEED = 42


@dataclass(frozen=True)
class Config:
    """Definições de amostra, desfecho e preditores (afetam o 01 e tudo depois)."""
    # desfecho internacional
    norm: str = "absolute"                 # "absolute" (kg) | "normalized" (kg/m2)
    quartile_reference: str = "international"   # "international" | "internal" (quartis do z no treino)
    noncompleters: str = "exclude"         # "exclude" | "as_lowest" (incapaz/arriscado entram na base)
    protocol_adjustment: str = "elsi_hydraulic"   # chave de PROTOCOL_FACTORS (ajuste ao protocolo de referência das normas)
    # desfecho condicional (referência ELSI ajustada no treino)
    reference_vars: tuple = ("age", "male", "height_cm")   # o "dado": sai da explicação
    ref_age_knots: int = 4                 # nós da spline de idade (por sexo)
    # caudas (pré-especificadas)
    cond_low: float = 10.0                 # centil condicional ≤ 10
    cond_high: float = 90.0                # centil condicional ≥ 90
    intl_low: float = 10.0                 # centil internacional < 10
    intl_high: float = 75.0                # centil internacional ≥ 75
    # amostra
    age_min: int = 50
    exclude_proxy: bool = True
    exclude_dementia: bool = True
    train_waves: tuple = (1, 2)
    test_waves: tuple = (3,)
    # preditores
    mobility_items: tuple = ("p5", "p6", "p7", "p8", "p9", "p10", "p12", "p13", "p16")  # sem p14/p15 (proxies de força)
    gait_time_range: tuple = (1.0, 60.0)   # s, para 3 m
    walk_distance_m: float = 3.0
    ipaq_cap_min_day: int = 180


# Ajuste do protocolo do ELSI ao protocolo de referência das normas (Tomkinson et al. 2025, Tabela Suplementar 3E):
# referência = hidráulico, sentado, cotovelo fletido, antebraço neutro, empunhadura ajustada à mão, AS DUAS mãos,
# 3 tentativas por mão, valor máximo. O ELSI usa dinamômetro hidráulico SAEHAN (Moreira et al., BMC Geriatr 2022),
# UMA mão (dominante), 3 tentativas, máximo -> só o fator "mão + tentativas + estatística" (1,021 absoluto; 1,020
# normalizado). Tomkinson et al. codificaram o aparelho do ELSI como mecânico e aplicaram 1,072 / 1,067 (sensibilidade).
PROTOCOL_FACTORS = {
    "elsi_hydraulic": {"absolute": 1.021, "normalized": 1.020},   # principal (v5.0.0)
    "tomkinson_coded": {"absolute": 1.072, "normalized": 1.067},  # como o ELSI entrou nas normas
    "none": {"absolute": 1.0, "normalized": 1.0},                 # valor bruto (v1–v4)
}

QUARTILE_LABELS = ["Q1 (<P25)", "Q2 (P25–49)", "Q3 (P50–74)", "Q4 (≥P75)"]

# cenários: chaves de Config (afetam o 01) ou de ModelConfig (afetam só o 02+)
SCENARIOS = {
    "main": {},
    "sens_ref_weight": {"reference_vars": ("age", "male", "height_cm", "weight_kg")},
    "sens_internal_quartiles": {"quartile_reference": "internal"},
    "sens_normalized": {"norm": "normalized"},
    "sens_noncompleters": {"noncompleters": "as_lowest"},
    "sens_mobility_all": {"mobility_items": ("p5", "p6", "p7", "p8", "p9", "p10", "p12", "p13", "p14", "p15", "p16")},
    "sens_with_proxy": {"exclude_proxy": False},
    "sens_weighted": {"weighted_training": True},
    "sens_train_w1": {"train_waves": (1,)},   # treino só na W1: mesmo formato das AVD/AIVD da W3 (W2 binário)
    "sens_protocol_tomkinson": {"protocol_adjustment": "tomkinson_coded"},   # fator que Tomkinson aplicou ao ELSI
    "sens_protocol_raw": {"protocol_adjustment": "none"},                     # sem ajuste de protocolo
}
SCENARIO = os.environ.get("SR_SCENARIO", "main")
if SCENARIO not in SCENARIOS:
    raise SystemExit(f"SR_SCENARIO='{SCENARIO}' não existe. Opções: {list(SCENARIOS)}")
_CFG_KEYS = set(Config.__dataclass_fields__)
CFG = replace(Config(), **{k: v for k, v in SCENARIOS[SCENARIO].items() if k in _CFG_KEYS})
FAST = os.environ.get("SR_FAST") == "1"     # teste rápido: grades pequenas, pasta <etapa>_fast

# <<< SECTION: config


# =============================================================================
# >>> SECTION: run_control
# =============================================================================
STAGES = ["01_prepare", "02_model", "03_explain", "04_report", "05_external", "06_sensitivity", "07_figures", "08_cutoffs", "09_subgroups"]
STAGE_SUBDIRS = {
    "01_prepare": ["data", "tables", "logs"],
    "02_model": ["models", "predictions", "tables", "logs"],
    "03_explain": ["tables", "cache", "logs"],
    "04_report": ["tables", "figures", "docx", "logs"],
    "05_external": ["tables", "figures", "logs"],
    "06_sensitivity": ["tables", "figures", "docx", "logs"],
    "07_figures": ["figures", "tables", "docx", "logs"],
    "08_cutoffs": ["tables", "logs"],
    "09_subgroups": ["tables", "logs"],
}
# seções do sr_core que afetam cada etapa (entram na assinatura)
STAGE_CORE_SECTIONS = {
    "01_prepare": ["config", "read", "harmonization", "features", "outcome", "sample"],
    "02_model": ["config", "features", "modelling"],
    "03_explain": ["config", "features", "modelling"],
    "04_report": ["config", "features", "modelling", "survey", "report"],
    "05_external": ["config", "features", "modelling"],
    "06_sensitivity": ["config", "report"],
    "07_figures": ["config", "features", "modelling", "survey", "report"],
    "08_cutoffs": ["config", "read", "modelling", "survey"],
    "09_subgroups": ["config", "features", "modelling"],
}
STAGE_UPSTREAM = {"01_prepare": [], "02_model": ["01_prepare"], "03_explain": ["01_prepare", "02_model"],
                  "04_report": ["01_prepare", "02_model", "03_explain"], "05_external": ["02_model"], "06_sensitivity": [], "07_figures": [], "08_cutoffs": ["01_prepare"], "09_subgroups": ["01_prepare", "02_model"]}


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def core_section(name: str) -> str:
    src = (CODE_DIR / "sr_core.py").read_text(encoding="utf-8")
    m = re.search(rf"# >>> SECTION: {name}\n(.*?)# <<< SECTION: {name}", src, flags=re.S)
    if not m:
        raise KeyError(f"seção '{name}' não encontrada em sr_core.py")
    # a versão e os títulos do artigo não mudam resultados: ficam fora da assinatura
    return re.sub(r"^(VERSION|PRESPECIFIED_ON) = .*$", "", m.group(1), flags=re.M)


def version_dir(version: str = VERSION, scenario: str = SCENARIO) -> Path:
    return OUTPUTS / f"v{version}" / scenario


def stage_dir(stage: str, version: str = VERSION, scenario: str = SCENARIO, fast: bool = None) -> Path:
    fast = FAST if fast is None else fast
    return version_dir(version, scenario) / (f"{stage}_fast" if fast and stage != "01_prepare" else stage)


def stage_config(stage: str) -> dict:
    """Configuração que entra na assinatura da etapa."""
    out = asdict(CFG)
    if stage != "01_prepare":
        out.update(asdict(MCFG))
    return out


def _pkg_versions():
    out = {"python": platform.python_version()}
    for p in ["numpy", "pandas", "scipy", "sklearn", "xgboost", "lightgbm", "shap", "pyreadstat", "pyarrow", "matplotlib", "docx"]:
        try:
            mod = __import__(p)
            out[p] = getattr(mod, "__version__", "?")
        except Exception:
            pass
    return out


def _version_key(v: str):
    return tuple(int(x) for x in v.lstrip("v").split("."))


def find_compatible(stage: str, signature, fallback_fast: bool):
    """Procura uma execução concluída da etapa com a assinatura dada, na versão atual e
    depois nas anteriores (mais nova primeiro). signature=None aceita qualquer uma."""
    versions = [VERSION] + sorted([p.name[1:] for p in OUTPUTS.glob("v*") if p.is_dir() and p.name[1:] != VERSION],
                                  key=_version_key, reverse=True)
    for v in versions:
        cands = [stage_dir(stage, v)]
        if FAST and fallback_fast:
            cands.append(stage_dir(stage, v, fast=False))
        for folder in cands:
            m = folder / "manifest.json"
            if not m.exists():
                continue
            man = json.loads(m.read_text(encoding="utf-8"))
            if man.get("status") == "complete" and (signature is None or man.get("signature") == signature):
                return v, folder, man
    return None


class Run:
    """Controla uma execução de etapa: pastas, log, trava de sobrescrita,
    leitura de etapas anteriores, registro de saídas e manifesto."""

    def __init__(self, stage: str, script_file: str):
        if stage not in STAGES:
            raise ValueError(stage)
        self.stage, self.script = stage, Path(script_file).resolve()
        self.dir = stage_dir(stage)
        self.t0 = time.time()
        self.outputs, self.inputs, self.upstream = [], {}, {}
        self.signature = self._signature()
        if os.environ.get("SR_SKIP_IF_DONE") == "1":
            done = find_compatible(stage, self.signature, fallback_fast=False)
            if done:
                print(f"[{stage} | {SCENARIO}] já concluída e compatível em v{done[0]} — nada a fazer.")
                raise SystemExit(0)
        manifest = self.dir / "manifest.json"
        if manifest.exists() and os.environ.get("SR_OVERWRITE") != "1":
            old = json.loads(manifest.read_text())
            raise SystemExit(
                f"PARADO: {self.dir} já tem uma execução concluída em {old.get('finished')}.\n"
                f"  Para refazer de propósito: SR_OVERWRITE=1. Para preservar: suba VERSION em sr_core.py "
                f"(hoje {VERSION}) e registre no CHANGELOG.md.")
        for sub in STAGE_SUBDIRS[stage]:
            (self.dir / sub).mkdir(parents=True, exist_ok=True)
        self.log = self._logger()
        self.log.info(f"{PROJECT} v{VERSION} | cenário={SCENARIO} | etapa={stage} | assinatura={self.signature[:12]}")
        self.log.info(f"config: {json.dumps(stage_config(stage), default=str)}" + (" | MODO RÁPIDO (SR_FAST=1)" if FAST and stage != "01_prepare" else ""))

    # ---------- assinatura ----------
    def _signature(self, stage=None, script=None) -> str:
        stage = stage or self.stage
        script = script or self.script
        parts = [Path(script).read_text(encoding="utf-8")]
        parts += [core_section(s) for s in STAGE_CORE_SECTIONS[stage]]
        parts.append(json.dumps(stage_config(stage), sort_keys=True, default=str))
        return sha256_text("\n".join(parts))

    def _logger(self):
        lg = logging.getLogger(f"{PROJECT}.{self.stage}")
        lg.setLevel(logging.INFO)
        lg.handlers.clear()
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
        fh = logging.FileHandler(self.dir / "logs" / f"{self.stage}.log", mode="w", encoding="utf-8")
        sh = logging.StreamHandler(sys.stdout)
        for h in (fh, sh):
            h.setFormatter(fmt)
            lg.addHandler(h)
        return lg

    # ---------- entradas ----------
    def add_input(self, path: Path, label: str | None = None):
        path = Path(path)
        self.inputs[label or path.name] = {"path": str(path), "sha256": sha256_file(path)}

    def require(self, upstream: str) -> Path:
        """Devolve a pasta da etapa anterior compatível (versão atual ou anterior)."""
        expected = self._signature(upstream, CODE_DIR / f"{upstream}.py")
        found = find_compatible(upstream, expected, fallback_fast=True)
        if found is None and os.environ.get("SR_ALLOW_STALE") == "1":
            found = find_compatible(upstream, None, fallback_fast=True)
            if found:
                self.log.warning(f"USANDO {upstream} v{found[0]} COM ASSINATURA DIFERENTE (SR_ALLOW_STALE=1)")
        if found is None:
            cur = stage_dir(upstream) / "manifest.json"
            if cur.exists():
                raise SystemExit(f"PARADO: {upstream} em v{VERSION} foi gerada com outro código/configuração. "
                                 f"Refaça {upstream} (SR_OVERWRITE=1) antes de {self.stage}.")
            raise SystemExit(f"PARADO: nenhuma execução compatível de {upstream} para o cenário '{SCENARIO}'. Rode {upstream}.py antes.")
        v, folder, man = found
        self.upstream[upstream] = {"version": v, "signature": man["signature"], "path": str(folder)}
        self.log.info(f"entrada: {upstream} de v{v} ({folder})")
        return folder

    # ---------- saídas ----------
    def _register(self, path: Path, kind: str, desc: str, item: str | None, rows=None):
        self.outputs.append({"file": str(path.relative_to(self.dir)), "kind": kind, "description": desc,
                             "paper_item": item, "rows": rows, "sha256": sha256_file(path)})
        self.log.info(f"  salvo: {path.relative_to(self.dir)}" + (f" [{item}]" if item else ""))
        return path

    @staticmethod
    def _name(name, item):
        return f"{item}_{name}" if item else name

    def save_data(self, df: pd.DataFrame, name: str, desc: str = "", sub="data"):
        p = self.dir / sub / f"{name}.parquet"
        df.to_parquet(p, index=False)
        return self._register(p, "data", desc, None, len(df))

    def save_table(self, df: pd.DataFrame, name: str, desc: str = "", item: str | None = None, sub="tables", index=False):
        p = self.dir / sub / f"{self._name(name, item)}.csv"
        df.to_csv(p, index=index, encoding="utf-8-sig")
        return self._register(p, "table", desc, item, len(df))

    def save_json(self, obj, name: str, desc: str = "", sub="tables"):
        p = self.dir / sub / f"{name}.json"
        p.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        return self._register(p, "json", desc, None)

    def save_figure(self, fig, name: str, desc: str = "", item: str | None = None, sub="figures"):
        base = self.dir / sub / self._name(name, item)
        fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
        fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
        self._register(base.with_suffix(".png"), "figure_png", desc, item)
        return self._register(base.with_suffix(".pdf"), "figure", desc, item)

    def save_file(self, path: Path, kind: str, desc: str = "", item: str | None = None):
        return self._register(Path(path), kind, desc, item)

    # ---------- fechamento ----------
    def finish(self, notes: str = ""):
        man = {
            "project": PROJECT, "version": VERSION, "scenario": SCENARIO, "stage": self.stage,
            "status": "complete", "signature": self.signature, "prespecified_on": PRESPECIFIED_ON,
            "started": datetime.fromtimestamp(self.t0).isoformat(timespec="seconds"),
            "finished": datetime.now().isoformat(timespec="seconds"),
            "runtime_s": round(time.time() - self.t0, 1),
            "config": stage_config(self.stage), "fast_mode": FAST and self.stage != "01_prepare", "inputs": self.inputs, "upstream": self.upstream,
            "environment": _pkg_versions(), "outputs": self.outputs, "notes": notes,
        }
        (self.dir / "manifest.json").write_text(json.dumps(man, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        reg = OUTPUTS / "RUNS.csv"
        row = pd.DataFrame([{"finished": man["finished"], "version": VERSION, "scenario": SCENARIO, "stage": self.dir.name,
                             "signature": self.signature[:12], "runtime_s": man["runtime_s"], "n_outputs": len(self.outputs),
                             "upstream": ";".join(f"{k}@v{v['version']}" for k, v in self.upstream.items()), "notes": notes}])
        row.to_csv(reg, mode="a", header=not reg.exists(), index=False, encoding="utf-8")
        self.log.info(f"concluído em {man['runtime_s']} s — manifesto: {self.dir / 'manifest.json'}")
# <<< SECTION: run_control

# =============================================================================
# >>> SECTION: read
# =============================================================================
def _read_dta(path: Path, enc):
    import pyreadstat
    try:
        return pyreadstat.read_dta(str(path), apply_value_formats=False, encoding=enc)[0]
    except (OSError, pyreadstat.ReadstatError):
        # caminhos com acento (ex.: DOCÊNCIA) podem falhar no Windows: lê uma cópia temporária
        if str(path).isascii():
            raise
        tmp = Path(tempfile.gettempdir()) / f"sr_{path.name}"
        shutil.copyfile(path, tmp)
        try:
            return pyreadstat.read_dta(str(tmp), apply_value_formats=False, encoding=enc)[0]
        finally:
            tmp.unlink(missing_ok=True)


def read_wave(wave: int) -> pd.DataFrame:
    """Lê a onda com o melhor encoding disponível e converte colunas de texto
    numérico (ex.: r1–r9 e q15–q21 em W3) para número."""
    path = RAW_FILES[wave]
    if not path.exists():
        raise SystemExit(f"PARADO: não encontrei {path}. Copie os .dta do ELSI para data/raw/ "
                         f"com os nomes elsi_w1.dta, elsi_w2.dta e elsi_w3.dta.")
    last = None
    for enc in (None, "latin1"):
        try:
            df = _read_dta(path, enc)
            break
        except Exception as e:  # encoding inválido em alguma string
            last = e
    else:
        raise last
    for c in df.columns[df.dtypes == object]:
        s = df[c].replace("", np.nan)
        conv = pd.to_numeric(s, errors="coerce")
        if s.notna().sum() and conv.notna().sum() >= 0.99 * s.notna().sum():
            df[c] = conv
    return df
# <<< SECTION: read


# =============================================================================
# >>> SECTION: harmonization
# =============================================================================
# Regras verificadas contra os rótulos E os códigos observados nas três ondas
# (ver ELSI_concordancia_ondas_1-3.xlsx). Códigos de ausência são SEMPRE por
# variável e por onda — nunca um conjunto global.

AUDIT: list[dict] = []      # valores inesperados encontrados durante a harmonização
CORRECTIONS: list[dict] = []    # correções de dados aplicadas (vão para S1_data_corrections)


def num(s):
    return pd.to_numeric(s, errors="coerce")


def col(df: pd.DataFrame, name: str) -> pd.Series:
    if name not in df.columns:
        raise KeyError(f"variável '{name}' ausente nesta onda")
    return num(df[name])


def expect(s: pd.Series, allowed, var: str, wave: int) -> pd.Series:
    """Registra valores fora do conjunto esperado (não altera os dados)."""
    allowed = set(allowed)
    bad = s.dropna()[~s.dropna().isin(allowed)]
    if len(bad):
        AUDIT.append({"wave": wave, "variable": var, "unexpected_values": sorted(bad.unique().tolist())[:20],
                      "n": int(len(bad))})
    return s


def na(s: pd.Series, codes) -> pd.Series:
    return s.mask(s.isin(list(codes)))


def yesno(df, var, wave, yes=(1,), no=(0,), missing=(8, 9)):
    s = col(df, var)
    expect(s, set(yes) | set(no) | set(missing), var, wave)
    out = pd.Series(np.nan, index=s.index)
    out[s.isin(yes)] = 1.0
    out[s.isin(no)] = 0.0
    return out


def any_yes(df, vars_, wave):
    """1 se qualquer item = sim; 0 se todos = não; NaN se nenhum sim e algum ausente."""
    m = pd.concat([yesno(df, v, wave) for v in vars_], axis=1)
    out = pd.Series(np.nan, index=df.index)
    out[(m == 0).all(axis=1)] = 0.0
    out[(m == 1).any(axis=1)] = 1.0
    return out


# ---------- sociodemográfico ----------
def h_age(df, w):
    return col(df, "idade")          # 66/77/88/99 são idades reais


def h_male(df, w):
    s = col(df, "sexo"); expect(s, {0, 1}, "sexo", w)
    return s.astype(float)


RACE = {1: "White", 2: "Black", 3: "Brown", 4: "Other", 5: "Other"}


def h_race(df, w):
    s = col(df, "e9"); expect(s, {1, 2, 3, 4, 5, 9}, "e9", w)
    return s.map(RACE)


def h_partner(df, w):
    s = col(df, "e7"); expect(s, {1, 2, 3, 4}, "e7", w)
    return (s == 2).astype(float).where(s.isin([1, 2, 3, 4]))


EDU_YEARS = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9, 11: 10, 12: 11, 13: 11,
             14: 13, 15: 15, 16: 17, 17: 18, 18: 21}


def h_education_years(df, w):
    s = col(df, "e22"); expect(s, set(EDU_YEARS) | {99}, "e22", w)
    return s.map(EDU_YEARS).astype(float)


# ---------- socioeconômico e trabalho ----------
def h_income_quintile(df, w):
    """Quintil da renda domiciliar per capita DENTRO da onda (evita deflacionar
    entre 2015 e 2024). rendadompc não tem rótulos de ausência."""
    s = col(df, "rendadompc")
    return pd.qcut(s.rank(method="first"), 5, labels=False).astype(float) + 1


def h_insurance(df, w):
    return yesno(df, "u1", w, missing=(9,))


WORK_DEMAND = {1: "Sedentary", 2: "Standing/light", 3: "Moderate", 4: "Heavy"}


def h_heavy_work(df, w):
    """1 = trabalho pesado na maior parte da vida.
    ATENÇÃO (v3): W1 i14 foi perguntado a todos sobre QUALQUER trabalho (remunerado ou não);
    W2/W3 i0_9 só a quem teve trabalho PAGO ≥ 6 meses (i0_1). As categorias intermediárias e
    'nunca trabalhou' não são comparáveis (nunca: 1,6% vs 17%; moderado: 31% vs 20%), mas a
    proporção de 'pesado' é estável entre ondas e por sexo -> só o binário é harmonizável.
    Nunca trabalhou (pago) = 0."""
    if w == 1:
        s = col(df, "i14"); expect(s, {1, 2, 3, 4, 5, 9}, "i14", w)
        return (s == 4).astype(float).where(s.isin([1, 2, 3, 4, 5]))
    s = col(df, "i0_9"); expect(s, {1, 2, 3, 4, 8, 9}, "i0_9", w)
    never = col(df, "i0_1") == 0
    out = (s == 4).astype(float).where(s.isin([1, 2, 3, 4]))
    out[(s == 8) & never] = 0.0
    return out


# ---------- contexto ----------
REGION = {1: "North", 2: "Northeast", 3: "Southeast", 4: "South", 5: "Central-West"}


def h_region(df, w):
    s = col(df, "regiao"); expect(s, set(REGION), "regiao", w)
    return s.map(REGION)


def h_urban(df, w):
    """W1/W2 urbano/rural; W3 'área urbanizada/não urbanizada' (ver S1)."""
    s = col(df, "zona"); expect(s, {1, 2}, "zona", w)
    return s.map({1: 1.0, 2: 0.0})


def h_household_size(df, w):
    """Total de moradores (W1 ar6; W2/W3 nmoradores). Mesma definição nas três ondas (= lista de moradores + 1,
    inclusive crianças). v4.0.0: inválido (NA) quando há menos moradores do que entrevistados do mesmo
    domicílio (1,3% na W3)."""
    s = col(df, "ar6" if w == 1 else "nmoradores")
    dom = {1: "iddom", 2: "iddom2", 3: "iddom3"}[w]
    if dom in df.columns:
        n_resp = df.groupby(dom)[dom].transform("size")
        s = s.where(~(s < n_resp))
    return s.where(s >= 1).clip(upper=10)


# ---------- rede social e bem-estar ----------
def h_loneliness(df, w):
    s = col(df, "s17"); expect(s, {1, 2, 3, 8, 9}, "s17", w)
    return na(s, {8, 9})


def h_life_satisfaction(df, w):
    s = col(df, "s18"); expect(s, set(range(0, 11)) | {88, 99}, "s18", w)
    return na(s, {88, 99})


# ---------- comportamental ----------
def _ipaq_minutes_day(df, w, base):
    """Minutos/dia de uma atividade IPAQ. W1: variável única em minutos
    (8888 NA, 9999 NS/NR). W2/W3: horas (_1) + minutos (_2), 88/99."""
    if w == 1:
        s = col(df, base)
        expect(s, set(range(0, 1441)) | {8888, 9999}, base, w)
        return na(s, {8888, 9999})
    h, m = col(df, f"{base}_1"), col(df, f"{base}_2")
    expect(h, set(range(0, 25)) | {88, 99}, f"{base}_1", w)
    expect(m, set(range(0, 60)) | {88, 99}, f"{base}_2", w)
    return na(h, {88, 99}) * 60 + na(m, {88, 99}).fillna(0).where(na(h, {88, 99}).notna())


def _ipaq_week(df, w, days_var, time_base):
    d = col(df, days_var); expect(d, set(range(0, 8)) | {9}, days_var, w)
    d = na(d, {9})
    m = _ipaq_minutes_day(df, w, time_base).clip(upper=CFG.ipaq_cap_min_day)
    out = d * m
    out[d == 0] = 0.0
    return out


def h_mvpa_min_week(df, w):
    walk = _ipaq_week(df, w, "l5", "l6")
    mod = _ipaq_week(df, w, "l7", "l8")
    vig = _ipaq_week(df, w, "l9", "l10")
    return walk + mod + vig


def h_current_smoking(df, w):
    """Fuma atualmente (1) vs não (0). Única informação de tabagismo comparável entre ondas (v4.0.0):
    "ex-fumante"/"nunca" não são — W1 l31 pergunta sobre qualquer consumo no passado, W2/W3 l30_0 sobre ≥100 cigarros.
    W1: l30 (1,2 atual; 3 não).
    W2: l30 (1,2 atual; 3 não; 8 = não se aplica porque l30_0 = 2, nunca fumou 100 cigarros -> 0).
    W3: l30 (1,2 atual; 0 não; 8 = não se aplica porque l30_0 = 0 -> 0)."""
    l30 = col(df, "l30")
    no_now = 3 if w in (1, 2) else 0
    expect(l30, {1, 2, no_now, 8, 9} if w > 1 else {1, 2, 3, 9}, "l30", w)
    out = pd.Series(np.nan, index=df.index)
    out[l30.isin([1, 2])] = 1.0
    out[l30 == no_now] = 0.0
    if w > 1:
        never_code = 2 if w == 2 else 0
        l300 = col(df, "l30_0"); expect(l300, {1, never_code, 9}, "l30_0", w)
        out[(l300 == never_code) & ~l30.isin([1, 2])] = 0.0
    return out


def h_alcohol_freq(df, w):
    s = col(df, "l24"); expect(s, {1, 2, 3, 9}, "l24", w)
    return na(s, {9})


def h_sleep_quality(df, w):
    s = col(df, "n74"); expect(s, {1, 2, 3, 4, 5, 9}, "n74", w)
    return na(s, {9})


# ---------- morbidade ----------
def h_srh(df, w):
    """1 excelente/muito boa … 5 muito ruim. W1 já junta as duas categorias no código 1;
    W2/W3: 0 excelente + 1 muito boa -> 1."""
    s = col(df, "n1")
    if w == 1:
        expect(s, {1, 2, 3, 4, 5, 9}, "n1", w)
        return na(s, {9})
    expect(s, {0, 1, 2, 3, 4, 5, 9}, "n1", w)
    return na(s, {9}).replace({0: 1})


def h_disease(var):
    def f(df, w):
        s = col(df, var); expect(s, {0, 1, 2, 9}, var, w)
        out = s.map({0: 0.0, 1: 1.0, 2: 0.0})      # 2 = só na gravidez (n28/n35)
        return out
    f.__name__ = f"h_disease_{var}"
    return f


def h_heart(df, w):
    return any_yes(df, ["n46", "n48", "n50"], w)


def h_lung(df, w):
    return any_yes(df, ["n54", "n55"], w)


CHRONIC = ["hypertension", "diabetes", "heart_disease", "stroke", "lung_disease", "arthritis",
           "osteoporosis", "cancer", "kidney_disease", "depression_dx"]


def h_falls(df, w):
    return yesno(df, "n18", w, missing=(9,))


def h_weight_loss(df, w):
    return yesno(df, "n69", w, missing=(9,))


# ---------- saúde mental e cognição ----------
CESD_ITEMS = ["r2", "r3", "r4", "r5", "r6", "r7", "r8", "r9"]
CESD_REVERSE = {"r5", "r7"}          # feliz; aproveitou a vida


def h_cesd8(df, w):
    items = []
    for v in CESD_ITEMS:
        s = col(df, v); expect(s, {0, 1, 8, 9}, v, w)
        s = na(s, {8, 9})
        items.append(1 - s if v in CESD_REVERSE else s)
    m = pd.concat(items, axis=1)
    valid = m.notna().sum(axis=1)
    return (m.mean(axis=1) * len(CESD_ITEMS)).round().where(valid >= 6)


def h_memory(var):
    def f(df, w):
        s = col(df, var); expect(s, set(range(0, 11)) | {88, 99}, var, w)
        return na(s, {88, 99})
    f.__name__ = f"h_memory_{var}"
    return f


# ---------- antropometria ----------
def h_height_cm(df, w):
    """W1 em METROS (99999 sem informação); W2/W3 em cm (888 NA, 999 não fez)."""
    s = col(df, "mf13")
    s = na(s, {99999}) * 100 if w == 1 else na(s, {888, 999})
    return s.where(s.between(120, 210))


def h_weight_kg(df, w):
    s = col(df, "mf22")
    s = na(s, {99999}) if w == 1 else na(s, {888, 999})   # 66–99 kg são reais
    return s.where(s.between(25, 250))


def h_bmi(df, w):
    return h_weight_kg(df, w) / (h_height_cm(df, w) / 100) ** 2


def h_waist_cm(df, w):
    s = col(df, "mf17m")
    s = na(s, {99999}) if w == 1 else na(s, {888, 999})
    return s.where(s.between(40, 200))


# ---------- funcionalidade ----------
ADL_ITEMS = ["p37", "p40", "p43", "p46", "p49", "p55"]
IADL_ITEMS = ["p22", "p24", "p26", "p30"]


def _difficulty(df, var, w):
    """1 = alguma dificuldade. Escala detectada nos dados (erro se ambígua):
    graduada 1 sem / 2–4 com dificuldade (5 nunca fez -> NA); binária 0/1."""
    s = col(df, var)
    vals = set(s.dropna().unique()) - {8, 9}
    if vals <= {0, 1} and 0 in vals:
        expect(s, {0, 1, 8, 9}, var, w)
        return na(s, {8, 9})
    if vals <= {1, 2, 3, 4, 5} and vals & {2, 3, 4}:
        expect(s, {1, 2, 3, 4, 5, 8, 9}, var, w)
        return s.map({1: 0.0, 2: 1.0, 3: 1.0, 4: 1.0})
    raise ValueError(f"W{w} {var}: escala não reconhecida {sorted(vals)}")


def _count(df, items, w, min_valid=0.8):
    m = pd.concat([_difficulty(df, v, w) for v in items], axis=1)
    valid = m.notna().sum(axis=1)
    return (m.mean(axis=1) * len(items)).round().where(valid >= np.ceil(min_valid * len(items)))


def h_adl(df, w):
    return _count(df, ADL_ITEMS, w)


def h_iadl(df, w):
    return _count(df, IADL_ITEMS, w)


def h_mobility(df, w):
    return _count(df, list(CFG.mobility_items), w)


# ---------- desempenho físico ----------
# códigos de não realização por onda (valores reais em segundos são < 66)
BAL_CODES = {1: {"unable": {9666, 9999}, "refused": {9888}, "na": {8888}},
             2: {"unable": {66, 99}, "refused": {88}, "na": {888}},
             3: {"unable": {66, 77}, "refused": {88}, "na": {888}}}
GAIT_CODES = {1: {"unable": {9666}, "refused": {9888}, "na": {8888}},
              2: {"unable": {666}, "refused": {888}, "na": {8888}},
              3: {"unable": {666, 777}, "refused": {888}, "na": {8888}}}


def balance_seconds(df, w):
    """Tempos (s) dos três testes de equilíbrio, com a correção da W3.
    W3 (v3): em parte das UPAs os tempos foram registrados DIVIDIDOS POR 10 (10 s -> 1,0;
    30 s -> 3,0; padrão na pessoa: 1.259 de 1.444 com mf30 = 1,0 também têm mf31 = 1,0 e
    mf32 em {1,0; 3,0}). Regra: na W3, mf30/mf31 em (0; 1,1] -> ×10; e, se a pessoa teve
    mf30 ou mf31 corrigido, mf32 em (0; 3,1] -> ×10. Zero = não manteve (fica 0)."""
    codes = BAL_CODES[w]
    allc = codes["unable"] | codes["refused"] | codes["na"]
    t = {v: col(df, v) for v in ("mf30", "mf31", "mf32")}
    if w == 3:
        fix = {}
        for v in ("mf30", "mf31"):
            fix[v] = t[v].between(0.001, 1.1) & ~t[v].isin(allc)
        person = fix["mf30"] | fix["mf31"]
        fix["mf32"] = person & t["mf32"].between(0.001, 3.1) & ~t["mf32"].isin(allc)
        for v in ("mf30", "mf31", "mf32"):
            t[v] = t[v].mask(fix[v], t[v] * 10)
            CORRECTIONS.append({"wave": w, "variable": v, "rule": "value ×10 (recorded in tens of seconds)",
                                "n_corrected": int(fix[v].sum())})
    return t


def h_balance(df, w):
    """0–3 posições mantidas pelo tempo-alvo (lado a lado 10 s, semi-tandem 10 s,
    tandem 30 s se ≤69 anos e 10 s se ≥70). 'Incapaz/arriscado' = 0 naquela posição;
    recusa ou NA = ausente."""
    age = h_age(df, w)
    codes = BAL_CODES[w]
    secs = balance_seconds(df, w)
    parts = []
    for var, target in [("mf30", 10), ("mf31", 10), ("mf32", None)]:
        s = secs[var]
        tgt = pd.Series(np.where(age <= 69, 30, 10), index=df.index) if target is None else target
        ok = (s < 60) & (s >= tgt - 0.05)
        p = ok.astype(float)
        p[s.isin(codes["unable"])] = 0.0
        p[s.isin(codes["refused"] | codes["na"]) | s.isna()] = np.nan
        parts.append(p)
    return pd.concat(parts, axis=1).sum(axis=1, min_count=3)


def h_gait_speed(df, w):
    lo, hi = CFG.gait_time_range
    codes = GAIT_CODES[w]
    t = []
    for var in ("mf35s", "mf38s"):
        s = col(df, var)
        s = s.mask(s.isin(codes["unable"] | codes["refused"] | codes["na"]))
        t.append(s.where(s.between(lo, hi)))
    best = pd.concat(t, axis=1).min(axis=1)
    return CFG.walk_distance_m / best


def h_gait_aid(df, w):
    return yesno(df, "mf39", w, missing=(8, 9))

# ---------- curso de vida (v3; perguntas retrospectivas presentes nas 3 ondas) ----------
def h_rural_childhood(df, w):
    return yesno(df, "e6", w, missing=(9,))


CHILD_SES = {1: "Above average", 2: "Average", 3: "Poor", 4: "Varied a lot"}


def h_childhood_ses(df, w):
    s = col(df, "h10"); expect(s, {1, 2, 3, 4, 9}, "h10", w)
    return s.map(CHILD_SES)


def h_childhood_health(df, w):
    s = col(df, "h11"); expect(s, {1, 2, 3, 4, 9}, "h11", w)
    return na(s, {9})


def h_childhood_hunger(df, w):
    return yesno(df, "h24", w, missing=(9,))


def h_parental_education(df, w):
    """Maior escolaridade entre mãe (e17) e pai (e21): 1 nunca estudou … 6 superior; 8, 9 = NA."""
    parts = []
    for v in ("e17", "e21"):
        s = col(df, v); expect(s, {1, 2, 3, 4, 5, 6, 8, 9}, v, w)
        parts.append(na(s, {8, 9}))
    return pd.concat(parts, axis=1).max(axis=1)


# ---------- morbidade (v3) ----------
def h_teeth(df, w):
    s = col(df, "o6"); expect(s, {1, 2, 3, 4, 9}, "o6", w)
    return na(s, {9})


MULTIMORBIDITY_VARS = ("n28", "n35", "n52", "n56", "n57", "n60", "n61", "n59", "n58")


def h_multimorbidity(df, w):
    """Nº de 11 condições crônicas (as 10 do modelo + coluna). NaN se faltar > 2."""
    parts = [h_disease(v)(df, w) for v in MULTIMORBIDITY_VARS] + [h_heart(df, w), h_lung(df, w)]
    m = pd.concat(parts, axis=1)
    return m.sum(axis=1).where(m.notna().sum(axis=1) >= len(parts) - 2)


# ---------- composição corporal estimada (v3) ----------
def h_whtr(df, w):
    return h_waist_cm(df, w) / h_height_cm(df, w)


def h_rfm(df, w):
    """Relative fat mass (Woolcott & Bergman, 2018): 64 − 20·altura/cintura; mulheres 76 − 20·altura/cintura."""
    male = h_male(df, w)
    return 64 - 20 * h_height_cm(df, w) / h_waist_cm(df, w) + 12 * (1 - male)


def h_ffm_est(df, w):
    """Massa livre de gordura estimada: peso × (1 − RFM/100)."""
    return h_weight_kg(df, w) * (1 - h_rfm(df, w) / 100)
# <<< SECTION: harmonization

# =============================================================================
# >>> SECTION: features
# =============================================================================
# Domínios e camadas PRÉ-ESPECIFICADOS em PRESPECIFIED_ON. Não mudar após ver resultados.
# layer: M1 social, curso de vida e comportamento | M2 +saúde | M3 +corpo e função
DOMAINS = {
    "Sociodemographic": "M1", "Life course": "M1", "Socioeconomic & work": "M1", "Context": "M1",
    "Social & wellbeing": "M1", "Behavioural": "M1",
    "Morbidity": "M2", "Mental health": "M2", "Cognition": "M2",
    "Anthropometry": "M3", "Functional status": "M3", "Physical performance": "M3",
}
# tipo de variável para a hipótese de assimetria: "deficit" só varia entre quem tem problema
# (zero na maioria); "capacity" varia no lado positivo; "context" o resto.
DEFICIT, CAPACITY, CONTEXT = "deficit", "capacity", "context"


@dataclass(frozen=True)
class Feature:
    name: str
    label: str
    domain: str
    kind: str                 # "cont" | "bin" | "ord" | "cat"
    builder: object
    rule: str                 # descrição curta da harmonização (vai para S1)
    role: str = CONTEXT       # deficit | capacity | context

    @property
    def layer(self):
        return DOMAINS[self.domain]


D = h_disease
FEATURES = [
    # M1 — sociodemográfico
    Feature("age", "Age (years)", "Sociodemographic", "cont", h_age, "idade, all waves"),
    Feature("male", "Male sex", "Sociodemographic", "bin", h_male, "sexo 1=male"),
    Feature("race", "Race/skin colour", "Sociodemographic", "cat", h_race, "e9; yellow+indigenous=Other; 9=NA"),
    Feature("partner", "Living with partner", "Sociodemographic", "bin", h_partner, "e7==2"),
    Feature("education_years", "Education (years)", "Sociodemographic", "cont", h_education_years, "e22 grade -> years; 99=NA"),
    # M1 — curso de vida (novo na v3)
    Feature("rural_childhood", "Lived in rural area up to age 15", "Life course", "bin", h_rural_childhood, "e6; 9=NA"),
    Feature("childhood_ses", "Family financial situation up to age 15", "Life course", "cat", h_childhood_ses, "h10 (1 above average, 2 average, 3 poor, 4 varied); 9=NA"),
    Feature("childhood_health", "Self-rated health up to age 15 (1 excellent – 4 poor)", "Life course", "ord", h_childhood_health, "h11; 9=NA"),
    Feature("childhood_hunger", "Went to bed hungry up to age 15", "Life course", "bin", h_childhood_hunger, "h24; 9=NA"),
    Feature("parental_education", "Highest parental education (1 none – 6 university)", "Life course", "ord", h_parental_education, "max(e17 mother, e21 father); 8,9=NA"),
    # M1 — socioeconômico e trabalho
    Feature("income_quintile", "Household income per capita (within-wave quintile)", "Socioeconomic & work", "ord", h_income_quintile, "rendadompc ranked within wave"),
    Feature("health_insurance", "Private health insurance", "Socioeconomic & work", "bin", h_insurance, "u1; 9=NA"),
    Feature("heavy_work", "Heavy physical work most of life", "Socioeconomic & work", "bin", h_heavy_work, "W1 i14==4; W2/W3 i0_9==4; never paid work (i0_1==0)=0; intermediate categories NOT comparable across waves", CAPACITY),
    # M1 — contexto
    Feature("region", "Region", "Context", "cat", h_region, "regiao"),
    Feature("urban", "Urban area", "Context", "bin", h_urban, "zona 1; W3 = urbanised area of city/village"),
    Feature("household_size", "Household size", "Context", "cont", h_household_size, "W1 ar6; W2/W3 nmoradores; capped at 10"),
    # M1 — rede social e bem-estar
    Feature("loneliness", "Loneliness (1 never – 3 always)", "Social & wellbeing", "ord", h_loneliness, "s17; 8,9=NA"),
    Feature("life_satisfaction", "Life satisfaction (0–10)", "Social & wellbeing", "cont", h_life_satisfaction, "s18; 88,99=NA"),
    # M1 — comportamental
    Feature("mvpa_min_week", "MVPA incl. walking (min/week)", "Behavioural", "cont", h_mvpa_min_week, "IPAQ: W1 l6/l8/l10 minutes; W2/W3 hours+minutes; 180 min/day cap", CAPACITY),
    Feature("current_smoking", "Current smoking", "Behavioural", "bin", h_current_smoking, "W1 l30; W2/W3 l30 (+ l30_0 = never 100 cigarettes -> 0); former/never not comparable"),
    Feature("alcohol_freq", "Alcohol frequency (1 never – 3 monthly+)", "Behavioural", "ord", h_alcohol_freq, "l24; 9=NA"),
    Feature("sleep_quality", "Sleep quality (1 very good – 5 very poor)", "Behavioural", "ord", h_sleep_quality, "n74; 9=NA"),
    # M2 — morbidade
    Feature("srh", "Self-rated health (1 excellent/very good – 5 very poor)", "Morbidity", "ord", h_srh, "n1; W2/W3 0->1", DEFICIT),
    Feature("hypertension", "Hypertension", "Morbidity", "bin", D("n28"), "n28; 2 (pregnancy only)=0", DEFICIT),
    Feature("diabetes", "Diabetes", "Morbidity", "bin", D("n35"), "n35; 2 (pregnancy only)=0", DEFICIT),
    Feature("heart_disease", "Heart disease (MI, angina or heart failure)", "Morbidity", "bin", h_heart, "n46|n48|n50", DEFICIT),
    Feature("stroke", "Stroke", "Morbidity", "bin", D("n52"), "n52 (n53 is AGE at diagnosis)", DEFICIT),
    Feature("lung_disease", "Chronic lung disease (asthma, COPD)", "Morbidity", "bin", h_lung, "n54|n55", DEFICIT),
    Feature("arthritis", "Arthritis/rheumatism", "Morbidity", "bin", D("n56"), "n56", DEFICIT),
    Feature("osteoporosis", "Osteoporosis", "Morbidity", "bin", D("n57"), "n57", DEFICIT),
    Feature("cancer", "Cancer (ever)", "Morbidity", "bin", D("n60"), "n60 (n59 is depression)", DEFICIT),
    Feature("kidney_disease", "Chronic kidney disease", "Morbidity", "bin", D("n61"), "n61", DEFICIT),
    Feature("depression_dx", "Diagnosed depression", "Morbidity", "bin", D("n59"), "n59", DEFICIT),
    Feature("spine_problem", "Chronic spine problem", "Morbidity", "bin", D("n58"), "n58; 9=NA", DEFICIT),
    Feature("multimorbidity", "Number of chronic conditions (0–11)", "Morbidity", "cont", h_multimorbidity, "sum of 11 conditions above; NaN if >2 missing", DEFICIT),
    Feature("falls", "Fall in past 12 months", "Morbidity", "bin", h_falls, "n18", DEFICIT),
    Feature("weight_loss", "Unintentional weight loss (3 months)", "Morbidity", "bin", h_weight_loss, "n69", DEFICIT),
    Feature("teeth", "Natural teeth (1 none – 4 ≥20)", "Morbidity", "ord", h_teeth, "o6; 9=NA"),
    # M2 — saúde mental e cognição
    Feature("cesd8", "Depressive symptoms (CES-D 8, 0–8)", "Mental health", "cont", h_cesd8, "r2–r9; r5,r7 reversed; ≥6 valid, prorated; W3 stored as text", DEFICIT),
    Feature("memory_immediate", "Immediate word recall (0–10)", "Cognition", "cont", h_memory("q13"), "q13; 88,99=NA", CAPACITY),
    Feature("memory_delayed", "Delayed word recall (0–10)", "Cognition", "cont", h_memory("q17"), "q17; 88,99=NA", CAPACITY),
    # M3 — antropometria e composição corporal
    Feature("height_cm", "Height (cm)", "Anthropometry", "cont", h_height_cm, "mf13; W1 metres ×100; codes 99999 / 888,999"),
    Feature("weight_kg", "Body weight (kg)", "Anthropometry", "cont", h_weight_kg, "mf22; codes 99999 / 888,999"),
    Feature("bmi", "Body mass index (kg/m²)", "Anthropometry", "cont", h_bmi, "mf22 / (height m)²"),
    Feature("waist_cm", "Waist circumference (cm)", "Anthropometry", "cont", h_waist_cm, "mf17m; codes 99999 / 888,999"),
    Feature("whtr", "Waist-to-height ratio", "Anthropometry", "cont", h_whtr, "mf17m / height"),
    Feature("rfm", "Relative fat mass (%)", "Anthropometry", "cont", h_rfm, "64 (men) or 76 (women) − 20·height/waist"),
    Feature("ffm_est", "Estimated fat-free mass (kg)", "Anthropometry", "cont", h_ffm_est, "weight × (1 − RFM/100)", CAPACITY),
    # M3 — funcionalidade
    Feature("adl_n", "ADL difficulties (0–6)", "Functional status", "cont", h_adl, "p37,p40,p43,p46,p49,p55; W1/W3 graded 2–4=1, W2 binary", DEFICIT),
    Feature("iadl_n", "IADL difficulties (0–4)", "Functional status", "cont", h_iadl, "p22,p24,p26,p30; same rule; 5 never did=NA", DEFICIT),
    Feature("mobility_n", "Mobility difficulties (count)", "Functional status", "cont", h_mobility, "p5–p13,p16 (p14/p15 strength proxies excluded); W1 graded, W2/W3 binary", DEFICIT),
    # M3 — desempenho físico
    Feature("balance_score", "Balance tests completed (0–3)", "Physical performance", "ord", h_balance, "mf30–mf32; unable=0; refused/NA=missing; codes differ by wave; W3 values recorded ÷10 corrected (S1_data_corrections)", CAPACITY),
    Feature("gait_speed", "Usual gait speed (m/s, 3 m)", "Physical performance", "cont", h_gait_speed, "3 / fastest of mf35s, mf38s; 1–60 s", CAPACITY),
    Feature("gait_aid", "Walking aid used in test", "Physical performance", "bin", h_gait_aid, "mf39; 8=NA", DEFICIT),
]
FEATURE_NAMES = [f.name for f in FEATURES]
FEATURE_BY_NAME = {f.name: f for f in FEATURES}
LAYERS = ("M0", "M1", "M2", "M3")
FEATURES_BY_LAYER = {L: [f.name for f in FEATURES if DOMAINS[f.domain] <= L] for L in ("M1", "M2", "M3")}
FEATURES_BY_LAYER["M0"] = ["age", "male", "height_cm", "weight_kg"]      # referência mínima (tamanho corporal)
assert len(set(FEATURE_NAMES)) == len(FEATURE_NAMES)
# <<< SECTION: features

# =============================================================================
# >>> SECTION: outcome
# =============================================================================
GRIP_TRIALS = ["mf27", "mf28", "mf29"]
GRIP_CODES = {1: {"unable": {9555, 9777}, "risky": {9666}, "refused": {9888}, "na": {8888}},
              2: {"unable": {555, 777}, "risky": {666}, "refused": {888}, "na": {8888}},
              3: {"unable": {555, 777}, "risky": {666}, "refused": {888}, "na": {8888}}}
PCT_COLS = ["P5", "P10", "P20", "P30", "P40", "P50", "P60", "P70", "P80", "P90", "P95"]


def grip(df, w):
    """Força máxima (kg) entre as tentativas válidas e o status de quem não tem medida."""
    codes = GRIP_CODES[w]
    allcodes = set().union(*codes.values())
    trials = pd.concat([col(df, v) for v in GRIP_TRIALS], axis=1)
    valid = trials.mask(trials.isin(list(allcodes))).where(lambda x: (x > 0) & (x < 100))
    gmax = valid.max(axis=1)
    status = pd.Series("measured", index=df.index, dtype=object)
    for st in ("na", "refused", "risky", "unable"):     # o último vence (mais informativo)
        status[gmax.isna() & trials.isin(list(codes[st])).any(axis=1)] = st
    status[gmax.isna() & ~trials.isin(list(allcodes)).any(axis=1)] = "missing"
    for v in GRIP_TRIALS:                                # códigos inesperados vão para a auditoria
        s = col(df, v)
        bad = s.dropna()[~(s.dropna().isin(list(allcodes)) | s.dropna().between(0, 99.99))]
        if len(bad):
            AUDIT.append({"wave": w, "variable": v, "unexpected_values": sorted(bad.unique().tolist())[:20], "n": int(len(bad))})
    surgery = yesno(df, "mf25", w, missing=(8, 9))
    return gmax, status, surgery


def load_norms(measure: str) -> pd.DataFrame:
    n = pd.read_csv(NORMS_FILE, comment="#")
    n = n[n["measure"] == measure].copy()
    vals = n[PCT_COLS].to_numpy()
    if not (np.diff(vals, axis=1) > 0).all():
        raise ValueError("normas não monotônicas entre percentis — conferir transcrição")
    n["age_mid"] = np.where(n["age_lo"] >= 100, 102.0, (n["age_lo"] + n["age_hi"]) / 2)
    return n


def norm_curves(age: np.ndarray, sex: str, measure: str) -> np.ndarray:
    """Percentis P5…P95 da norma interpolados na idade exata (n x 11)."""
    n = load_norms(measure)
    tab = n[n["sex"] == sex].sort_values("age_mid")
    a = np.clip(age, tab["age_mid"].min(), tab["age_mid"].max())
    return np.column_stack([np.interp(a, tab["age_mid"], tab[c]) for c in PCT_COLS])


def norm_z(value: pd.Series, age: pd.Series, male: pd.Series, measure: str) -> pd.Series:
    """z relativo às normas: (1) interpola cada percentil na idade exata entre os
    pontos médios das faixas; (2) converte os percentis em z (normal padrão) e
    interpola linearmente valor -> z, com extrapolação linear além de P5/P95."""
    from scipy.stats import norm as N
    zs = N.ppf(np.array([int(c[1:]) for c in PCT_COLS]) / 100)
    out = pd.Series(np.nan, index=value.index)
    for sex, is_male in (("male", 1.0), ("female", 0.0)):
        idx = value.index[(male == is_male) & value.notna() & age.notna()]
        if not len(idx):
            continue
        curve = norm_curves(age[idx].to_numpy(), sex, measure)
        v = value[idx].to_numpy()
        z = np.empty(len(v))
        for i in range(len(v)):
            xs = curve[i]
            if v[i] < xs[0]:
                z[i] = zs[0] + (v[i] - xs[0]) * (zs[1] - zs[0]) / (xs[1] - xs[0])
            elif v[i] > xs[-1]:
                z[i] = zs[-1] + (v[i] - xs[-1]) * (zs[-1] - zs[-2]) / (xs[-1] - xs[-2])
            else:
                z[i] = np.interp(v[i], xs, zs)
        out[idx] = z
    return out


def build_outcome(df, w, age, male, height_cm):
    """Desfecho internacional: z (e centil) da força em relação às normas e quartil (0–3).
    O desfecho condicional é ajustado no 01 (precisa do treino inteiro)."""
    from scipy.stats import norm as N
    gmax, status, surgery = grip(df, w)
    fac = PROTOCOL_FACTORS[CFG.protocol_adjustment]
    # valor comparado às normas = força no protocolo de referência das normas (ver PROTOCOL_FACTORS).
    # grip_max_kg fica BRUTO (referência condicional, cortes EWGSOP2/FNIH, tabelas descritivas).
    adj_kg = gmax * fac["absolute"]
    value = adj_kg if CFG.norm == "absolute" else gmax / (height_cm / 100) ** 2 * fac["normalized"]
    z = norm_z(value, age, male, CFG.norm)
    o = pd.DataFrame({"grip_max_kg": gmax, "grip_adj_kg": adj_kg, "grip_status": status,
                      "hgs_hand_left": yesno(df, "mf26", w, yes=(1,), no=(0,), missing=(8,)),
                      "hand_surgery_3m": surgery, "hgs_value": value, "hgs_z": z, "hgs_centile": N.cdf(z) * 100})
    o["quartile_intl"] = pd.cut(o["hgs_centile"], [-0.01, 25, 50, 75, 100.01], right=False, labels=False).astype(float)
    if CFG.noncompleters == "as_lowest":
        o.loc[o["grip_status"].isin(["unable", "risky"]), "quartile_intl"] = 0.0
    return o


def internal_quartile_cuts(z_train: pd.Series):
    """Cortes dos quartis do z no TREINO (aplicados a todas as ondas, sem olhar o teste)."""
    return z_train.quantile([0.25, 0.5, 0.75]).to_numpy()


def assign_quartile(z: pd.Series, cuts) -> pd.Series:
    return pd.Series(np.searchsorted(cuts, z.to_numpy(), side="right"), index=z.index).astype(float).where(z.notna())


class ConditionalReference:
    """Força esperada para idade, sexo e altura (referência ELSI ajustada SÓ no treino).

    Modelo de localização-escala, separado por sexo:
      média     kg = spline cúbica da idade (nós nos quantis) + termos lineares (altura, [peso])
                     + interação idade × (altura, [peso])
      escala    |resíduo| no mesmo desenho  ->  σ(x), reescalado para var(resíduo/σ) = 1
      forma     distribuição EMPÍRICA (ponderada) dos resíduos padronizados no treino
    centil condicional = F(resíduo/σ) ; z condicional = Φ⁻¹(centil).
    Como na espirometria (GLI), o centil diz onde a pessoa está entre pessoas de mesma
    idade, sexo e altura. Ponderado pelo peso amostral normalizado (referência populacional)."""

    def __init__(self, vars_=None, age_knots=None):
        self.vars = tuple(vars_ or CFG.reference_vars)
        self.age_knots = age_knots or CFG.ref_age_knots
        self.linear = [v for v in self.vars if v not in ("age", "male")]
        self.by_sex = "male" in self.vars
        self.groups_ = {}

    def _groups(self, d):
        return {"male": d["male"] == 1, "female": d["male"] == 0} if self.by_sex else {"all": pd.Series(True, index=d.index)}

    def _design(self, d, g, fit=False):
        parts = [np.ones((len(d), 1))]
        if "age" in self.vars:
            from sklearn.preprocessing import SplineTransformer
            if fit:
                self.groups_[g]["spline"] = SplineTransformer(n_knots=self.age_knots, degree=3, knots="quantile",
                                                              extrapolation="linear", include_bias=False).fit(d[["age"]])
            parts.append(self.groups_[g]["spline"].transform(d[["age"]]))
        if self.linear:
            if fit:
                self.groups_[g]["mu_sd"] = (d[self.linear].mean(), d[self.linear].std())
            m, s = self.groups_[g]["mu_sd"]
            lin = ((d[self.linear] - m) / s).to_numpy()
            parts.append(lin)
            if "age" in self.vars:                       # o efeito da altura muda com a idade
                if fit:
                    self.groups_[g]["age_ms"] = (d["age"].mean(), d["age"].std())
                am, asd = self.groups_[g]["age_ms"]
                parts.append(lin * ((d[["age"]].to_numpy() - am) / asd))
        return np.hstack(parts)

    @staticmethod
    def _wls(X, y, w):
        sw = np.sqrt(w)
        return np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)[0]

    def fit(self, d: pd.DataFrame, y: pd.Series, w: pd.Series):
        ok = d[list(self.vars)].notna().all(axis=1) & y.notna() & w.notna()
        for g, m in self._groups(d).items():
            m = (m & ok).to_numpy()
            self.groups_[g] = {}
            dg, yg, wg = d[m], y[m].to_numpy(), w[m].to_numpy()
            X = self._design(dg, g, fit=True)
            b = self._wls(X, yg, wg)
            r = yg - X @ b
            gam = self._wls(X, np.abs(r), wg)
            sig = np.clip(X @ gam, np.quantile(np.abs(r), 0.05), None)
            c = np.sqrt(np.average((r / sig) ** 2, weights=wg))
            e = r / (sig * c)
            o = np.argsort(e)
            cw = np.cumsum(wg[o])
            self.groups_[g].update(beta=b, gamma=gam, sig_floor=np.quantile(np.abs(r), 0.05), c=c, n=int(m.sum()),
                                   e_sorted=e[o], F=(cw - wg[o] / 2) / cw[-1])
        return self

    def transform(self, d: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        from scipy.stats import norm as N
        out = pd.DataFrame(np.nan, index=d.index, columns=["grip_expected_kg", "grip_sd_kg", "z_cond", "centile_cond"])
        ok = d[list(self.vars)].notna().all(axis=1)
        for g, m in self._groups(d).items():
            m = m & ok
            if not m.any():
                continue
            P = self.groups_[g]
            X = self._design(d[m], g)
            mu = X @ P["beta"]
            sig = np.clip(X @ P["gamma"], P["sig_floor"], None) * P["c"]
            e = (y[m].to_numpy() - mu) / sig
            F = np.interp(e, P["e_sorted"], P["F"])
            out.loc[m, "grip_expected_kg"] = mu
            out.loc[m, "grip_sd_kg"] = sig
            out.loc[m, "centile_cond"] = np.where(np.isnan(e), np.nan, 100 * F)
            out.loc[m, "z_cond"] = np.where(np.isnan(e), np.nan, N.ppf(F))
        return out

    def table(self) -> pd.DataFrame:
        rows = []
        for g, P in self.groups_.items():
            n_lin = len(self.linear) * (2 if "age" in self.vars else 1)
            names = ["intercept"] + ([f"age_spline_{i + 1}" for i in range(len(P["beta"]) - 1 - n_lin)]
                                     if "age" in self.vars else []) + [f"{v}_per_SD" for v in self.linear] \
                + ([f"{v}_x_age" for v in self.linear] if "age" in self.vars else [])
            for nm, b, gm in zip(names, P["beta"], P["gamma"]):
                rows.append({"group": g, "term": nm, "mean_kg": b, "abs_resid_kg": gm})
            rows.append({"group": g, "term": "n_train", "mean_kg": P["n"], "abs_resid_kg": np.nan})
            rows.append({"group": g, "term": "scale_factor_c", "mean_kg": P["c"], "abs_resid_kg": np.nan})
        return pd.DataFrame(rows)


TAILS = {   # nome -> (centil usado, lado, corte); "low" = abaixo/igual, "high" = acima/igual
    "tail_cond_low": ("centile_cond", "low"), "tail_cond_high": ("centile_cond", "high"),
    "tail_intl_low": ("hgs_centile", "low"), "tail_intl_high": ("hgs_centile", "high"),
}


def add_tails(d: pd.DataFrame) -> pd.DataFrame:
    cuts = {"tail_cond_low": CFG.cond_low, "tail_cond_high": CFG.cond_high,
            "tail_intl_low": CFG.intl_low, "tail_intl_high": CFG.intl_high}
    for name, (cvar, side) in TAILS.items():
        c = d[cvar]
        if name == "tail_cond_low":
            t = c <= cuts[name]
        elif name == "tail_intl_low":
            t = c < cuts[name]
        else:
            t = c >= cuts[name]
        d[name] = t.astype(float).where(c.notna())
        if CFG.noncompleters == "as_lowest":
            nc = d["grip_status"].isin(["unable", "risky"])
            d.loc[nc, name] = 1.0 if side == "low" else 0.0
    return d
# <<< SECTION: outcome


# =============================================================================
# >>> SECTION: sample
# =============================================================================
def design_vars(df, w):
    return pd.DataFrame({
        "wave": w,
        "row_id": [f"W{w}_{i}" for i in range(len(df))],
        "cluster": "W" + str(w) + "_" + col(df, "upa").astype("Int64").astype(str),
        "stratum": "W" + str(w) + "_" + col(df, "estrato").astype("Int64").astype(str),
        "weight": col(df, "peso_calibrado"),
        "split": "train" if w in CFG.train_waves else ("test" if w in CFG.test_waves else "unused"),
    }, index=df.index)


def exclusion_flags(df, w, frame):
    """Critérios em ordem (fluxograma). Cada linha recebe o PRIMEIRO motivo."""
    proxy = col(df, "r1") == 1
    dementia = (col(df, "n62") == 1) | (col(df, "n63") == 1)
    st = frame["grip_status"]
    rules = [
        ("age_below_min", frame["age"].isna() | (frame["age"] < CFG.age_min)),
        ("proxy_interview", proxy if CFG.exclude_proxy else pd.Series(False, index=df.index)),
        ("dementia_or_parkinson", dementia if CFG.exclude_dementia else pd.Series(False, index=df.index)),
        ("hand_surgery_3m", frame["hand_surgery_3m"] == 1),
        ("no_physical_measures", st.isin(["na", "missing"])),
        ("grip_refused", st == "refused"),
        ("grip_unable_or_risky", st.isin(["unable", "risky"]) & (CFG.noncompleters == "exclude")),
        ("outcome_not_computable", frame["quartile_intl"].isna()),
    ]
    reason = pd.Series("included", index=df.index, dtype=object)
    for name, mask in rules:
        reason[(reason == "included") & mask.fillna(False)] = name
    return reason, [r[0] for r in rules]
# <<< SECTION: sample

# =============================================================================
# >>> SECTION: modelling
# Configuração de modelagem (afeta só 02+; mudar aqui NÃO invalida o 01).
@dataclass(frozen=True)
class ModelConfig:
    weighted_training: bool = False     # sample_weight = weight_norm no treino (cenário sens_weighted)
    outer_folds: int = 5                # CV externa (desempenho honesto no treino)
    inner_folds: int = 3                # CV interna (tuning)
    n_iter: int = 30                    # combinações por busca aleatória
    n_boot: int = 1000                  # bootstrap por conglomerado dos ICs no teste (W3)
    tasks: tuple = ("Acond", "B", "Aintl", "Tcond_low", "Tcond_high", "Tintl_low", "Tintl_high")
    cond_drop_reference: bool = True    # nas tarefas condicionais, idade/sexo/altura saem dos preditores


_MCFG_KEYS = set(ModelConfig.__dataclass_fields__)
MCFG = replace(ModelConfig(), **{k: v for k, v in SCENARIOS[SCENARIO].items() if k in _MCFG_KEYS})
if os.environ.get("SR_TASKS"):
    MCFG = replace(MCFG, tasks=tuple(t.strip() for t in os.environ["SR_TASKS"].split(",") if t.strip()))
if FAST:
    MCFG = replace(MCFG, outer_folds=3, inner_folds=2, n_iter=3, n_boot=50)

# Tarefas. Regra de escolha do algoritmo primário (pré-especificada): maior métrica
# principal nas predições fora da dobra (OOF) do treino — NUNCA no teste.
_REG = ("ElasticNet", "LightGBM", "XGBoost")
_BIN = ("Logistic", "LightGBM")
TASKS = {
    "Acond": dict(name="Strength for body size (conditional z)", target="z_cond", kind="regression",
                  metric="c_index", features="cond", algorithms=_REG, layers=("M1", "M2", "M3")),
    "B": dict(name="International quartile (ordinal)", target="quartile", kind="ordinal",
              metric="qwk", features="all", algorithms=("OrdinalLogit", "LightGBM", "XGBoost"), layers=("M3",)),
    "Aintl": dict(name="Strength vs world norms (international z)", target="hgs_z", kind="regression",
                  metric="c_index", features="all", algorithms=_REG, layers=("M0", "M3")),
    "Tcond_low": dict(name="Bottom for body size (conditional ≤P10)", target="tail_cond_low", kind="binary",
                      metric="auc", features="cond", algorithms=_BIN, layers=("M3",)),
    "Tcond_high": dict(name="Top for body size (conditional ≥P90)", target="tail_cond_high", kind="binary",
                       metric="auc", features="cond", algorithms=_BIN, layers=("M3",)),
    "Tintl_low": dict(name="Bottom vs world (international <P10)", target="tail_intl_low", kind="binary",
                      metric="auc", features="all", algorithms=_BIN, layers=("M3",)),
    "Tintl_high": dict(name="Top vs world (international ≥P75)", target="tail_intl_high", kind="binary",
                       metric="auc", features="all", algorithms=_BIN, layers=("M3",)),
}
bad = [t for t in MCFG.tasks if t not in TASKS]
if bad:
    raise SystemExit(f"SR_TASKS com tarefa desconhecida: {bad}. Opções: {list(TASKS)}")
ASYMMETRY_PAIRS = {"conditional": ("Tcond_low", "Tcond_high"), "international": ("Tintl_low", "Tintl_high")}
Z_CUTS = {"P10": -1.2815516, "P25": -0.6744898, "P50": 0.0, "P75": 0.6744898, "P90": 1.2815516}


def task_features(task: str, layer: str = "M3") -> list:
    names = FEATURES_BY_LAYER[layer]
    if TASKS[task]["features"] == "cond" and MCFG.cond_drop_reference:
        names = [n for n in names if n not in CFG.reference_vars]
    return names


def feature_lists(names):
    kinds = {f.name: f.kind for f in FEATURES}
    num = [n for n in names if kinds[n] in ("cont", "ord", "bin")]
    cat = [n for n in names if kinds[n] == "cat"]
    return num, cat


def make_preprocessor(names, scale: bool):
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    num, cat = feature_lists(names)
    num_steps = [("imp", SimpleImputer(strategy="median", add_indicator=True))]
    if scale:
        num_steps.append(("sc", StandardScaler()))
    return ColumnTransformer([
        ("num", Pipeline(num_steps), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat),
    ], remainder="drop", verbose_feature_names_out=False)


def original_feature(out_name: str, names) -> str:
    """Nome da variável original a partir da coluna transformada (one-hot, indicador de ausência)."""
    n = out_name.replace("missingindicator_", "")
    for f in sorted(names, key=len, reverse=True):
        if n == f or n.startswith(f + "_"):
            return f
    raise KeyError(out_name)


from sklearn.base import BaseEstimator, ClassifierMixin, clone  # noqa: E402


class FrankHallOrdinal(BaseEstimator, ClassifierMixin):
    """Classificação ordinal (Frank & Hall, 2001): K-1 classificadores binários P(y > k),
    cada um com pesos balanceados; probabilidades por classe forçadas a serem monotônicas.
    models_[0] = P(≥Q2) ("sair da base"); models_[-1] = P(Q4) ("chegar ao topo")."""

    def __init__(self, base=None, balanced: bool = True):
        self.base = base
        self.balanced = balanced

    def fit(self, X, y, sample_weight=None):
        y = np.asarray(y).astype(int)
        self.classes_ = np.arange(int(y.max()) + 1)
        sw = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, float)
        self.models_ = []
        for k in self.classes_[:-1]:
            yk = (y > k).astype(int)
            w = sw.copy()
            if self.balanced:
                p = np.average(yk, weights=sw)
                w = sw * np.where(yk == 1, 0.5 / max(p, 1e-6), 0.5 / max(1 - p, 1e-6))
            m = clone(self.base)
            m.fit(X, yk, sample_weight=w)
            self.models_.append(m)
        return self

    def predict_proba(self, X):
        gt = np.column_stack([m.predict_proba(X)[:, 1] for m in self.models_])
        gt = np.minimum.accumulate(gt, axis=1)                 # P(y>k) não pode crescer com k
        K = len(self.classes_)
        p = np.empty((gt.shape[0], K))
        p[:, 0] = 1 - gt[:, 0]
        for k in range(1, K - 1):
            p[:, k] = gt[:, k - 1] - gt[:, k]
        p[:, K - 1] = gt[:, K - 2]
        p = np.clip(p, 0, None)
        return p / p.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)


def _lgbm(kind):
    import lightgbm as lgb
    common = dict(random_state=SEED, n_jobs=1, verbose=-1, subsample_freq=1)
    return lgb.LGBMRegressor(**common) if kind == "regression" else lgb.LGBMClassifier(**common)


def _xgb(kind):
    import xgboost as xgb
    common = dict(random_state=SEED, n_jobs=1, tree_method="hist")
    return xgb.XGBRegressor(**common) if kind == "regression" else xgb.XGBClassifier(eval_metric="logloss", **common)


_LGBM_GRID = {"n_estimators": [300, 600, 1000], "learning_rate": [0.01, 0.03, 0.05, 0.1],
              "num_leaves": [15, 31, 63], "min_child_samples": [20, 50, 100],
              "subsample": [0.7, 0.85, 1.0], "colsample_bytree": [0.6, 0.8, 1.0], "reg_lambda": [0, 1, 5]}
_XGB_GRID = {"n_estimators": [300, 600, 1000], "learning_rate": [0.01, 0.03, 0.05, 0.1],
             "max_depth": [3, 4, 5, 6], "min_child_weight": [1, 5, 10],
             "subsample": [0.7, 0.85, 1.0], "colsample_bytree": [0.6, 0.8, 1.0], "reg_lambda": [0, 1, 5]}


def algorithm(task: str, name: str):
    """Devolve (estimador, grade de hiperparâmetros com prefixo do pipeline, precisa_escalar)."""
    from sklearn.linear_model import ElasticNet, LogisticRegression
    kind = TASKS[task]["kind"]
    if kind == "regression":
        if name == "ElasticNet":
            return (ElasticNet(max_iter=5000, random_state=SEED),
                    {"clf__alpha": list(np.logspace(-3, 1, 9)), "clf__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]}, True)
        if name == "LightGBM":
            return _lgbm(kind), {f"clf__{k}": v for k, v in _LGBM_GRID.items()}, False
        if name == "XGBoost":
            return _xgb(kind), {f"clf__{k}": v for k, v in _XGB_GRID.items()}, False
    elif kind == "ordinal":
        if name == "OrdinalLogit":
            return (FrankHallOrdinal(LogisticRegression(max_iter=3000)),
                    {"clf__base__C": list(np.logspace(-3, 2, 11))}, True)
        if name == "LightGBM":
            return FrankHallOrdinal(_lgbm(kind)), {f"clf__base__{k}": v for k, v in _LGBM_GRID.items()}, False
        if name == "XGBoost":
            return FrankHallOrdinal(_xgb(kind)), {f"clf__base__{k}": v for k, v in _XGB_GRID.items()}, False
    elif kind == "binary":
        if name == "Logistic":
            return LogisticRegression(max_iter=3000), {"clf__C": list(np.logspace(-3, 2, 11))}, True
        if name == "LightGBM":
            return _lgbm(kind), {f"clf__{k}": v for k, v in _LGBM_GRID.items()}, False
        if name == "XGBoost":
            return _xgb(kind), {f"clf__{k}": v for k, v in _XGB_GRID.items()}, False
    raise ValueError(f"algoritmo {name} não definido para a tarefa {task}")


def make_pipeline(task, name, names):
    from sklearn.pipeline import Pipeline
    est, grid, scale = algorithm(task, name)
    return Pipeline([("pre", make_preprocessor(names, scale)), ("clf", est)]), grid


def scorer(task):
    from sklearn.metrics import cohen_kappa_score, make_scorer
    kind = TASKS[task]["kind"]
    if kind == "regression":
        return "neg_mean_squared_error"
    if kind == "binary":
        return "roc_auc"
    return make_scorer(cohen_kappa_score, weights="quadratic")


def predict(task, model, X):
    kind = TASKS[task]["kind"]
    if kind == "regression":
        return model.predict(X)
    if kind == "binary":
        return model.predict_proba(X)[:, 1]
    return model.predict_proba(X)


# ---------- métricas ----------
def c_index(y, pred):
    """Concordância para desfecho contínuo: P(ordenar corretamente um par) = (tau_b + 1)/2."""
    from scipy.stats import kendalltau
    t = kendalltau(pred, y).statistic
    return (t + 1) / 2


def metrics_reg(y, pred):
    from scipy.stats import norm as N
    from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    slope, _ = np.polyfit(pred, y, 1)
    out = {"c_index": c_index(y, pred), "r2": r2_score(y, pred),
           "rmse_z": float(np.sqrt(mean_squared_error(y, pred))),
           "mae_centile": float(np.mean(np.abs(N.cdf(y) - N.cdf(pred))) * 100),
           "cal_slope": float(slope), "cal_in_large": float(np.mean(y - pred))}
    for lab, cut, sign in (("auc_ge_P75", Z_CUTS["P75"], 1), ("auc_ge_P90", Z_CUTS["P90"], 1),
                           ("auc_lt_P10", Z_CUTS["P10"], -1), ("auc_lt_P25", Z_CUTS["P25"], -1)):
        yy = (y >= cut) if sign == 1 else (y < cut)
        out[lab] = roc_auc_score(yy, sign * pred) if 0 < yy.mean() < 1 else np.nan
    return out


def metrics_ord(y, proba):
    from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, recall_score, roc_auc_score
    y = np.asarray(y).astype(int)
    pred = proba.argmax(axis=1)
    K = proba.shape[1]
    out = {"qwk": cohen_kappa_score(y, pred, weights="quadratic"),
           "accuracy": float(np.mean(pred == y)), "adjacent_accuracy": float(np.mean(np.abs(pred - y) <= 1)),
           "balanced_accuracy": balanced_accuracy_score(y, pred),
           "macro_auc_ovr": roc_auc_score(y, proba, multi_class="ovr", average="macro", labels=list(range(K)))}
    rec = recall_score(y, pred, labels=list(range(K)), average=None, zero_division=0)
    for k in range(K):
        yk = (y == k)
        out[f"auc_Q{k + 1}"] = roc_auc_score(yk, proba[:, k]) if 0 < yk.mean() < 1 else np.nan
        out[f"sens_Q{k + 1}"] = float(rec[k])
    out["auc_ge_Q2"] = roc_auc_score(y >= 1, proba[:, 1:].sum(axis=1)) if 0 < (y >= 1).mean() < 1 else np.nan
    return out


def metrics_bin(y, p):
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
    y, p = np.asarray(y).astype(int), np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    lp = np.log(p / (1 - p))
    # inclinação de calibração (regressão logística de y em logit(p)), por Newton
    b = np.zeros(2)
    X = np.column_stack([np.ones_like(lp), lp])
    for _ in range(25):
        mu = 1 / (1 + np.exp(-X @ b))
        H = X.T @ (X * (mu * (1 - mu))[:, None]) + 1e-9 * np.eye(2)
        b += np.linalg.solve(H, X.T @ (y - mu))
    return {"auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p), "prevalence": float(y.mean()),
            "brier": brier_score_loss(y, p), "cal_slope": float(b[1]), "cal_in_large": float(y.mean() - p.mean())}


def metrics(task, y, p):
    kind = TASKS[task]["kind"]
    return {"regression": metrics_reg, "ordinal": metrics_ord, "binary": metrics_bin}[kind](y, p)


def fast_metric(task):
    """Métrica principal, em versão rápida, para permutação e bootstrap no 03."""
    kind = TASKS[task]["kind"]
    if kind == "regression":
        return c_index
    if kind == "binary":
        from sklearn.metrics import roc_auc_score
        return lambda y, p: roc_auc_score(y, p)

    def qwk_fast(y, proba, K=4):
        pred = proba.argmax(axis=1)
        O = np.bincount(np.asarray(y).astype(int) * K + pred, minlength=K * K).reshape(K, K).astype(float)
        W = (np.subtract.outer(np.arange(K), np.arange(K)) ** 2) / (K - 1) ** 2
        E = np.outer(O.sum(1), O.sum(0)) / O.sum()
        return 1 - (W * O).sum() / (W * E).sum()
    return qwk_fast


# ---------- bootstrap por conglomerado (UPA dentro do estrato) ----------
def cluster_boot_indices(strata, clusters, B, seed=SEED):
    """B reamostras: em cada estrato, sorteia as UPAs com reposição e leva todas as linhas delas."""
    rng = np.random.default_rng(seed)
    clusters = np.asarray(clusters).astype(str)
    strata = np.asarray(strata).astype(str)
    rows = pd.Series(np.arange(len(clusters))).groupby(clusters).apply(lambda s: s.to_numpy()).to_dict()
    by_stratum = [np.array(sorted(set(c))) for _, c in pd.Series(clusters).groupby(strata)]
    out = []
    for _ in range(B):
        picks = [rng.choice(cl, size=len(cl), replace=True) for cl in by_stratum]
        out.append(np.concatenate([rows[c] for p in picks for c in p]))
    return out


def bootstrap_ci(fn, y, pred, strata, clusters, n=None, seed=SEED, alpha=0.05):
    """IC percentil por bootstrap de conglomerados (mesmas reamostras para todas as métricas)."""
    n = n or MCFG.n_boot
    y, pred = np.asarray(y), np.asarray(pred)
    acc = []
    for i in cluster_boot_indices(strata, clusters, n, seed):
        try:
            acc.append(fn(y[i], pred[i]))
        except ValueError:
            continue
    df = pd.DataFrame(acc)
    return {k: (float(df[k].quantile(alpha / 2)), float(df[k].quantile(1 - alpha / 2))) for k in df.columns}
# <<< SECTION: modelling


# =============================================================================
# >>> SECTION: survey
# =============================================================================
# Estimativas com o desenho amostral do ELSI (peso, estrato, UPA). Variância por
# linearização de Taylor (razão para domínios); estratos com uma só UPA não contribuem
# para a variância (registrado em n_singleton_strata).
def svy_mean(y, w, strata, clusters, domain=None, proportion=False):
    from scipy.stats import norm as N
    y = np.asarray(y, float)
    w = np.asarray(w, float)
    d = np.ones(len(y), bool) if domain is None else np.asarray(domain, bool)
    d = d & ~np.isnan(y) & ~np.isnan(w)
    W = w[d].sum()
    if W <= 0:
        return {"est": np.nan, "se": np.nan, "ci_low": np.nan, "ci_high": np.nan, "n": 0, "n_singleton_strata": 0}
    est = float(np.sum(w[d] * y[d]) / W)
    z = np.where(d, w * (np.nan_to_num(y) - est) / W, 0.0)
    t = pd.DataFrame({"h": np.asarray(strata).astype(str), "c": np.asarray(clusters).astype(str), "z": z}) \
        .groupby(["h", "c"])["z"].sum().reset_index()
    var, single = 0.0, 0
    for _, g in t.groupby("h"):
        n = len(g)
        if n < 2:
            single += 1
            continue
        var += n / (n - 1) * float(((g["z"] - g["z"].mean()) ** 2).sum())
    se = float(np.sqrt(var))
    if proportion and 0 < est < 1:          # IC no logito (fica dentro de 0–1)
        lo, hi = np.log(est / (1 - est)) + np.array([-1, 1]) * N.ppf(0.975) * se / (est * (1 - est))
        ci = 1 / (1 + np.exp(-np.array([lo, hi])))
    else:
        ci = est + np.array([-1, 1]) * N.ppf(0.975) * se
    return {"est": est, "se": se, "ci_low": float(ci[0]), "ci_high": float(ci[1]), "n": int(d.sum()),
            "n_singleton_strata": single}


def weighted_quantile(x, w, q):
    x, w = np.asarray(x, float), np.asarray(w, float)
    m = ~np.isnan(x) & ~np.isnan(w)
    x, w = x[m], w[m]
    if not len(x):
        return np.full(np.size(q), np.nan)
    o = np.argsort(x)
    cw = np.cumsum(w[o])
    return np.interp(np.asarray(q) * cw[-1], cw - w[o] / 2, x[o])
# <<< SECTION: survey

# =============================================================================
# >>> SECTION: report
# =============================================================================
# itens do artigo (o 04 usa estes códigos para nomear arquivos)
PAPER_ITEMS = {
    "F1": "Study design and participant flow (ELSI-Brazil waves 1–3)",
    "T1": "Participant characteristics by wave (survey-weighted)",
    "F2": "Where older Brazilians stand: ELSI handgrip centiles against international norms",
    "T2": "National position against international norms by sex, age and social groups (survey-weighted)",
    "T3": "Predictive performance: cross-validation (W1+W2) and temporal validation (W3)",
    "F3": "Decomposition of explained variance by domain (Shapley R²) and the unexplained share",
    "F4": "Top and bottom are not mirror images: domain shares in models of the two tails",
    "F5": "Quantile regression: effects across the conditional strength distribution",
    "S1": "Harmonisation rules and audit by wave",
    "S2": "Exclusions and missing data by wave; included vs excluded",
    "S3": "Reference norms, conditional reference model and its calibration",
    "S4": "Algorithm concordance, permutation importance and SHAP by task",
    "S5": "Within-PSU sensitivity and between-PSU variance",
    "S6": "Sensitivity analyses: key estimates across scenarios",
}
PALETTE = {"low": "#eb6834", "high": "#2a78d6",          # categóricas validadas (base vs topo)
           "groups": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"],   # 4 blocos de domínios, ordem fixa
           "unexplained": "#e4e3df", "norm_band": "#e9e8e4", "norm_line": "#8a8983",
           "ink": "#0b0b0b", "ink2": "#52514e", "elsi": "#2a78d6"}
DOMAIN_GROUPS = {   # agregação para figuras (4 blocos legíveis)
    "Body size & composition": ["Anthropometry"],
    "Physical function": ["Functional status", "Physical performance"],
    "Health": ["Morbidity", "Mental health", "Cognition"],
    "Social, life course & behaviour": ["Sociodemographic", "Life course", "Socioeconomic & work", "Context",
                                        "Social & wellbeing", "Behavioural"],
}
# <<< SECTION: report

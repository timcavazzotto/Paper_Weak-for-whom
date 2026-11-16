"""
run_scenarios.py — roda o cenário principal e TODOS os cenários de sensibilidade, em sequência,
e no fim monta a comparação (06_sensitivity), os critérios de força baixa (08_cutoffs) e as figuras/tabelas finais (07_figures).
================================================================================================
Cada etapa já concluída e compatível (mesma assinatura, nesta versão ou em uma anterior) é pulada.
Por isso o executor pode ser interrompido e rodado de novo: ele continua de onde parou.
Se uma etapa falhar, o erro fica no log e o executor passa para o próximo cenário.

Uso (na raiz do projeto):
    .venv\\Scripts\\python code\\run_scenarios.py                       # tudo (várias horas; ideal à noite)
    .venv\\Scripts\\python code\\run_scenarios.py --fast                # teste rápido de tudo
    .venv\\Scripts\\python code\\run_scenarios.py --only sens_ref_weight,sens_weighted
    .venv\\Scripts\\python code\\run_scenarios.py --list                # mostra o plano e o que já está feito

Log do executor: outputs/run_scenarios_<data-hora>.log
"""
import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

CODE = Path(__file__).resolve().parent
ROOT = CODE.parent

# Tarefas do 02/03 em cada cenário: só as que a sensibilidade testa (economiza horas).
# O cenário principal roda todas.
PLAN = {
    "main": None,
    "sens_ref_weight": "Acond,Tcond_low,Tcond_high",               # peso dentro do esperado
    "sens_weighted": "Acond,Tcond_low,Tcond_high",                 # treino com pesos amostrais
    "sens_with_proxy": "Acond,Tcond_low,Tcond_high,Tintl_low,Tintl_high",
    "sens_noncompleters": "B,Tcond_low,Tcond_high,Tintl_low,Tintl_high",   # incapazes/arriscados na base
    "sens_mobility_all": "Acond,Tcond_low,Tcond_high",
    "sens_normalized": "Aintl,B,Tintl_low,Tintl_high",             # normas por altura²
    "sens_internal_quartiles": "B",
    "sens_train_w1": "Acond,Tcond_low,Tcond_high",                 # treino só na W1 (formato das AVD igual ao da W3)
    "sens_protocol_tomkinson": "Aintl,B,Tintl_low,Tintl_high",     # fator 1,072 (como o ELSI entrou nas normas)
    "sens_protocol_raw": "Aintl,B,Tintl_low,Tintl_high",           # sem ajuste de protocolo (v1–v4)
}
STAGES = ["01_prepare", "02_model", "03_explain", "04_report"]

ap = argparse.ArgumentParser()
ap.add_argument("--fast", action="store_true", help="modo rápido (SR_FAST=1)")
ap.add_argument("--only", default="", help="cenários separados por vírgula")
ap.add_argument("--skip-report", action="store_true", help="não roda o 04 nos cenários de sensibilidade")
ap.add_argument("--list", action="store_true", help="só mostra o plano")
args = ap.parse_args()

scenarios = [s for s in PLAN if not args.only or s in args.only.split(",")]
(ROOT / "outputs").mkdir(exist_ok=True)
logf = ROOT / "outputs" / f"run_scenarios_{datetime.now():%Y%m%d_%H%M}.log"


def say(msg):
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S} | {msg}"
    print(line, flush=True)
    if not args.list:
        with open(logf, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def env_for(scenario):
    env = os.environ.copy()
    for k in ("SR_SCENARIO", "SR_TASKS", "SR_FAST", "SR_OVERWRITE", "SR_SKIP_IF_DONE"):
        env.pop(k, None)
    env["SR_SCENARIO"] = scenario
    env["SR_SKIP_IF_DONE"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    if PLAN[scenario]:
        env["SR_TASKS"] = PLAN[scenario]
    if args.fast:
        env["SR_FAST"] = "1"
    return env


# checagem prévia: todos os pacotes do requirements.txt instalados neste ambiente?
import importlib  # noqa: E402
PKG_IMPORT = {"scikit-learn": "sklearn", "python-docx": "docx", "pyreadstat": "pyreadstat", "pyarrow": "pyarrow"}
missing = []
req = ROOT / "requirements.txt"
if req.exists():
    for line in req.read_text(encoding="utf-8").splitlines():
        name = line.split("#")[0].strip().split(">")[0].split("=")[0].split("<")[0].strip()
        if not name:
            continue
        try:
            importlib.import_module(PKG_IMPORT.get(name, name.replace("-", "_")))
        except ImportError:
            missing.append(name)
if missing:
    print(f"PARADO: faltam pacotes neste ambiente: {', '.join(missing)}\n"
          f"  Rode:  {sys.executable} -m pip install -r requirements.txt")
    sys.exit(1)

say(f"plano: {', '.join(scenarios)} | modo {'RÁPIDO' if args.fast else 'COMPLETO'} | log: {logf.name}")
if args.list:
    for s in scenarios:
        print(f"  {s:26s} tarefas: {PLAN[s] or 'todas'}")
    sys.exit(0)

summary = []
t_all = time.time()
for sc in scenarios:
    stages = STAGES if (sc == "main" or not args.skip_report) else STAGES[:3]
    for st in stages:
        t0 = time.time()
        say(f"[{sc}] {st} ...")
        r = subprocess.run([sys.executable, str(CODE / f"{st}.py")], env=env_for(sc), cwd=ROOT,
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        out = (r.stdout or "") + (r.stderr or "")
        skipped = "nada a fazer" in out
        if r.returncode != 0 and "já tem uma execução concluída" in out:
            say(f"[{sc}] {st} preservada: já existe uma execução concluída nesta versão (código da etapa mudou sem subir VERSION)")
            summary.append((sc, st, "preservada", time.time() - t0))
            continue
        if r.returncode != 0:
            tail = "\n".join(out.strip().splitlines()[-15:])
            say(f"[{sc}] {st} FALHOU (código {r.returncode}). Últimas linhas:\n{tail}")
            summary.append((sc, st, "FALHOU", time.time() - t0))
            break                                   # etapas seguintes dependem desta
        summary.append((sc, st, "já feita" if skipped else "ok", time.time() - t0))
        say(f"[{sc}] {st} {'já feita (pulada)' if skipped else 'ok'} ({(time.time() - t0) / 60:.1f} min)")

# comparação entre cenários
say("06_sensitivity ...")
env = os.environ.copy()
for k in ("SR_SCENARIO", "SR_TASKS", "SR_SKIP_IF_DONE"):
    env.pop(k, None)
env["SR_OVERWRITE"] = "1"
env["PYTHONIOENCODING"] = "utf-8"
if args.fast:
    env["SR_FAST"] = "1"
r = subprocess.run([sys.executable, str(CODE / "06_sensitivity.py")], env=env, cwd=ROOT,
                   capture_output=True, text=True, encoding="utf-8", errors="replace")
say("06_sensitivity " + ("ok" if r.returncode == 0 else "FALHOU:\n" + "\n".join(((r.stdout or "") + (r.stderr or "")).splitlines()[-15:])))

# critérios de força baixa × saúde (cenário principal; lê o 01)
say("08_cutoffs ...")
env8 = env.copy()
env8.pop("SR_OVERWRITE", None)
env8["SR_SKIP_IF_DONE"] = "1"
r = subprocess.run([sys.executable, str(CODE / "08_cutoffs.py")], env=env8, cwd=ROOT,
                   capture_output=True, text=True, encoding="utf-8", errors="replace")
say("08_cutoffs " + ("ok" if r.returncode == 0 else "FALHOU:\n" + "\n".join(((r.stdout or "") + (r.stderr or "")).splitlines()[-15:])))

# desempenho por grupo sociodemográfico (cenário principal; lê 01 e 02)
say("09_subgroups ...")
r = subprocess.run([sys.executable, str(CODE / "09_subgroups.py")], env=env8, cwd=ROOT,
                   capture_output=True, text=True, encoding="utf-8", errors="replace")
say("09_subgroups " + ("ok" if r.returncode == 0 else "FALHOU:\n" + "\n".join(((r.stdout or "") + (r.stderr or "")).splitlines()[-15:])))

# figuras e tabelas finais
say("07_figures ...")
r = subprocess.run([sys.executable, str(CODE / "07_figures.py")], env=env, cwd=ROOT,
                   capture_output=True, text=True, encoding="utf-8", errors="replace")
say("07_figures " + ("ok" if r.returncode == 0 else "FALHOU:\n" + "\n".join(((r.stdout or "") + (r.stderr or "")).splitlines()[-15:])))

say("resumo:")
for sc, st, status, sec in summary:
    say(f"  {sc:26s} {st:12s} {status:9s} {sec / 60:6.1f} min")
say(f"total: {(time.time() - t_all) / 3600:.1f} h")

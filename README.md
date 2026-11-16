# Strength Reserve — ELSI-Brasil (ondas 1–3)

Pipeline em Python do artigo sobre *strength reserve* (força de preensão) no ELSI-Brasil.
Treino nas ondas 1 e 2 (2015–2021), validação temporal na onda 3 (2023–24).

## As três perguntas (v3)

1. **Onde os idosos brasileiros estão** em relação às normas mundiais (Tomkinson et al., 2025)?
   Centil internacional, com o desenho amostral, depois de ajustar a força do ELSI ao protocolo de referência das
   normas (× 1,021: uma mão em vez das duas; ver `PROTOCOL_FACTORS`).
2. **O que explica a força para o próprio corpo?** Centil condicional: força em relação ao
   esperado para idade, sexo e altura, por uma referência ajustada só no treino (como o
   escore z da espirometria).
3. **Topo e base são espelho?** Modelos separados para a base e o topo (condicional ≤P10 / ≥P90;
   mundial <P10 / ≥P75) e comparação dos domínios que explicam cada um.

## Estrutura

```
strength_reserve/
├── code/
│   ├── sr_core.py        núcleo: configuração, versionamento, harmonização, desfechos, modelos, desenho amostral
│   ├── 01_prepare.py     base analítica, referência condicional, caudas, suplemento de harmonização
│   ├── 02_model.py       7 tarefas: CV aninhada, tuning, validação temporal (W3)
│   ├── 03_explain.py     permutação, SHAP, assimetria topo × base, Shapley R², regressão quantílica, UPA
│   ├── 04_report.py      tabelas (.csv + .docx) e figuras (.pdf + .png) do artigo
│   ├── 06_sensitivity.py comparação das estimativas-chave entre cenários (tabela e figura S6)
│   ├── 07_figures.py     figuras combinadas e tabelas finais do artigo (lê 01–06 e 08; não recalcula nada)
│   ├── 08_cutoffs.py     critérios de força baixa (EWGSOP2, FNIH, internacional, condicional) × desfechos de saúde
│   ├── 09_subgroups.py   desempenho dos modelos por grupo sociodemográfico na W3 (justiça do modelo)
│   └── run_scenarios.py  roda o principal + todas as sensibilidades e, no fim, o 06
├── data/
│   ├── raw/              elsi_w1.dta, elsi_w2.dta, elsi_w3.dta  (NÃO versionar; dados sob termo da Fiocruz)
│   └── reference/        tomkinson2025_hgs_norms.csv
├── outputs/
│   ├── RUNS.csv          registro de todas as execuções
│   └── v<VERSÃO>/<CENÁRIO>/<ETAPA>/   manifest.json, data/, tables/, figures/, models/, docx/, logs/
├── requirements.txt
├── CHANGELOG.md
└── README.md
```

## Instalação no Windows (uma vez por computador)

1. Descompacte o zip dentro de
   `C:\Users\Usuario\Documents\ARQUIVOS TIMOTHY\DOCÊNCIA\UNICENTRO\2026\DEMED\PipeLab\Pesquisa\Timothy\MuscleReserve\Timothy`.
   Vai aparecer a pasta `strength_reserve`.
2. Copie os três arquivos do ELSI para `strength_reserve\data\raw\` com estes nomes exatos:
   `elsi_w1.dta`, `elsi_w2.dta`, `elsi_w3.dta`.
3. No VS Code: **File > Open Folder** e escolha a pasta `strength_reserve`. Abra o terminal
   (**Terminal > New Terminal**, PowerShell) e rode:

```powershell
py -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -r requirements.txt
```

   Depois, **Ctrl+Shift+P > Python: Select Interpreter** e escolha o `.venv`.

O caminho tem espaço e acento (`ARQUIVOS TIMOTHY`, `DOCÊNCIA`). Isso é suportado; se a leitura
de um `.dta` falhar por causa do acento, o código lê uma cópia temporária automaticamente.

## Como rodar

Sempre a partir da pasta `strength_reserve` (a raiz do projeto).

**1. Teste rápido primeiro (~15–30 min no total):**

```powershell
.venv\Scripts\python code\01_prepare.py
$env:SR_FAST="1"
.venv\Scripts\python code\02_model.py
.venv\Scripts\python code\03_explain.py
.venv\Scripts\python code\04_report.py
Remove-Item Env:SR_FAST
```

O modo rápido grava em pastas `*_fast` e não atrapalha a execução completa.

**2. Execução completa:**

```powershell
.venv\Scripts\python code\02_model.py      # a etapa mais longa: algumas horas (depende dos núcleos)
.venv\Scripts\python code\03_explain.py    # ~30–90 min
.venv\Scripts\python code\04_report.py     # ~1 min
```

O 01 não precisa ser refeito: ele é igual nos dois modos.

Para dividir o 02 em partes (por exemplo, rodar à noite):

```powershell
$env:SR_TASKS="Acond,Tcond_low,Tcond_high"; .venv\Scripts\python code\02_model.py
```

Cada subconjunto de tarefas gera uma assinatura diferente. O 03 e o 04 precisam ser rodados
com o mesmo `SR_TASKS` (ou rode tudo de uma vez, sem `SR_TASKS`).

## Critérios de força baixa × saúde (08)

```powershell
.venv\Scripts\python code\08_cutoffs.py      # ~10–20 min (bootstrap); depois rode o 07
.venv\Scripts\python code\09_subgroups.py    # ~5–15 min (bootstrap por grupo); depois rode o 07
```

Compara EWGSOP2, FNIH, FNIH-IMC, Tomkinson <P10 e o centil condicional ≤P10 contra internação, urgência,
consultas, saúde ruim, AVD e marcha lenta, nos cortes usuais e na mesma prevalência. Só precisa do 01 e dos `.dta`.

## Figuras e tabelas finais

```powershell
.venv\Scripts\python code\07_figures.py
```

Gera `Fig1`–`Fig5`, `FigS1`–`FigS5` (PDF vetorial + PNG 600 dpi) e as tabelas 1–4 (`.csv` e `.docx`) em
`outputs\v<versão>\main\07_figures\`. Roda em segundos e pode ser repetido à vontade (`$env:SR_OVERWRITE="1"`).
No topo do arquivo: `BUDGET` (o que vai para o texto principal), `WIDTH_MM`, fonte, cores e `SHOW_BADGES`.
O `run_scenarios.py` já chama o 07 no fim.

## Cenários de sensibilidade

O jeito mais simples é o executor, que roda tudo em sequência (ideal deixar à noite):

```powershell
.venv\Scripts\python code\run_scenarios.py --list     # mostra o plano
.venv\Scripts\python code\run_scenarios.py            # roda tudo e, no fim, o 06 (tabela e figura S6)
```

- Cada cenário roda só as tarefas que ele testa (ver `PLAN` no `run_scenarios.py`).
- Etapas já concluídas e compatíveis são puladas, inclusive as da versão anterior. Se o computador desligar,
  é só rodar de novo: ele continua de onde parou.
- Se um cenário falhar, o erro vai para `outputs/run_scenarios_<data>.log` e o executor segue para o próximo.
- `--fast` faz o teste rápido de tudo; `--only sens_ref_weight,sens_weighted` roda só alguns.

| Cenário | O que muda | Tarefas |
|---|---|---|
| `main` | — | todas |
| `sens_ref_weight` | o peso também entra no esperado (referência condicional) | Acond, caudas condicionais |
| `sens_weighted` | treina com os pesos amostrais | Acond, caudas condicionais |
| `sens_with_proxy` | inclui entrevistas com proxy | Acond, 4 caudas |
| `sens_noncompleters` | incapaz/arriscado entram na base | B, 4 caudas |
| `sens_mobility_all` | mobilidade com os itens p14/p15 | Acond, caudas condicionais |
| `sens_normalized` | normas por altura² (kg/m²) | Aintl, B, caudas mundiais |
| `sens_internal_quartiles` | quartis do z no treino em vez dos internacionais | B |
| `sens_train_w1` | treino só na W1 (AVD/AIVD no mesmo formato da W3) | Acond, caudas condicionais |
| `sens_protocol_tomkinson` | ajuste de protocolo × 1,072 (como Tomkinson codificou o ELSI) | Aintl, B, caudas mundiais |
| `sens_protocol_raw` | sem ajuste de protocolo (valor bruto, como v1–v4) | Aintl, B, caudas mundiais |

Também dá para rodar um cenário à mão:

```powershell
$env:SR_SCENARIO="sens_ref_weight"; $env:SR_TASKS="Acond,Tcond_low,Tcond_high"
.venv\Scripts\python code\01_prepare.py
.venv\Scripts\python code\02_model.py
.venv\Scripts\python code\03_explain.py
Remove-Item Env:SR_SCENARIO; Remove-Item Env:SR_TASKS
```

## Versionamento

- A versão fica em `sr_core.py` (`VERSION = "MAJOR.MINOR.PATCH"`). Toda mudança sobe a versão e ganha uma linha no `CHANGELOG.md`.
  - **MAJOR**: muda desfecho, exclusões ou harmonização (refaz tudo).
  - **MINOR**: muda modelagem ou explicação (refaz 02 em diante).
  - **PATCH**: muda só tabelas/figuras (refaz o 04).
- Cada etapa tem uma assinatura (hash do código e da configuração que a afetam). Uma etapa só lê a anterior se a assinatura bater; se a etapa anterior não existir na versão atual, ela é reaproveitada de uma versão anterior com a mesma assinatura.
- Uma etapa concluída não é sobrescrita por engano: para refazer de propósito, `$env:SR_OVERWRITE="1"` (e depois `Remove-Item Env:SR_OVERWRITE`).
- Todas as execuções ficam em `outputs/RUNS.csv`.

## Formatos

| Tipo | Formato |
|---|---|
| dados intermediários | `.parquet` |
| tabelas | `.csv` (UTF-8 com BOM, abre no Excel) e `.docx` (04) |
| figuras | `.pdf` (vetorial) + `.png` (300 dpi) |
| modelos | `.joblib` |
| metadados | `.json`, log `.log` |

Arquivos que vão para o artigo levam o código do item (`T1_`, `F2_`, `S3_` …), definido em `PAPER_ITEMS` no `sr_core.py`.

# CHANGELOG

Formato: versão — data — o que mudou — etapas a refazer.

## 5.1.1 — 2026-11-08
- PATCH: `09_subgroups.py` informa o n real de cada grupo (antes 0 quando o grupo era pequeno demais para estimar);
  `07_figures.py` omite grupos sem ninguém na W3. Refazer só 09 e 07.

## 5.1.0 — 2026-11-08
- MINOR (nova etapa; 01–06 da 5.0.x são reaproveitados; refaz 08 e 07 e roda o novo 09).
- **Novo `09_subgroups.py`** (justiça do modelo, TRIPOD+AI 14 e 23a): desempenho na W3 do algoritmo primário de
  cada tarefa (Acond, Aintl, caudas condicionais e internacionais) dentro de sexo, faixa etária, região,
  escolaridade, raça/cor e quintil de renda. C-index, R², inclinação de calibração (contínuas); AUC, inclinação e
  razão O/E (caudas). IC 95% por bootstrap de UPAs (B_SUB=500). Grupos com < 100 pessoas ou < 20 eventos não são
  estimados. Saídas: `T9_subgroup_long.csv`, `T9_subgroup_performance.csv`.
- **`08_cutoffs.py`: OR bruto** (sem ajuste; nas ondas empilhadas, só onda) ao lado do ajustado, para cada
  critério e para os grupos discordantes (STROBE 16a). Colunas novas em C4 e C6.
- **`07_figures.py`:** Table S4 nova (desempenho por grupo); Table S6 (critérios × saúde) com OR bruto e ajustado;
  tabelas suplementares renumeradas na ordem de citação do texto: S1 posição nacional · S2 sensibilidade ·
  S3 desempenho · S4 subgrupos · S5 assimetria · S6 critérios × saúde · S7 discordantes.
- `run_scenarios.py` roda o 09 depois do 08.
- Rodar só o que mudou: `.venv\Scripts\python code\08_cutoffs.py`, depois `code\09_subgroups.py`, depois
  `code\07_figures.py`.

## 5.0.2 — 2026-11-08
- PATCH só do `07_figures.py` (01–06 e 08 da 5.0.0/5.0.1 são reaproveitados; refaz apenas o 07).
- Nova **Table S6** (ID interno `Table7`): cenários de sensibilidade lado a lado — posição nacional (centil mediano,
  % abaixo de P25), desempenho condicional (R², AUC base e topo) e assimetria topo − base (antropometria, estado
  funcional, capacidade), lida do `06_sensitivity/S6_sensitivity_wide.csv`. "= main" quando o cenário não muda a
  comparação com as normas; IC na linha de baixo; negrito quando o IC da assimetria exclui zero.
- Rodar: `.venv\Scripts\python code\07_figures.py` (cenário principal).

## 5.0.1 — 2026-11-07
- PATCH de robustez (01 e 02 da 5.0.0 são reaproveitados; refaz 03 em diante).
- `03_explain.py`, bootstrap com reajuste: no Windows passa a rodar em SEQUÊNCIA (cada modelo usando todos os núcleos)
  em vez de vários modelos em fios paralelos, que derrubavam o processo (código 3221225477). `SR_REFIT_PARALLEL=1`
  volta ao modo antigo. `statsmodels` importado no início (falha logo, não depois de 35 min).
- `run_scenarios.py`: confere antes de começar se todos os pacotes do `requirements.txt` estão instalados.

## 5.0.0 — 2026-11-07
- MAJOR (muda o desfecho internacional: refaz tudo, 01–08, em todos os cenários).
- **Ajuste de protocolo antes de comparar com as normas.** Tomkinson et al. (2025) harmonizaram cada estudo para um
  protocolo de referência (hidráulico, sentado, cotovelo fletido, antebraço neutro, empunhadura ajustada, **as duas
  mãos**, 3 tentativas, máximo; Tabela Suplementar 3E). O ELSI usa dinamômetro hidráulico SAEHAN (Moreira et al.,
  BMC Geriatr 2022) e **uma mão**, então a força comparada às normas passa a ser força máxima × **1,021**
  (normalizada × 1,020). `PROTOCOL_FACTORS` e `Config.protocol_adjustment` no `sr_core.py`.
  - `grip_max_kg` continua BRUTO: referência condicional (invariante a uma constante), cortes EWGSOP2/FNIH e tabelas
    descritivas. Nova coluna `grip_adj_kg` (força no protocolo das normas), usada nas figuras ELSI × normas.
  - Afeta: centil/z internacional, quartis internacionais, caudas mundiais, modelos Aintl, B, Tintl, Tabela 2, Fig 1c–d,
    critério "Intl <P10" no 08. Não afeta: centil condicional, Acond, caudas condicionais, assimetria condicional.
- Novas sensibilidades: `sens_protocol_tomkinson` (× 1,072 / 1,067, o fator que Tomkinson aplicou ao ELSI, que eles
  codificaram como dinamômetro mecânico) e `sens_protocol_raw` (sem ajuste, como v1–v4). Tarefas: Aintl, B, Tintl.
- Prévia (W3, ponderado): abaixo do P25 54,1% (bruto) -> 50,8% (× 1,021) -> 42,9% (× 1,072); ≥P75 4,3 -> 5,5 -> 9,1%.
- A onda 2 do ELSI (n = 7.905) é um dos 100 estudos das normas; as estimativas principais vêm da onda 3 (fora das normas).
- `run_scenarios.py` agora roda também o 08 (antes do 07). `06_sensitivity.py`: posição nacional com % <P10, <P25, ≥P75.
- `07_figures.py`: notas das Tabelas 1, 2 e 5 descrevem o ajuste; marcadores para até 11 cenários.

## 4.2.1 — 2026-11-06
- PATCH (só o 07). Estrutura do artigo para a JSHS: no máximo 6 itens (figuras + tabelas) no texto principal.
  Texto principal: Table 1 · Fig 1 desenho e posição nacional · Fig 2 força para o corpo (nova, fusão das antigas
  Figs 2 e 3: observado × previsto, ROC base × topo, Shapley R², SHAP) · Fig 3 hipóteses base × topo · Fig 4 onde está a
  assimetria · Fig 5 pontos de corte. Suplemento: FigS1–S7 (inclui as antigas Figs 2 e 3 completas) e Tables S1–S5
  (posição nacional, desempenho, assimetria, critérios × desfechos, discordantes).
- Numeração automática das figuras pela ordem do `BUDGET` (IDs internos ficam nas funções; mapa em `tables/figure_map.csv`).
- Rodar só o 07.

## 4.2.0 — 2026-11-06
- MINOR. `08_cutoffs.py`: discordantes EWGSOP2 × condicional também **dentro de faixas etárias** (50–64, 65–74, ≥75;
  modelo reajustado em cada faixa; grupo com < 30 pessoas na faixa não é estimado). Responde à preocupação de que
  o ajuste por idade extrapola, já que "só EWGSOP2" (77 anos) e "só condicional" (57 anos) quase não se sobrepõem.
  Novas tabelas `C7_discordant_by_age_profile` e `C7_discordant_by_age_or`.
- `07_figures.py`: Table6 ganha o bloco por faixa etária (ondas empilhadas).
- Rodar: 08 e 07 (01–06 reaproveitados da v4.0.0).

## 4.1.0 — 2026-11-06
- MINOR (análise nova; nada de 01–06 precisa ser refeito: a v4.0.0 é reaproveitada).
- `08_cutoffs.py` (novo): qual critério de força baixa discrimina melhor a saúde real? Validade concorrente
  (transversal) de EWGSOP2, FNIH, FNIH-IMC, Tomkinson <P10 e centil condicional ≤P10 contra internação e
  urgência (12 meses; principais), ≥6 consultas, saúde ruim, AVD e marcha lenta. Nos cortes usuais e na
  **mesma prevalência** (cada escore cortado, por sexo e onda, na prevalência do EWGSOP2). OR ajustado por idade,
  idade² e sexo, ganho de AUC, NRI do condicional contra cada critério, discordantes EWGSOP2 × condicional.
  Ponderado; IC 95% por bootstrap de conglomerados (`B_CUT`, padrão 500). W3 principal; W1–W3 de apoio.
  Lê só o 01 e os .dta (u42, u49, u6). Tabelas C1–C6 em `outputs/v4.1.0/main/08_cutoffs/tables/`.
- `07_figures.py`: Fig6 (quem cada critério classifica, OR e ΔAUC na mesma prevalência, discordantes), Table5
  (critérios × desfechos) e Table6 (discordantes); nota na Fig4 sobre o marcador do `sens_train_w1` na linha
  de capacidade (o crédito antropométrico migra do IMC para a massa livre de gordura).
- `sr_core.py`: etapa 08 registrada (seção run_control; não muda assinaturas de 01–07).

## 4.0.0 — 2026-11-05
- MAJOR (muda harmonização: refaz tudo, 01–07, em todos os cenários).
- **Tabagismo**: sai `smoking` (atual / ex / nunca), entra `current_smoking` (fuma atualmente, 0/1). "Ex-fumante" e
  "nunca" não são comparáveis entre ondas: W1 l31 pergunta sobre qualquer consumo no passado ("já fumou" 55%),
  W2/W3 l30_0 sobre ≥100 cigarros na vida (34%). Fumante atual: 17,0% / 12,7% / 12,2%.
- **Tamanho do domicílio**: conferido que a definição é a mesma nas três ondas (= lista de moradores + 1, inclusive
  crianças). Fica NA quando o domicílio tem menos moradores do que entrevistados do mesmo domicílio (1,3% na W3).
  A queda W1 3,1 -> W3 2,2 moradores (mora só: 12% -> 30%) não é erro de código; vai como limitação.
- **Novo cenário `sens_train_w1`**: treino só na W1, validação na W3. A W2 perguntou as AVD/AIVD em formato binário
  ("tem alguma dificuldade", prevalência ~metade), a W1 e a W3 em formato graduado; o cenário testa se as conclusões
  dependem da W2. Tarefas: Acond e caudas condicionais.
- Auditoria onda a onda dos 55 preditores (texto das perguntas conferido). Mesmo enunciado, mas níveis diferentes na W1
  (limitação, sem correção possível): perda de peso, CES-D, depressão, doença cardíaca e pulmonar, plano de saúde.
- `04_report.py` / `07_figures.py`: Tabela 1 mostra fumante atual (linha binária) no lugar das três categorias.
- PRESPECIFIED_ON mantido (2026-11-04): domínios, desfechos e hipóteses não mudaram.

## 3.1.2 — 2026-11-05
- PATCH. `07_figures.py`, Fig4: marcador de cada cenário fixo em todas as linhas e na legenda (antes dependia da posição
  na linha e não batia com a legenda); só entram na legenda os cenários com caudas condicionais. Nova linha
  "Social & wellbeing" (bloco "Not hypothesised"), achado robusto na base.
- `run_scenarios.py`: etapa bloqueada pela trava de sobrescrita aparece como "preservada", não como "FALHOU".

## 3.1.1 — 2026-11-05
- PATCH (só figuras e tabelas; nenhuma etapa de 01–06 precisa ser refeita).
- `07_figures.py` (novo): figuras combinadas em painéis (a, b, c…) no padrão dos artigos de referência
  (caixas tracejadas, título + subtítulo por painel, Okabe-Ito: laranja = base, azul = topo; 180 mm; PDF + PNG 600 dpi):
  Fig1 desenho e posição nacional · Fig2 desempenho (ROC, calibração, algoritmos, camadas) · Fig3 explicação
  (SHAP, permutação com 3 algoritmos, Shapley R², dependência) · Fig4 quadro de hipóteses H1–H3 (participação,
  diferença topo − base, sensibilidades; selos opcionais) · Fig5 onde está a assimetria (|SHAP| base × topo,
  regressão quantílica, região) · FigS1–S5. Tabelas finais 1–4 em .csv e em `tables_main.docx` / `tables_supplement.docx`.
- `BUDGET` no topo do 07 define o que vai para o texto principal e para o suplemento (muda conforme a revista).
- `run_scenarios.py` roda o 07 no fim.
- `04_report.py`: não tenta desenhar a F4 em cenários sem caudas (ex.: `sens_internal_quartiles`, só o modelo B) — corrigia o erro "Number of columns must be a positive integer, not 0".
- Tabelas em formato de artigo (.docx): três linhas horizontais e nenhuma vertical, título acima, notas abaixo,
  grupos com níveis recuados, cabeçalho em dois níveis quando precisa, uma tabela por página (paisagem quando larga),
  Times New Roman 9 pt (`TABLE_FONT`, `TABLE_PT`). Diferenças com IC que exclui 0 em negrito (Tabela 4).
  Numeração automática conforme `BUDGET` (Table 1… no texto principal; Table S1… no suplemento).
- (resolvido na 4.0.0) tabagismo não é comparável entre ondas — W1 l31 "fumou no passado"
  (qualquer consumo; "já fumou" 55%) vs W2/W3 l30_0 "≥100 cigarros" (34%). Só "fuma atualmente" é harmonizável.
  Raça/cor "Outra" 3,0% na W1 vs 0,5% depois: diferença real de declaração (indígena/amarela), não de código.

## 3.1.0 — 2026-11-04
- MINOR (01 e 02 da v3.0.0 são reaproveitados no cenário principal; refaz 03 e 04).
- `03_explain.py`: IC da assimetria base × topo com **bootstrap com reajuste dos modelos** (UPAs do treino reamostradas,
  mesmos hiperparâmetros do 02, SHAP na mesma amostra da W3; `B_REFIT`, padrão 50) -> `F4_asymmetry_refit.csv`.
  O IC antigo (só pessoas da W3, modelo fixo) fica em `asymmetry_domain.csv` e não vai para o artigo.
- `04_report.py`: F4/T4 usam o IC com reajuste; barras de erro nunca negativas (corrige o erro da v3.0.0).
- `run_scenarios.py` (novo): roda o principal e as 7 sensibilidades em sequência, cada uma só com as tarefas que
  ela testa; pula etapas já concluídas e compatíveis (`SR_SKIP_IF_DONE`), registra tudo num log e continua se um cenário falhar.
- `06_sensitivity.py` (novo): tabela S6 (longa e larga, .csv e .docx) e figura S6 comparando amostra, desempenho,
  assimetria, regressão quantílica e posição nacional entre cenários.
- `sr_core.py`: `find_compatible` (busca de etapa compatível usada por `require` e pelo modo pular); etapa 06; item S6.
- Checagem da marcha da W3: 7,8% de tempos inteiros (W1 1,7%) e nenhum < 2 s; sem erro de escala, sem correção
  (limitação: mais arredondamento na W3).
- Etapas: 03, 04, 06 (e as sensibilidades).

## 3.0.0 — 2026-11-04
- Novo desfecho principal (MAJOR): **centil condicional** — força em relação ao esperado para idade, sexo e altura.
  Referência de localização-escala por sexo (spline de idade + altura + interação idade × altura; escala pelo
  |resíduo|; forma pela distribuição empírica), ajustada só no treino e ponderada; calibração em `S3_reference_calibration`.
  Sensibilidade `sens_ref_weight` (peso no esperado).
- **Caudas pré-especificadas**: condicional ≤P10 / ≥P90; mundial <P10 / ≥P75. Hipótese: base e topo têm determinantes diferentes.
- Harmonização:
  - **Correção da W3**: tempos de equilíbrio (mf30–mf32) registrados ÷10 em parte das UPAs (10 s -> 1,0; 30 s -> 3,0);
    corrigidos ×10 com regra por pessoa (`S1_data_corrections`). Média do escore de equilíbrio na W3: 1,72 -> 2,24 (W2 2,28).
  - **`work_demand` -> `heavy_work`**: W1 perguntou a todos sobre qualquer trabalho; W2/W3 só a quem teve trabalho pago
    ≥ 6 meses. Só "trabalho pesado" é comparável entre ondas.
  - Novos preditores: curso de vida (área rural e situação financeira até 15 anos, saúde na infância, fome na infância,
    maior escolaridade dos pais), problema crônico de coluna, nº de condições crônicas, dentes naturais, relação
    cintura-altura, massa gorda relativa (RFM) e massa livre de gordura estimada.
  - Saem: `bedridden` (só existe na W3), `lives_alone` (redundante com o tamanho do domicílio).
  - Avaliados e NÃO incluídos: idade em que começou a trabalhar (pergunta diferente na W1), dor (só W2/W3),
    panturrilha (só W3), quadril (só W1/W2). Não há identificador de entrevistador nem data da entrevista em W2/W3.
- 12 domínios (novo: Life course), 55 preditores; papel de cada variável (déficit / capacidade / contexto).
- `02_model.py`: 7 tarefas (Acond, B, Aintl, Tcond_low, Tcond_high, Tintl_low, Tintl_high); camada M0 (idade, sexo,
  altura, peso) no Aintl; IC 95% por **bootstrap de conglomerados** (UPA dentro do estrato); `SR_TASKS`.
- `03_explain.py`: assimetria topo × base (SHAP pareado na mesma amostra da W3, IC bootstrap; por domínio e por papel),
  decomposição Shapley do R² por domínio (MQO, 2^12 combinações), regressão quantílica com teste β(0,90) − β(0,10),
  ICC por UPA e R² por domínio com efeitos fixos de UPA.
- `04_report.py` (novo): T1 e T2 com o desenho amostral (linearização de Taylor), T3, T4, figuras F2–F5 e S3, `.docx`.
- Leitura de `.dta` em caminhos com acento no Windows (cópia temporária se necessário).
- Etapas: 01–04.

## 2.0.0 — 2026-11-02
- Desfecho redefinido (MAJOR): sai o binário topo vs. resto (P80 internacional ≈ 3% de positivos).
  - Modelo A (principal): regressão do z da força relativo às normas (centil contínuo).
  - Modelo B (secundário): classificação ordinal do quartil internacional (<P25, P25–49, P50–74, ≥P75), Frank & Hall com pesos balanceados.
  - Quartis internos (cortes do z no treino) como sensibilidade (`sens_internal_quartiles`).
- `02_model.py`: CV aninhada (5×3, 30 combinações), algoritmo primário escolhido nas OOF do treino, ajuste final, validação na W3 com IC 95% bootstrap, camadas M1/M2 com o algoritmo primário, coerência A×B.
- Configuração de modelagem separada (`ModelConfig`): mudar a modelagem não invalida mais o 01.
- Modo rápido `SR_FAST=1` (grades mínimas; grava em `<etapa>_fast`).
- `03_explain.py`: permutação em bloco por domínio (todos os algoritmos M3, IC 95% bootstrap), concordância entre algoritmos, permutação por variável e SHAP do primário (A: z previsto; B: submodelo P(Q4)).
- Etapas: 01, 02 e 03.

## 1.0.0 — 2026-11-02
- Primeira versão de `sr_core.py` e `01_prepare.py`.
- Harmonização de 45 preditores em 11 domínios pré-especificados (camadas M1–M3), conforme a concordância
  `ELSI_concordancia_ondas_1-3.xlsx`.
- Desfecho: força máxima (mf27–mf29) em z/centil relativo às normas de Tomkinson et al. (2025), absolutas,
  interpoladas por idade; topo = centil ≥ 80.
- Exclusões, em ordem: idade < 50, entrevista por proxy (r1), demência/Parkinson (n62/n63), cirurgia de mão em
  3 meses (mf25), sem medidas físicas, recusa, incapaz/arriscado.
- Correções em relação aos scripts anteriores: IMC (mf22/altura, não mf1/mf2); altura em metros em W1;
  autoavaliação de saúde = n1 (não n2); câncer = n60 (n59 é depressão); AVC = n52 (n53 é idade);
  códigos 9555–9888 da força em W1; atividade física em W1 = l6/l8/l10 em minutos.
- Etapas: 01.

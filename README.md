# Detecção de Código Gerado por IA — TCC

Experimento comparativo que avalia a capacidade de modelos de linguagem grandes (LLMs) em classificar trechos de código como **escritos por humanos** ou **gerados por IA**, usando quatro estratégias de prompting diferentes.

---

## Objetivo

Investigar se LLMs conseguem distinguir código humano de código gerado por IA, e como diferentes estratégias de prompt afetam essa capacidade. O experimento cobre três linguagens de programação (Python, C, C++) e três provedores de IA distintos.

---

## Estrutura do Projeto

```
tcc/
├── main.py                        # Ponto de entrada — menu interativo
├── config.py                      # Parâmetros centralizados + experimento ativo
├── requirements.txt
├── .env                           # Chaves de API (não versionado)
│
├── src/                           # Todo o código-fonte
│   ├── experiment.py              # Orquestração do experimento (loop modelo × estratégia)
│   ├── data_loader.py             # Carregamento e filtragem do dataset
│   ├── providers.py               # Padrões Strategy + State: abstração e disponibilidade dos provedores
│   ├── metrics.py                 # Métricas completas + geração de gráficos
│   ├── evaluate.py                # Cálculo rápido de métricas por arquivo CSV
│   ├── compare.py                 # Tabela comparativa de desempenho entre modelos
│   └── logging_config.py          # Configuração centralizada do sistema de logging
│
├── experiments/
│   └── experiment1/               # Definição do experimento (entrada)
│       └── prompts/               # Templates de prompt usados neste experimento
│           ├── zero_shot.txt
│           ├── one_shot.txt
│           ├── few_shot.txt
│           └── cot.txt
│
├── results/
│   └── experiment1/               # CSVs gerados + métricas (criado em tempo de execução)
│       └── metrics/               # Gráficos, CSVs de resumo e relatório HTML
│
├── logs/
│   └── experiment1/               # Logs rotativos deste experimento (criado em tempo de execução)
│
├── backup/                        # Versão anterior do projeto (referência)
└── tests/
    ├── conftest.py
    ├── unit/
    └── integration/
```

O experimento ativo é controlado por `CURRENT_EXPERIMENT` em `config.py`. Os paths são derivados automaticamente:
- **Entrada** (prompts): `experiments/<experimento>/prompts/`
- **Saída** (results, metrics): `results/<experimento>/`
- **Logs**: `logs/<experimento>/`

---

## Organização por Experimentos

O projeto suporta múltiplos experimentos independentes. Cada experimento tem seus próprios prompts, resultados, métricas e logs.

### Experimento ativo

Em `config.py`:

```python
CURRENT_EXPERIMENT = "experiment1"
```

### Criar um novo experimento

1. Alterar `CURRENT_EXPERIMENT` em `config.py`:
   ```python
   CURRENT_EXPERIMENT = "experiment2"
   ```
2. Criar os prompts para o novo experimento:
   ```
   experiments/experiment2/prompts/zero_shot.txt
   experiments/experiment2/prompts/one_shot.txt
   experiments/experiment2/prompts/few_shot.txt
   experiments/experiment2/prompts/cot.txt
   ```
3. Rodar `python main.py` — results e metrics vão para `results/experiment2/`, logs para `logs/experiment2/`.

Os experimentos anteriores ficam intactos em suas próprias pastas.

---

## Como Executar

### 1. Instalar dependências

```bash
pip install -r requirements.txt
```

### 2. Configurar as chaves de API

Crie o arquivo `.env` na raiz do projeto:

```
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=AIza...
ANTHROPIC_API_KEY=sk-ant-...
```

> **Atenção**: use `GOOGLE_API_KEY` para o Gemini (não `GEMINI_API_KEY`).

### 3. Usar o menu principal

```bash
python main.py
```

```
============================================================
  TCC — Detecção de Código Gerado por IA
  Experimento atual: experiment1
============================================================

  [1] Rodar experimento
  [2] Calcular métricas
  [3] Gerar tabela comparativa de modelos
  [4] Rodar tudo  (1 → 2 → 3)
  [0] Sair
```

| Opção | O que faz |
|---|---|
| **1** | Executa o experimento — gera um CSV por combinação `(modelo, estratégia)` em `results/`. Combinações já concluídas são automaticamente puladas |
| **2** | Calcula métricas por modelo e estratégia, salva matrizes de confusão e gráfico de F1 em `metrics/` |
| **3** | Gera tabelas pivot, heatmaps por métrica, ranking geral dos modelos e relatório HTML comparativo |
| **4** | Executa as opções 1, 2 e 3 em sequência |

### Executar módulos diretamente (alternativa)

```bash
python -m src.experiment        # apenas o experimento
python -m src.metrics           # apenas as métricas
python -m src.compare           # apenas a tabela comparativa
```

---

## Dataset

**H-AIRosettaMP** — disponível no HuggingFace (`isThisYouLLM/H-AIRosettaMP`).

Contém trechos de código em diversas linguagens, rotulados como `human` (escrito por humano) ou `Ai_generated` (gerado por IA). O experimento filtra apenas **Python, C e C++** e amostra até 50 exemplos por classe por linguagem (configurável em `config.py`), totalizando até 300 amostras.

---

## Modelos Avaliados

| Identificador interno | API model ID | Modelo | Provedor |
|---|---|---|---|
| `gpt-4o-mini` | `gpt-4o-mini` | GPT-4o Mini | OpenAI |
| `claude-haiku` | `claude-haiku-4-5-20251001` | Claude Haiku 4.5 | Anthropic |
| `gemini-2.5-flash` | `gemini-2.5-flash` | Gemini 2.5 Flash | Google |

> O identificador interno é o nome usado dentro do código. O API model ID é a string enviada diretamente à API de cada provedor. A ordem em `MODELS` (`config.py`) define a sequência de execução — o Gemini fica por último por ter latência maior, preservando cota dos outros provedores em caso de interrupção.

---

## Estratégias de Prompting

Cada modelo é avaliado com quatro estratégias, definidas nos arquivos de `experiments/<experimento>/prompts/`:

### Zero-shot (`zero_shot.txt`)
O modelo recebe apenas o código e a instrução de classificação, sem nenhum exemplo. Testa o conhecimento intrínseco do modelo sobre padrões de código.

### One-shot (`one_shot.txt`)
Um único exemplo de referência (código AI com explicação) é fornecido antes do código a classificar. Ancora o modelo no formato de saída esperado.

### Few-shot (`few_shot.txt`)
Três exemplos fixos são fornecidos antes do código a classificar, cobrindo exatamente as linguagens do experimento: Python (AI), C (human) e C++ (AI). O contraste entre exemplos e linguagens ajuda o modelo a calibrar a decisão sem viés de linguagem.

### Chain-of-Thought — CoT (`cot.txt`)
O prompt instrui o modelo a raciocinar explicitamente sobre cinco dimensões antes de concluir: estilo de comentários, nomenclatura, estrutura, tratamento de erros e padrão geral. Técnica que melhora o desempenho em tarefas de raciocínio ao externalizar o processo de decisão.

### Formato de saída dos prompts

Todos os prompts exigem que o modelo responda **exclusivamente** com um objeto JSON:

```json
{
  "prediction": "human" | "ai",
  "confidence": 0-100,
  "explanation": "razão breve"
}
```

---

## Saídas do Pipeline

### Resultados brutos (`results/<experimento>/`)

Um arquivo CSV por combinação `(modelo, estratégia)`:

```
results/experiment1/gpt-4o-mini_zero_shot.csv
results/experiment1/gpt-4o-mini_one_shot.csv
...
```

Colunas: `id`, `language`, `true_label`, `model`, `strategy`, `prediction`, `confidence`, `explanation`.

### Métricas (`results/<experimento>/metrics/`)

| Arquivo | Conteúdo |
|---|---|
| `summary.csv` | Accuracy, F1, Precision, Recall por `(modelo, estratégia)` |
| `cm_<modelo>_<estratégia>.png` | Matriz de confusão individual |
| `f1_comparison.png` | Gráfico de barras F1 por estratégia e modelo |
| `compare_<métrica>.csv` | Tabela pivot: estratégias × modelos para cada métrica |
| `heatmap_<métrica>.png` | Heatmap visual da tabela pivot |
| `ranking_models.csv` | Ranking dos modelos por F1 médio entre estratégias |
| `comparison_report.html` | Relatório HTML completo com todas as tabelas comparativas |

---

## Métricas Calculadas

| Métrica | Descrição |
|---|---|
| **Accuracy** | Proporção de predições corretas sobre o total de amostras válidas |
| **Precision** | Dos classificados como `ai`, quantos eram realmente `ai` |
| **Recall** | Dos que eram realmente `ai`, quantos foram corretamente identificados |
| **F1-Score** | Média harmônica entre precision e recall; métrica principal de comparação |
| **Confusion Matrix** | Distribuição de verdadeiros/falsos positivos e negativos por modelo e estratégia |

A classe positiva usada no cálculo de precision, recall e F1 é `"ai"`. Predições `"unknown"` são excluídas do cálculo e contadas separadamente em `n_unknown`.

---

## Arquitetura de Software

### Padrão Strategy (`src/providers.py`)

O acesso aos provedores de IA é abstraído pelo **padrão de projeto Strategy** (GoF). Isso desacopla completamente a lógica do experimento dos detalhes de cada SDK.

```
AIProviderStrategy  (ABC)
       │
       ├── OpenAIStrategy    →  openai.OpenAI.chat.completions.create()
       ├── GeminiStrategy    →  google_genai.Client.models.generate_content()
       └── ClaudeStrategy    →  anthropic.Anthropic.messages.create()
```

A função `get_provider(model_name)` age como **factory**. Adicionar um novo provedor exige apenas criar uma subclasse de `AIProviderStrategy` e registrá-la em `_PROVIDER_MAP` — nenhuma outra parte do código muda.

### Padrão State (`src/providers.py`) — gerenciamento de disponibilidade

Cada provider é envolvido por um **`StatefulProvider`**, o *context* do **padrão de projeto State** (GoF). Mantém o estado atual de disponibilidade do modelo e delega cada chamada ao estado ativo.

```
ModelState  (ABC)
       │
       ├── AvailableState    →  chama a API normalmente
       ├── RateLimitedState  →  aguarda com backoff exponencial e retenta
       └── UnavailableState  →  falha rápido sem chamar a API
```

**Ciclo de transições:**

```
              chamada normal
AvailableState ─────────────────────────────────► AvailableState
       │                                                 ▲
       │ rate limit (HTTP 429)              recuperou    │
       ▼                                                 │
RateLimitedState ────────────────────────────────────────
  15s / 30s / 60s
       │
       │ max retries atingido  OU  3 falhas consecutivas
       ▼
UnavailableState  (terminal para o experimento em curso)
```

**Comportamento por estado:**

| Estado | O que acontece em `call()` |
|---|---|
| `AvailableState` | Executa a strategy normalmente. Erros de rede transientes → retenta até 5× com espera de 10 s. Rate limit → transiciona para `RateLimited`. 3 falhas consecutivas → `Unavailable`. |
| `RateLimitedState` | Aguarda 15 s, 30 s e 60 s (backoff exponencial) antes de retentar. Sucesso → volta para `Available`. Após 3 tentativas → `Unavailable`. |
| `UnavailableState` | Lança `RuntimeError` imediatamente, sem consumir cota de API. |

### Assinaturas de API verificadas

**OpenAI**
```python
client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}],
    temperature=0,
    max_tokens=300,
)
# Acesso ao texto: response.choices[0].message.content
```

**Google Gemini** — SDK: `google-genai` (`from google import genai`)
```python
client.models.generate_content(
    model="gemini-2.5-flash",
    contents=prompt,
)
# Acesso ao texto: response.text
```

**Anthropic Claude**
```python
client.messages.create(
    model="claude-haiku-4-5-20251001",
    max_tokens=300,
    messages=[{"role": "user", "content": prompt}],
)
# Acesso ao texto: response.content[0].text
```

### Sistema de Logging (`src/logging_config.py`)

Logs gravados em `logs/<experimento>/experiment_YYYYMMDD_HHMMSS.log` (rotativo, máx. 10 MB, 5 backups).

| Destino | Nível mínimo | Conteúdo |
|---|---|---|
| **Console** | `INFO` | Fluxo principal do experimento — progresso, timings, erros relevantes |
| **Arquivo rotativo** | `DEBUG` | Rastreamento completo — cada amostra, transições de estado, tempos de backoff, requests HTTP |

O console aplica `_ConsoleFilter`, que bloqueia mensagens INFO/DEBUG de bibliotecas externas (`httpx`, `httpcore`, `openai`, `google_genai`, `anthropic`, `huggingface_hub`). Essas bibliotecas logam cada request HTTP em nível INFO, o que poluiria o terminal; no arquivo de log ficam disponíveis para debug.

| Nível | Quando aparece | Exemplos |
|---|---|---|
| `DEBUG` | Só no arquivo | Amostra N processada, provider criado, transição de estado, HTTP 200 OK |
| `INFO` | Console + arquivo | Experimento iniciado/concluído, timing por combo, ETA, arquivos salvos |
| `WARNING` | Console + arquivo | Rate limit detectado, resposta não parseável, amostra pulada |
| `ERROR` | Console + arquivo | Falha de API para uma amostra (experimento continua) |
| `CRITICAL` | Console + arquivo | Provider marcado como indisponível |

### Fluxo de Execução (`src/experiment.py`)

```
main()
 ├─ para cada modelo × estratégia [X/N]:
 │    ├─ loga "[X/N] Iniciando: modelo | estratégia"
 │    ├─ run_experiment(df, model, strategy, combo_num, total_combos)
 │    │    ├─ load_prompt(strategy)                  # lê o .txt (cacheado com @lru_cache)
 │    │    └─ ThreadPoolExecutor(max_workers=MAX_WORKERS)
 │    │         └─ process_sample(i, row) — até 5 threads em paralelo
 │    │              ├─ _get_thread_provider(model)  # provider exclusivo por thread (threading.local)
 │    │              ├─ provider.is_available? → False: registra "provider_unavailable", pula
 │    │              ├─ build_prompt(template, code, MAX_CODE_CHARS)
 │    │              ├─ provider.call(prompt)         # State gerencia backoff e transições
 │    │              └─ parse_response(raw)
 │    └─ loga "[X/N] ✓ modelo | estratégia — Xmin Ys (Z.Zs/amostra) | restam N combos, ETA: ~Wmin"
 └─ loga tempo total do pipeline
```

O terminal exibe apenas o essencial: início e conclusão de cada combo com tempo decorrido, velocidade e ETA calculada pela média das combinações já concluídas. A barra de progresso (`tqdm`) atualiza a cada 8 segundos para não sobrecarregar o terminal. Resultados são salvos em `results/<experimento>/<modelo>_<estratégia>.csv`; se o arquivo já existe, a combinação é pulada.

### Testabilidade

| Ponto de injeção | Como testar |
|---|---|
| `OpenAIStrategy(client=mock)` | Passa um mock no lugar do client real — sem `OPENAI_API_KEY` |
| `GeminiStrategy(client=mock)` | Idem para o SDK do Google |
| `ClaudeStrategy(client=mock)` | Idem para o SDK da Anthropic |
| `get_provider(name, _registry={...})` | Substitui o registry por providers falsos |
| `_transform_df(df, languages, n)` | Função pura — sem I/O |
| `build_prompt(template, code, max_chars)` | Função pura — sem filesystem, sem API |
| `load_all_results(files=[...])` | Aceita lista de caminhos diretamente — sem depender de `results/` |
| `parse_response(text)` | Função pura — sem dependências externas |
| `_calc_metrics(y_true, y_pred)` | Função pura — sem dependências externas |

---

## Rate Limits por Modelo

Os limites abaixo foram confirmados pelos headers de resposta das APIs, mensagens de erro 429 nos logs e documentação oficial. O código os respeita de forma **proativa** via `_RateLimiter` em `src/providers.py` — throttling antes da chamada, não após o erro.

### GPT-4o-mini — OpenAI Tier 1

| Limite | Valor | Fonte |
|---|---|---|
| RPM (requests/min) | 10.000 | Header `x-ratelimit-limit-requests` nos logs |
| TPM (tokens/min) | 200.000 | Header `x-ratelimit-limit-tokens` nos logs |
| RPM configurado | 500 | Conservador — sem impacto na velocidade |

### Claude Haiku 4.5 — Anthropic Tier 1

| Limite | Valor | Fonte |
|---|---|---|
| RPM (requests/min) | 50 | Mensagem de erro 429 nos logs |
| ITPM (input tokens/min) | 50.000 | Documentação oficial Anthropic |
| OTPM (output tokens/min) | **10.000** | Mensagem de erro 429 nos logs (CoT) |
| RPM configurado | **30** | Limitante real: OTPM. Com CoT (~250 tok/resposta): 10.000 ÷ 250 = 40 req/min → 30 com margem |

> O limite de OTPM é o gargalo para a estratégia CoT, que gera respostas longas. Com `max_tokens=300` e média de ~250 tokens de saída, o limite de 50 RPM causaria ~12.500 OTPM — acima do limite. Usar 30 RPM garante ~7.500 OTPM, seguro em todas as estratégias.

### Gemini 2.5 Flash — Google (tier pago)

| Limite | Valor | Fonte |
|---|---|---|
| RPM (requests/min) | ≥ 1.000 | Ausência de 429 com 5 workers nos logs |
| RPM configurado | 500 | Conservador — sem impacto prático |

> O tier gratuito do Gemini tem apenas 10 RPM. Se a execução lançar 429, ajustar `MODEL_RPM["gemini-2.5-flash"]` para 8 em `config.py`.

### Como ajustar os limites

Todos os limites estão centralizados em `config.py`:

```python
MODEL_RPM = {
    "gpt-4o-mini":      500,   # real: 10 000 RPM
    "claude-haiku":      30,   # real: 50 RPM / 10 000 OTPM (CoT é o limitante)
    "gemini-2.5-flash": 500,   # real: ≥ 1 000 RPM (tier pago)
}

# Tokens de saída por estratégia
MAX_TOKENS_DEFAULT = 300   # zero_shot / one_shot / few_shot
MAX_TOKENS_COT     = 500   # CoT: cobre 5 dimensões + JSON sem truncar
```

O `_RateLimiter` em `src/providers.py` garante que cada thread aguarde o intervalo mínimo entre chamadas ao mesmo modelo, de forma thread-safe via `threading.Lock`. O valor de `max_tokens` é selecionado por estratégia em `run_experiment()` e propagado até a chamada de API de cada provider.

---

## Variáveis de Ambiente

```
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=AIza...
ANTHROPIC_API_KEY=sk-ant-...
```

---

## Tecnologias e Dependências

| Pacote | Uso |
|---|---|
| `anthropic` | SDK oficial da Anthropic para Claude |
| `openai` | SDK oficial da OpenAI para GPT |
| `google-genai` | SDK atual do Google para Gemini |
| `datasets` | Carregamento do dataset H-AIRosettaMP do HuggingFace |
| `pandas` | Manipulação de dados tabulares e leitura/escrita de CSVs |
| `scikit-learn` | Cálculo de métricas de classificação |
| `tqdm` | Barra de progresso no terminal |
| `python-dotenv` | Leitura das chaves de API a partir do `.env` |
| `matplotlib` / `seaborn` | Gráficos de comparação, heatmaps e matrizes de confusão |

---

## Melhorias Implementadas

### Bugs corrigidos

| Severidade | Problema | Localização | Solução |
|-----------|----------|-------------|---------|
| **Crítico** | `RateLimitedState.call()` se chamava recursivamente — em rate limits persistentes poderia causar stack overflow | `providers.py` | Convertido para loop `while` iterativo com `continue` |
| **Crítico** | `except Exception:` em `parse_response` descartava o erro original sem nenhum log | `experiment.py` | Adicionado `logger.debug` com o erro e o texto bruto que falhou |
| **Crítico** | `provider.call()` poderia retornar `None` ou string vazia, causando `TypeError` em cascata | `experiment.py` | Validação explícita antes de `parse_response`; resposta vazia registrada como `"empty_response"` |
| **Alto** | `logging.basicConfig()` no escopo de módulo causava reconfiguração do root logger ao importar | `metrics.py`, `evaluate.py` | Removido do escopo de módulo; `setup_logging()` chamado dentro de `main()` |
| **Alto** | Variável de ambiente `GEMINI_API_KEY` não correspondia ao nome real no `.env` (`GOOGLE_API_KEY`) | `providers.py` | Corrigido para `_require_env("GOOGLE_API_KEY")` |
| **Alto** | Erros de rede transientes (`getaddrinfo failed`, `WinError 10053/10054`, reset de conexão) contavam como falhas permanentes, levando o provider a `Unavailable` após 3 erros consecutivos | `src/providers.py` | Adicionada `_is_network_error()` e loop de retry em `AvailableState`: até 5× com espera de 10 s antes de escalar para falha consecutiva |
| **Médio** | `int(data.get("confidence", 50))` lançava `ValueError` se a API retornasse string não numérica | `experiment.py` | Protegido com `try/except`, retornando `50` como fallback |
| **Baixo** | `os.getenv()` retornava `None` silenciosamente; a API só falhava na primeira chamada | `providers.py` | Substituído por `_require_env()` com erro imediato e nome exato da variável |
| **Baixo** | Colunas do dataset não eram validadas; `KeyError` só aparecia amostras depois | `data_loader.py` | Verificação de colunas obrigatórias logo após `ds.to_pandas()` |

### Refatorações estruturais

| Melhoria | Descrição |
|---------|-----------|
| **Organização por experimentos** | Prompts, resultados, métricas e logs separados por experimento. Trocar de experimento = alterar uma linha em `config.py`. |
| **`src/` como pacote** | Todo o código-fonte em `src/`, separado de configuração e ponto de entrada. |
| **`main.py` com menu** | Ponto de entrada único com menu interativo, eliminando necessidade de chamar scripts individuais. |
| **`compare.py`** | Novo módulo que gera tabelas pivot, heatmaps por métrica, ranking de modelos e relatório HTML completo para comparação entre experimentos. |
| **`config.py` centralizado** | Constantes antes espalhadas em múltiplos arquivos consolidadas em um único lugar. |
| **`_calc_metrics()` como fonte única** | Cálculo de métricas extraído para uma única função, eliminando duplicação. |
| **`@lru_cache` em `load_prompt`** | Cada arquivo de prompt é lido do disco uma única vez por execução. |
| **Paralelismo com `ThreadPoolExecutor`** | Até `MAX_WORKERS` chamadas de API simultâneas. Cada thread mantém seu próprio `StatefulProvider` via `threading.local`. Ganho observado: ~4× no Gemini (~10 s → ~2,5 s/amostra com 5 workers). |
| **Feedback de progresso no terminal** | Cada combo exibe `[X/N] Iniciando` ao começar e `✓ modelo | estratégia — tempo (s/amostra) | ETA` ao concluir. A barra tqdm atualiza a cada 8 s (`mininterval=8`). |
| **Filtro de ruído no console (`_ConsoleFilter`)** | Bibliotecas externas (`httpx`, `openai`, `google_genai` etc.) logam cada request HTTP em nível INFO. `_ConsoleFilter` bloqueia essas mensagens no console, mantendo-as apenas no arquivo de log. |
| **Rate limiter proativo (`_RateLimiter`)** | `_RateLimiter` com `threading.Lock` em `src/providers.py` throttle chamadas antes de enviá-las, respeitando `MODEL_RPM` em `config.py`. Elimina rajadas de 429 causadas pelos 5 workers disparando simultaneamente. O limitante real do Claude Haiku é o OTPM (10 000/min) para CoT, não o RPM (50/min) — confirmado nos logs. |
| **`max_tokens` por estratégia** | CoT gera respostas mais longas (5 dimensões de análise); com `max_tokens=300` o JSON era truncado antes de fechar, gerando `parse_error`. Adicionados `MAX_TOKENS_DEFAULT=300` e `MAX_TOKENS_COT=500` em `config.py`, selecionados automaticamente em `run_experiment()` e propagados até a chamada de API. |
| **Ordem de execução dos modelos** | Modelos ordenados por latência crescente: `gpt-4o-mini` → `claude-haiku` → `gemini-2.5-flash`. O modelo mais lento fica por último para minimizar impacto em caso de interrupção. |
| **`results/<experimento>/` na raiz** | Resultados e métricas ficam em `results/<nome>/`, separados de `experiments/<nome>/` (que contém apenas prompts). Estrutura mais clara e consistente com `logs/<nome>/`. |
| **Injeção de dependência nas Strategies** | `client=None` permite substituir por mocks em testes sem variáveis de ambiente. |
| **`_transform_df` como função pura** | Lógica de filtragem e amostragem extraída de `load_data`, testável com DataFrames sintéticos. |

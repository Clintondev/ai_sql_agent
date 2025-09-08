# Agente de IA para Consulta Nutricional (DuckDB + Parquet + LLM)

## Visão Geral
- Objetivo: permitir perguntas em linguagem natural sobre alimentos (calorias, proteína, açúcar, sódio, categorias, volumes), respondidas via SQL gerado por IA e executado no DuckDB sobre um Parquet do Open Food Facts.
- Destaques:
  - Conversa em PT-BR com geração de SQL segura (apenas SELECT) e enriquecida por exemplos (few-shot).
  - View auxiliar `food_flat` com nutrientes por 100 g e por porção, unidades e normalizações de nome/marca/categoria, além de quantidade total em g/ml.
  - Resumo automático (NLG) dos resultados: média/mín/máx e exemplos TOP.

## Stack Tecnológica
- Linguagem: Python 3.12
- Engine de consulta: DuckDB (in-memory) lendo Parquet
- Dados: Open Food Facts Parquet (`food.parquet`)
- LLM: Google Gemini (via `google.generativeai`), few-shot + regras de segurança
- UI: CLI com `rich` (opcional), pandas para exibição/tabulação

## Arquitetura (Alto Nível)
```
flowchart LR
  User[Usuário CLI] -->|Pergunta PT-BR| Agent[SQLAgent]
  Agent -->|Prompt c/ schema + few-shot| LLM[Gemini]
  LLM -->|SQL (SELECT)| Agent
  Agent -->|Execução| DuckDB[(DuckDB in-memory)]
  DuckDB -->|View food & food_flat| Parquet[(food.parquet)]
  DuckDB -->|DataFrame| Agent
  Agent -->|Resultado + NLG| User
```

## Fluxo Detalhado
```
sequenceDiagram
  participant U as Usuário
  participant A as SQLAgent (Python)
  participant G as Gemini
  participant D as DuckDB
  U->>A: Pergunta: "quanta proteína tem um iogurte itambé por 100g?"
  A->>G: Prompt (esquema, regras, exemplos)
  G-->>A: SQL seguro (SELECT)
  A->>D: Executa SQL (food / food_flat)
  D-->>A: DataFrame (pandas)
  A->>A: NLG (média/mín/máx, top exemplos)
  A-->>U: Tabela + Resumo
```

## Modelo de Dados (resumo)
- Tabela `food` (do Parquet): ~110 colunas. Destaques:
  - `product_name` (LIST<STRUCT(lang, text)>)
  - `brands`, `categories`
  - `nutriments` (LIST<STRUCT(name, value, "100g", serving, unit, ...)>): energy-kcal, proteins, fat, saturated-fat, sugars, carbohydrates, fiber, salt, sodium, vitaminas etc.
  - `product_quantity`, `product_quantity_unit`, `quantity`, `serving_quantity`, `serving_size`
- View `food_flat` (derivada):
  - Identificação: `code`, `name`, `brands`, `categories`
  - Normalizações (para filtro robusto): `name_norm`, `brands_norm`, `categories_norm`
  - Quantidades: `quantity_g`, `quantity_ml`
  - Nutrientes por 100 g: `energy_kcal_100g`, `proteins_100g`, `fat_100g`, `saturated_fat_100g`, `carbs_100g`, `sugars_100g`, `fiber_100g`, `salt_100g`, `sodium_100g`
  - Por porção e unidades: `energy_kcal_serving`, `energy_kcal_unit`, `proteins_serving`, `proteins_unit`

## Engenharia de Prompt (segurança e precisão)
- Somente `SELECT` (bloqueio de DDL/DML e múltiplas instruções).
- Remoção de cercas e pós-processamento (sanitização).
- Few-shot de consultas típicas (proteína/kcal, Coca-Cola por volume, etc.).
- Regras específicas:
  - Preferir `food_flat` para métricas nutricionais.
  - Projetar sempre `name`, `brands` e unidades relevantes.
  - Filtrar `IS NOT NULL` em métricas e usar colunas normalizadas para LIKE.
  - Acesso a `nutriments` via `UNNEST` quando necessário.
  - Heurística para "N unidades": usar mediana por porção quando existir; fallback por 100 g com pesos típicos (ex.: ovo 50 g).

## Capacidades
- Perguntas em PT-BR sobre:
  - Proteína / calorias / açúcar / gordura / saturada / carboidratos / fibra por 100 g ou por porção.
  - Sódio/sal, colesterol, vitaminas e minerais mais comuns (C, A, D, cálcio, ferro, potássio…).
  - Categorias e identificação de produtos por nome/marca.
  - Quantidade total de embalagem (g/ml), com filtros (≥ 1 L, ≥ 1 kg, etc.).
  - Score/qualidade: NOVA, NutriScore, nutrition-score-fr.
  - Resumo automático (NLG) com estatísticas simples.

## Execução e Uso
- Modo IA (Gemini):
  - `export GOOGLE_API_KEY='...'`
  - `python sql_agent.py`
  - Perguntas livres em PT-BR (ex.: "quanta proteína tem um iogurte itambé por 100g?")
- Modo manual (sem IA):
  - `python sql_agent.py --no-llm`
  - Comandos: `/schema`, `/sample 5`, `/sql <consulta>`, `/format pretty|json|csv`, `/nlg on|off`

## Exemplos de Perguntas (dieta)
- "iogurtes com menos de 5 g de açúcar por 100g"
- "coca-cola com volume a partir de 1 litro, com categorias"
- "pães com mais fibra por 100g"
- "top 10 barras por proteína/kcal"
- "quanto de proteína tem em 3 ovos caipiras" (usa mediana por porção; fallback ovo=50 g)

## Segurança e Privacidade
- SQL estritamente `SELECT` com filtros anti-injeção simples.
- Execução local; sem envio de dados do Parquet ao LLM (somente esquema e amostras).

## Performance
- Leitura direta do Parquet com `read_parquet` (DuckDB in-memory).
- Sugestão: materialização opcional em `.duckdb` para latência menor e caching de `food_flat`.

## Limitações e Próximos Passos
- Parser determinístico de intenções (unidades/quantidades) no código, com tabela de pesos/porções por categoria.
- Densidades específicas para conversão ml↔g por categoria de bebida.
- Deduplicação por `code`/marca ao resumir.
- API/serviço (FastAPI) e UI web.

## Estrutura de Arquivos
- `sql_agent.py`: agente, LLM, CLI e views.
- `docs/SHOWCASE.md`: este documento (visão geral, arquitetura e guia).


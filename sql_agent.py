
import os
import re
import sys
import time
import textwrap
from typing import List, Optional, Tuple

import duckdb
import pandas as pd

# Integração opcional com Google Gemini
try:
    import google.generativeai as genai
except Exception:  # Mantém import opcional (permite rodar /schema, /sql direto)
    genai = None

# Impressão rica opcional
try:
    from rich.console import Console
    from rich.table import Table
    from rich.box import SIMPLE_HEAVY
except Exception:
    Console = None
    Table = None
    SIMPLE_HEAVY = None


# ==========================
# Impressão / Visualização
# ==========================

class UIPrinter:
    def __init__(self, mode: str = "auto", max_colwidth: int = 100, no_color: bool = False):
        self.no_color = no_color
        self.max_colwidth = max_colwidth
        if mode == "auto":
            self.mode = "pretty" if Console is not None and not no_color else "plain"
        else:
            self.mode = mode
        self.console = Console(no_color=True) if (Console and no_color) else (Console() if Console else None)

    def set_mode(self, mode: str):
        if mode == "auto":
            self.mode = "pretty" if Console is not None and not self.no_color else "plain"
        else:
            self.mode = mode

    def _clip(self, s: str) -> str:
        if s is None:
            return ""
        s = str(s)
        if len(s) > self.max_colwidth:
            return s[: self.max_colwidth - 1] + "…"
        return s

    def _format_scalar(self, v):
        if isinstance(v, (int,)):
            return f"{v:,}".replace(",", ".")  # milhar com ponto
        return str(v)

    def print_df(self, df: pd.DataFrame, title: Optional[str] = None):
        if df is None:
            print("(Sem dados)")
            return
        if df.empty:
            print("(Nenhum resultado)")
            return

        # Caso 1x1: scalar
        if df.shape == (1, 1):
            val = df.iloc[0, 0]
            label = df.columns[0]
            print(f"{label}: {self._format_scalar(val)}")
            return

        if self.mode == "pretty" and self.console and Table:
            table = Table(title=title, box=SIMPLE_HEAVY, show_lines=False, header_style="bold")
            for col in df.columns:
                table.add_column(str(col))
            for _, row in df.iterrows():
                table.add_row(*[self._clip(x) for x in row.tolist()])
            self.console.print(table)
        elif self.mode == "csv":
            print(df.to_csv(index=False))
        elif self.mode == "json":
            print(df.to_json(orient="records", force_ascii=False))
        else:  # plain
            with pd.option_context("display.max_colwidth", self.max_colwidth, "display.max_rows", 200):
                print(df.to_string(index=False))

    def print_info(self, msg: str):
        if self.console and self.mode == "pretty" and not self.no_color:
            self.console.print(f"[bold cyan]{msg}[/bold cyan]")
        else:
            print(msg)

    def print_warn(self, msg: str):
        if self.console and self.mode == "pretty" and not self.no_color:
            self.console.print(f"[bold yellow]{msg}[/bold yellow]")
        else:
            print(msg)

    def print_err(self, msg: str):
        if self.console and self.mode == "pretty" and not self.no_color:
            self.console.print(f"[bold red]{msg}[/bold red]")
        else:
            print(msg)


# ==========================
# Utilidades
# ==========================

def _clean_sql(sql: str) -> str:
    """Normaliza saída de LLM removendo cercas e texto extra."""
    if not sql:
        return ""
    s = sql.strip()
    # Remover cercas ```sql ... ```
    if s.lower().startswith("```sql"):
        s = s[6:].strip()
    if s.startswith("```"):
        s = s[3:].strip()
    if s.endswith("```"):
        s = s[:-3].strip()
    # Manter apenas a partir de SELECT
    m = re.search(r"\bselect\b", s, flags=re.I)
    if m:
        s = s[m.start():].strip()
    # Remover ponto-e-vírgula final
    s = s.rstrip("; ")
    return s


def _ensure_select_only(sql: str) -> bool:
    """Retorna True se for uma única instrução SELECT segura."""
    if not sql:
        return False
    # Bloqueia múltiplas instruções e DDL/DML perigosos
    banned = [
        r"\b(drop|alter|truncate|insert|update|delete|create|attach|copy)\b",
    ]
    # Não permitir múltiplos ';'
    if sql.count(";") > 0:
        return False
    if not re.match(r"^\s*select\b", sql, flags=re.I):
        return False
    for pat in banned:
        if re.search(pat, sql, flags=re.I):
            return False
    return True


def _add_limit_if_missing(sql: str, default_limit: int = 50) -> str:
    """Garante LIMIT padrão quando não houver LIMIT/ TOP / SAMPLE."""
    if re.search(r"\blimit\s+\d+\b", sql, flags=re.I):
        return sql
    if re.search(r"\b(sample|using\s+sample)\b", sql, flags=re.I):
        return sql
    return f"{sql}\nLIMIT {default_limit}"


# ==========================
# Cliente de LLM (Gemini)
# ==========================

class GeminiClient:
    def __init__(self, model_name: Optional[str] = None):
        if genai is None:
            raise RuntimeError("Biblioteca google.generativeai não está instalada.")

        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "A variável de ambiente GOOGLE_API_KEY não foi definida.\n"
                "Defina com: export GOOGLE_API_KEY='SUA_CHAVE'"
            )
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name or os.environ.get("GEMINI_MODEL", "gemini-1.5-flash"))

    def generate_sql(self, question: str, table_name: str, schema_prompt: str, examples: Optional[List[str]] = None) -> str:
        examples = examples or []
        prompt = f"""
        Você é um especialista em SQL no dialeto DuckDB. Seu objetivo é gerar UMA única consulta SELECT válida e concisa para responder a pergunta.

        Contexto da base:
        - Nome da tabela principal: {table_name}
        - View auxiliar disponível: {table_name}_flat (colunas amigáveis para nutrientes)
        - Esquema e amostras:\n{schema_prompt}

        Regras gerais:
        - Use SOMENTE as tabelas {table_name} ou {table_name}_flat (ou subqueries sobre elas).
        - Gere apenas uma instrução SELECT. Não use DDL/DML.
        - Evite SELECT *. Selecione colunas relevantes.
        - Use funções do DuckDB. Prefira expressões simples e legíveis.
        - Ao "listar produtos" ou responder itens, projete SEMPRE colunas amigáveis: name (ou product_name[1].text), brands e, quando fizer sentido, categories e product_quantity.
        - Para perguntas de nutrientes/quantidades (proteína, calorias, etc.), PREFIRA a view {table_name}_flat. Projete name, brands, a(s) métrica(s) pedidas e também as unidades correspondentes (por ex.: proteins_unit, energy_kcal_unit). Aplique filtros IS NOT NULL nas métricas para remover vazios.
        - Quando o usuário mencionar uma quantidade de unidades (ex.: "3 ovos"), NÃO some valores de vários produtos. Em vez disso, estime a quantidade por unidade usando uma medida representativa (mediana via quantile_cont) dos itens equivalentes e MULTIPLIQUE pelo número de unidades solicitado. Se disponível, use *_serving (por porção). Caso contrário, use *_100g e assuma um peso típico por unidade (ex.: 50 g para um ovo).
        - Ao filtrar por termo no nome, considere ambos: list_contains(list_transform(product_name, p -> lower(p.text)), lower('<termo>')) OU lower(categories) LIKE '%<termo>%'. Nas views, prefira colunas normalizadas name_norm, brands_norm, categories_norm com LIKE '%termo%'.
        - product_name é LIST<STRUCT>; para projetar use product_name[1].text; para filtrar use list_contains(list_transform(product_name, p -> lower(p.text)), '<termo>').
        - Para nutrientes detalhados na tabela original use UNNEST(nutriments) AS n e depois n.unnest."100g" (por 100g), n.unnest.serving (por porção), n.unnest.unit; nomes comuns: energy-kcal, proteins, fat, saturated-fat, sugars, carbohydrates, fiber, salt, sodium.
        - Alternativamente, a view {table_name}_flat já expõe colunas: energy_kcal_100g, proteins_100g, fat_100g, saturated_fat_100g, carbs_100g, sugars_100g, fiber_100g, salt_100g, sodium_100g, além de *_serving e *_unit quando aplicável, e quantity_g/quantity_ml; e ainda name_norm, brands_norm, categories_norm para filtragem robusta.

        Exemplos de consultas válidas (DuckDB):
        {"\n".join(examples)}

        Pergunta do usuário (PT-BR): {question}

        Responda SOMENTE com a consulta SQL (sem explicações).
        """
        resp = self.model.generate_content(prompt)
        return _clean_sql(resp.text)

    def repair_sql(self, failed_sql: str, error_msg: str, question: str, table_name: str, schema_prompt: str) -> str:
        prompt = f"""
        A consulta DuckDB abaixo falhou. Corrija-a gerando UMA nova consulta SELECT válida.

        Pergunta original: {question}
        Tabela: {table_name}
        View auxiliar: {table_name}_flat
        Esquema e amostras:\n{schema_prompt}

        Consulta que falhou:
        {failed_sql}

        Erro ao executar:
        {error_msg}

        Regras:
        - Produza apenas uma instrução SELECT válida no DuckDB.
        - Evite SELECT * e use colunas existentes.
        - Seja conciso. Se necessário, ajuste filtros ou projeções.
        - Prefira projetar name e brands com a(s) métrica(s) e unidade(s) relevantes, usando IS NOT NULL para filtrar vazios.
        - Dica: para nutrientes use UNNEST(nutriments) ou utilize a view {table_name}_flat (energy_kcal_100g, proteins_100g, etc.).
        """
        resp = self.model.generate_content(prompt)
        return _clean_sql(resp.text)


# ==========================
# SQL Agent (DuckDB)
# ==========================

class SQLAgent:
    def __init__(
        self,
        parquet_path: str,
        table_name: str = "food",
        default_limit: int = 50,
        model_client: Optional[GeminiClient] = None,
        do_schema_samples: bool = True,
        schema_samples_per_col: int = 2,
        schema_max_cols: int = 30,
        verbose: bool = False,
        enrich: bool = True,
    ) -> None:
        self.parquet_path = parquet_path
        self.table_name = table_name
        self.default_limit = default_limit
        self.model_client = model_client
        self.do_schema_samples = do_schema_samples
        self.schema_samples_per_col = max(0, schema_samples_per_col)
        self.schema_max_cols = max(1, schema_max_cols)
        self.verbose = verbose
        self.con = duckdb.connect(":memory:")
        self._attach_parquet_view()
        self.schema_prompt = self._build_schema_prompt()
        self.enrich = enrich

    # ---------- Conexão / Schema ----------
    def _attach_parquet_view(self) -> None:
        if not os.path.exists(self.parquet_path):
            raise FileNotFoundError(
                f"Arquivo parquet não encontrado: {self.parquet_path}"
            )
        try:
            self.con.execute(
                f"CREATE OR REPLACE VIEW {self.table_name} AS SELECT * FROM read_parquet('{self.parquet_path}')"
            )
            # View auxiliar achatada com nutrientes por 100g
            self.con.execute(
                f"""
                CREATE OR REPLACE VIEW {self.table_name}_flat AS
                WITH base AS (
                  SELECT
                    code,
                    brands,
                    /* versões normalizadas (minúsculas e sem acentos) para buscas */
                    lower(brands) AS brands_lc,
                    categories,
                    lower(categories) AS categories_lc,
                    product_quantity,
                    product_quantity_unit,
                    quantity,
                    serving_quantity,
                    serving_size,
                    COALESCE({self.table_name}.product_name[1].text, NULL) AS name,
                    lower(COALESCE({self.table_name}.product_name[1].text, '')) AS name_lc,
                    nutriments
                  FROM {self.table_name}
                ),
                qty AS (
                  SELECT
                    code,
                    brands,
                    brands_lc,
                    categories,
                    categories_lc,
                    product_quantity,
                    product_quantity_unit,
                    quantity,
                    serving_quantity,
                    serving_size,
                    name,
                    name_lc,
                    nutriments,
                    /* normalização de quantidade total em g e ml (combinando product_quantity(+_unit) e quantity) */
                    lower(COALESCE(product_quantity_unit, '')) AS pq_unit,
                    lower(COALESCE(quantity, '')) AS q_text,
                    /* padrões em quantity (texto livre): "N x Q UNIT" e "Q UNIT" */
                    regexp_extract(lower(COALESCE(quantity, '')), '(\\d+)\\s*[x×]\\s*([0-9]+(?:[\\.,][0-9]+)?)\\s*(kg|g|l|ml)', 1) AS q_pack_n,
                    regexp_extract(lower(COALESCE(quantity, '')), '(\\d+)\\s*[x×]\\s*([0-9]+(?:[\\.,][0-9]+)?)\\s*(kg|g|l|ml)', 2) AS q_pack_num,
                    regexp_extract(lower(COALESCE(quantity, '')), '(\\d+)\\s*[x×]\\s*([0-9]+(?:[\\.,][0-9]+)?)\\s*(kg|g|l|ml)', 3) AS q_pack_unit,
                    regexp_extract(lower(COALESCE(quantity, '')), '([0-9]+(?:[\\.,][0-9]+)?)\\s*(kg|g|l|ml)', 1) AS q_simple_num,
                    regexp_extract(lower(COALESCE(quantity, '')), '([0-9]+(?:[\\.,][0-9]+)?)\\s*(kg|g|l|ml)', 2) AS q_simple_unit
                  FROM base
                ),
                qty2 AS (
                  SELECT
                    code,
                    brands,
                    brands_lc,
                    categories,
                    categories_lc,
                    product_quantity,
                    product_quantity_unit,
                    quantity,
                    serving_quantity,
                    serving_size,
                    name,
                    name_lc,
                    nutriments,
                    /* product_quantity (+ unit) */
                    TRY_CAST(replace(COALESCE(product_quantity, ''), ',', '.') AS DOUBLE) AS pq_num_d,
                    pq_unit,
                    /* quantity (texto) parseado */
                    TRY_CAST(replace(COALESCE(q_pack_n, ''), ',', '.') AS DOUBLE)      AS pack_n_d,
                    TRY_CAST(replace(COALESCE(q_pack_num, ''), ',', '.') AS DOUBLE)    AS pack_num_d,
                    COALESCE(q_pack_unit, '')                                          AS pack_u,
                    TRY_CAST(replace(COALESCE(q_simple_num, ''), ',', '.') AS DOUBLE)  AS simple_num_d,
                    COALESCE(q_simple_unit, '')                                        AS simple_u
                  FROM qty
                ),
                qty3 AS (
                  SELECT
                    code,
                    brands,
                    brands_lc,
                    categories,
                    categories_lc,
                    product_quantity,
                    product_quantity_unit,
                    quantity,
                    serving_quantity,
                    serving_size,
                    name,
                    name_lc,
                    nutriments,
                    /* total em g */
                    CASE
                      WHEN pq_unit IN ('kg','g') AND pq_num_d IS NOT NULL THEN pq_num_d * (CASE WHEN pq_unit='kg' THEN 1000 ELSE 1 END)
                      WHEN pack_u IN ('kg','g') THEN
                        (CASE WHEN pack_n_d IS NOT NULL AND pack_num_d IS NOT NULL THEN pack_n_d * pack_num_d * (CASE WHEN pack_u='kg' THEN 1000 ELSE 1 END) END)
                      WHEN simple_u IN ('kg','g') THEN
                        (CASE WHEN simple_num_d IS NOT NULL THEN simple_num_d * (CASE WHEN simple_u='kg' THEN 1000 ELSE 1 END) END)
                    END AS quantity_g,
                    /* total em ml */
                    CASE
                      WHEN pq_unit IN ('l','ml') AND pq_num_d IS NOT NULL THEN pq_num_d * (CASE WHEN pq_unit='l' THEN 1000 ELSE 1 END)
                      WHEN pack_u IN ('l','ml') THEN
                        (CASE WHEN pack_n_d IS NOT NULL AND pack_num_d IS NOT NULL THEN pack_n_d * pack_num_d * (CASE WHEN pack_u='l' THEN 1000 ELSE 1 END) END)
                      WHEN simple_u IN ('l','ml') THEN
                        (CASE WHEN simple_num_d IS NOT NULL THEN simple_num_d * (CASE WHEN simple_u='l' THEN 1000 ELSE 1 END) END)
                    END AS quantity_ml
                  FROM qty2
                )
                SELECT
                  q.code,
                  q.brands,
                  q.brands_lc AS brands_norm,
                  q.categories,
                  q.categories_lc AS categories_norm,
                  q.product_quantity,
                  q.product_quantity_unit,
                  q.quantity,
                  q.serving_quantity,
                  q.serving_size,
                  q.name,
                  q.name_lc AS name_norm,
                  q.quantity_g,
                  q.quantity_ml,
                  MAX(CASE WHEN lower(n.unnest.name)='energy-kcal' THEN n.unnest."100g" END)      AS energy_kcal_100g,
                  MAX(CASE WHEN lower(n.unnest.name)='energy-kcal' THEN n.unnest.serving END)     AS energy_kcal_serving,
                  MAX(CASE WHEN lower(n.unnest.name)='energy-kcal' THEN n.unnest.unit END)        AS energy_kcal_unit,
                  MAX(CASE WHEN lower(n.unnest.name)='proteins' THEN n.unnest."100g" END)         AS proteins_100g,
                  MAX(CASE WHEN lower(n.unnest.name)='proteins' THEN n.unnest.serving END)        AS proteins_serving,
                  MAX(CASE WHEN lower(n.unnest.name)='proteins' THEN n.unnest.unit END)           AS proteins_unit,
                  MAX(CASE WHEN lower(n.unnest.name)='fat' THEN n.unnest."100g" END)              AS fat_100g,
                  MAX(CASE WHEN lower(n.unnest.name)='saturated-fat' THEN n.unnest."100g" END)    AS saturated_fat_100g,
                  MAX(CASE WHEN lower(n.unnest.name)='carbohydrates' THEN n.unnest."100g" END)    AS carbs_100g,
                  MAX(CASE WHEN lower(n.unnest.name)='sugars' THEN n.unnest."100g" END)           AS sugars_100g,
                  MAX(CASE WHEN lower(n.unnest.name)='fiber' THEN n.unnest."100g" END)            AS fiber_100g,
                  MAX(CASE WHEN lower(n.unnest.name)='salt' THEN n.unnest."100g" END)             AS salt_100g,
                  MAX(CASE WHEN lower(n.unnest.name)='sodium' THEN n.unnest."100g" END)           AS sodium_100g
                FROM qty3 q, UNNEST(nutriments) AS n
                GROUP BY
                  q.code,
                  q.brands,
                  q.brands_lc,
                  q.categories,
                  q.categories_lc,
                  q.product_quantity,
                  q.product_quantity_unit,
                  q.quantity,
                  q.serving_quantity,
                  q.serving_size,
                  q.name,
                  q.name_lc,
                  q.quantity_g,
                  q.quantity_ml
                """
            )
        except Exception as e:
            raise RuntimeError(f"Falha ao criar VIEW sobre o parquet: {e}")

    def _get_schema_df(self) -> pd.DataFrame:
        return self.con.execute(f"DESCRIBE {self.table_name}").fetchdf()

    def _sample_values_for_column(self, col: str, n: int = 3) -> List[str]:
        try:
            df = self.con.execute(
                f"SELECT {col} AS v FROM {self.table_name} WHERE {col} IS NOT NULL LIMIT {n}"
            ).fetchdf()
            vals = []
            for v in df["v"].tolist():
                # stringify com limite
                s = str(v)
                if len(s) > 120:
                    s = s[:117] + "..."
                vals.append(s)
            return vals
        except Exception:
            return []

    def _build_schema_prompt(self) -> str:
        if self.verbose:
            print("[info] Lendo DESCRIBE da tabela…", flush=True)
        df = self._get_schema_df()
        lines = [f"Colunas da tabela {self.table_name} (máx {self.schema_max_cols}):"]
        cols_full = df[["column_name", "column_type"]].values.tolist()
        cols = cols_full[: self.schema_max_cols]
        self._columns = [c for c, _ in cols]

        sampled = 0
        for c, t in cols:
            t_low = str(t).lower()
            can_sample = (
                self.do_schema_samples
                and sampled < self.schema_max_cols
                and self.schema_samples_per_col > 0
                and not any(x in t_low for x in ["list", "struct", "map"])
            )
            if can_sample:
                if self.verbose and sampled == 0:
                    print("[info] Coletando amostras de colunas simples…", flush=True)
                samples = self._sample_values_for_column(c, self.schema_samples_per_col)
                sampled += 1
            else:
                samples = []
            samp_str = ", ".join(samples) if samples else "(sem amostras)"
            lines.append(f"- {c} :: {t} | amostras: {samp_str}")

        # Regras específicas (apenas se colunas existirem)
        extra_rules = []
        if "product_name" in self._columns:
            extra_rules.append(
                "- product_name é LIST<STRUCT>. Para projetar: product_name[1].text. Para filtrar: list_contains(list_transform(product_name, p -> p.text), 'termo')."
            )
        if "product_quantity" in self._columns:
            extra_rules.append(
                "- product_quantity é VARCHAR possivelmente com vírgulas; para número use CAST(REPLACE(product_quantity, ',', '.') AS REAL)."
            )
        if "nutriments" in self._columns:
            extra_rules.append(
                "- nutriments é LIST<STRUCT>. Para acessar valores: use UNNEST(nutriments) AS n e projete n.unnest.'100g' (por 100g), n.unnest.serving (por porção), n.unnest.unit."
            )
            extra_rules.append(
                "- Nomes comuns em nutriments: energy-kcal, proteins, fat, saturated-fat, sugars, carbohydrates, fiber, salt, sodium, energy-kj."
            )
        # Anunciar view auxiliar
        extra_rules.append(
            f"- View auxiliar disponível: {self.table_name}_flat com colunas: code, name, brands, categories, product_quantity, quantity_g, quantity_ml, energy_kcal_100g, energy_kcal_serving, energy_kcal_unit, proteins_100g, proteins_serving, proteins_unit, fat_100g, saturated_fat_100g, carbs_100g, sugars_100g, fiber_100g, salt_100g, sodium_100g."
        )
        if extra_rules:
            lines.append("Regras específicas:")
            lines.extend(extra_rules)

        return "\n".join(lines)

    # ---------- Execução ----------
    def run_sql(self, sql: str) -> pd.DataFrame:
        return self.con.execute(sql).fetchdf()

    def ask(self, question: str, retries: int = 2) -> Tuple[str, Optional[pd.DataFrame], Optional[str]]:
        if not self.model_client:
            return "", None, "Nenhum modelo LLM configurado. Use /sql para executar consultas manuais."

        # Exemplos de few-shot para guiar o modelo
        examples = [
            # Proteína por 100g usando view flat (com unidade e IS NOT NULL)
            f"SELECT name, brands, proteins_100g, proteins_unit FROM {self.table_name}_flat WHERE (brands_norm LIKE '%itambe%' OR name_norm LIKE '%itambe%') AND name_norm LIKE '%iogurte%' AND proteins_100g IS NOT NULL ORDER BY proteins_100g DESC NULLS LAST LIMIT 20",
            # Calorias por 100g para iogurtes (filtrando nulos e ordenando)
            f"SELECT name, brands, energy_kcal_100g, energy_kcal_unit FROM {self.table_name}_flat WHERE name_norm LIKE '%iogurte%' AND energy_kcal_100g IS NOT NULL ORDER BY energy_kcal_100g DESC NULLS LAST LIMIT 20",
            # Categorias para Coca-Cola usando tabela original (nome e categorias)
            f"SELECT product_name[1].text AS name, brands, categories FROM {self.table_name} WHERE list_contains(list_transform(product_name, p -> lower(p.text)), 'coca-cola') OR brands ILIKE '%coca%' LIMIT 20",
            # Acesso direto a nutriments via UNNEST (com nome/marca)
            f"SELECT COALESCE(product_name[1].text, NULL) AS name, brands, n.unnest.'100g' AS proteins_100g, n.unnest.unit AS proteins_unit FROM {self.table_name}, UNNEST(nutriments) AS n WHERE lower(n.unnest.name) = 'proteins' AND lower(COALESCE(product_name[1].text, '')) LIKE '%iogurte%' AND n.unnest.'100g' IS NOT NULL LIMIT 20",
            # Filtro por unidade/quantidade: produtos com total > 1 kg
            f"SELECT name, brands, quantity_g FROM {self.table_name}_flat WHERE quantity_g IS NOT NULL AND quantity_g > 1000 ORDER BY quantity_g DESC LIMIT 20",
            # Filtro por volume: latas/garrafas acima de 1 litro
            f"SELECT name, brands, quantity_ml FROM {self.table_name}_flat WHERE quantity_ml IS NOT NULL AND quantity_ml >= 1000 ORDER BY quantity_ml DESC LIMIT 20",
            # Estimativa por unidades: 3 ovos caipiras (usar mediana por porção)
            f"SELECT 3 AS unidades, round(quantile_cont(proteins_serving, 0.5), 2) AS proteina_por_unidade_g, 'g' AS unit, round(3 * quantile_cont(proteins_serving, 0.5), 2) AS proteina_total_g FROM {self.table_name}_flat WHERE name_norm LIKE '%ovo%' AND proteins_serving IS NOT NULL",
            # Queda para 100g caso não haja porção: assumir 50 g por ovo
            f"SELECT 3 AS unidades, round(0.5 * quantile_cont(proteins_100g, 0.5), 2) AS proteina_por_unidade_g, 'g' AS unit, round(3 * 0.5 * quantile_cont(proteins_100g, 0.5), 2) AS proteina_total_g FROM {self.table_name}_flat WHERE name_norm LIKE '%ovo%' AND proteins_100g IS NOT NULL",
        ]

        sql = self.model_client.generate_sql(question, self.table_name, self.schema_prompt, examples=examples)
        if not _ensure_select_only(sql):
            # Tentar reparar forçando SELECT
            sql = f"SELECT * FROM {self.table_name} LIMIT {self.default_limit}"
        sql = _add_limit_if_missing(sql, self.default_limit)

        last_err = None
        for _ in range(1 + retries):
            try:
                df = self.run_sql(sql)
                # Enriquecimento automático (se só veio 'code')
                if self.enrich and df is not None and not df.empty:
                    sql, df = self._maybe_enrich(sql, df)
                return sql, df, None
            except Exception as e:
                last_err = str(e)
                if self.model_client and retries > 0:
                    sql = self.model_client.repair_sql(sql, last_err, question, self.table_name, self.schema_prompt)
                    sql = _clean_sql(sql)
                    if not _ensure_select_only(sql):
                        sql = _add_limit_if_missing(f"SELECT * FROM {self.table_name}", self.default_limit)
                    sql = _add_limit_if_missing(sql, self.default_limit)
                    retries -= 1
                else:
                    break
        return sql, None, last_err or "Falha ao executar a consulta."

    def _extract_limit(self, sql: str) -> Optional[int]:
        m = re.search(r"\blimit\s+(\d+)\b", sql, flags=re.I)
        if not m:
            return None
        try:
            return int(m.group(1))
        except Exception:
            return None

    def _maybe_enrich(self, sql: str, df: pd.DataFrame) -> Tuple[str, pd.DataFrame]:
        cols = [c.lower() for c in df.columns]
        has_code = "code" in cols
        has_name = any(c in cols for c in ["product", "product_name", "nome", "produto"])  # heurística
        if not has_code or has_name:
            return sql, df
        # Envolver a consulta original e trazer colunas amigáveis
        limit = self._extract_limit(sql) or self.default_limit
        wrapped = f"""
WITH q AS (
    {sql}
)
SELECT
    COALESCE(t.product_name[1].text, NULL) AS produto,
    t.code,
    t.brands,
    t.product_quantity,
    t.categories
FROM q
JOIN {self.table_name} t USING (code)
LIMIT {limit}
"""
        try:
            enriched = self.run_sql(wrapped)
            if enriched is not None and not enriched.empty:
                return wrapped.strip(), enriched
        except Exception:
            pass
        return sql, df

    # ---------- Utilidades Interativas ----------
    def print_schema(self) -> None:
        print(self.schema_prompt)

    def sample(self, n: int = 5) -> pd.DataFrame:
        return self.run_sql(f"SELECT * FROM {self.table_name} LIMIT {n}")


# ==========================
# CLI / REPL
# ==========================

def main() -> None:
    import argparse

    default_parquet = os.environ.get(
        "PARQUET_PATH", "/home/r2d2/Documents/repos/nfce-api/data/off/food.parquet"
    )

    parser = argparse.ArgumentParser(
        description="Agente de IA para consultas SQL (DuckDB) sobre arquivo Parquet"
    )
    parser.add_argument("--parquet", default=default_parquet, help="Caminho do arquivo .parquet")
    parser.add_argument("--table", default="food", help="Nome lógico da tabela/VIEW")
    parser.add_argument("--limit", type=int, default=50, help="LIMIT padrão para consultas")
    parser.add_argument("--no-llm", action="store_true", help="Desabilita LLM (modo manual)")
    parser.add_argument("--model", default=os.environ.get("GEMINI_MODEL", "gemini-1.5-flash"), help="Modelo Gemini a usar")
    parser.add_argument("--ask", default=None, help="Pergunta única (modo não interativo)")
    parser.add_argument("--no-schema-samples", action="store_true", help="Não coletar amostras por coluna no schema")
    parser.add_argument("--schema-samples", type=int, default=2, help="Qtde de amostras por coluna simples no schema")
    parser.add_argument("--schema-max-cols", type=int, default=30, help="Máximo de colunas do schema para incluir no prompt")
    parser.add_argument("--verbose", action="store_true", help="Exibe logs de progresso")
    parser.add_argument("--no-enrich", action="store_true", help="Desativa enriquecimento automático do resultado por code")
    parser.add_argument("--output", choices=["auto", "pretty", "plain", "csv", "json"], default="auto", help="Formato de saída no terminal")
    parser.add_argument("--no-nlg", action="store_true", help="Desativa resumo em linguagem natural")
    parser.add_argument("--max-colwidth", type=int, default=100, help="Largura máxima por coluna para exibição")
    parser.add_argument("--no-color", action="store_true", help="Desativa cores (Rich)")

    args = parser.parse_args()

    # Impressão / UI
    printer = UIPrinter(mode=args.output, max_colwidth=args.max_colwidth, no_color=args.no_color)

    # Instancia cliente LLM (opcional)
    model_client = None
    if not args.no_llm:
        try:
            model_client = GeminiClient(model_name=args.model)
        except Exception as e:
            printer.print_warn(f"Aviso: LLM indisponível ({e}). Continuando em modo manual.")

    try:
        if args.verbose:
            printer.print_info("Carregando esquema e preparando VIEW…")
        else:
            printer.print_info("Preparando esquema (pode levar alguns segundos)…")
        agent = SQLAgent(
            parquet_path=args.parquet,
            table_name=args.table,
            default_limit=args.limit,
            model_client=model_client,
            do_schema_samples=not args.no_schema_samples,
            schema_samples_per_col=args.schema_samples,
            schema_max_cols=args.schema_max_cols,
            verbose=args.verbose,
            enrich=not args.no_enrich,
        )
        if args.verbose:
            printer.print_info("Esquema pronto.")
        else:
            printer.print_info("Esquema pronto.")
    except Exception as e:
        printer.print_err(f"Erro ao inicializar o agente: {e}")
        sys.exit(1)

    pd.set_option("display.max_rows", 200)
    pd.set_option("display.max_colwidth", 120)

    if args.ask:
        if args.verbose:
            printer.print_info("Gerando SQL com IA…")
        else:
            printer.print_info("Gerando SQL…")
        start = time.time()
        sql, df, err = agent.ask(args.ask)
        elapsed = (time.time() - start) * 1000
        printer.print_info(f"SQL Gerado:\n{sql}\n")
        if err:
            printer.print_err(f"Erro: {err}")
            sys.exit(2)
        printer.print_df(df)
        if nlg_enabled:
            summary = summarize_df(df)
            if summary:
                printer.print_info(summary)
        rows = 0 if df is None else len(df)
        printer.print_info(f"Linhas: {rows} | Tempo: {elapsed:.0f} ms")
        return

    printer.print_info("--- Agente de Consulta SQL (DuckDB + Parquet) ---")
    printer.print_info(f"Tabela: {args.table}  |  Arquivo: {args.parquet}")
    printer.print_info("Comandos: /help para ver ajuda.")
    if model_client:
        printer.print_info("Modo: IA habilitada (Gemini)")
    else:
        printer.print_info("Modo: manual (sem LLM)")

    last_df: Optional[pd.DataFrame] = None
    last_sql: Optional[str] = None
    last_elapsed_ms: Optional[float] = None
    nlg_enabled: bool = not args.no_nlg

    def summarize_df(df: Optional[pd.DataFrame]) -> Optional[str]:
        if df is None or df.empty:
            return None
        lines: List[str] = []
        try:
            cnt = len(df)
            lines.append(f"Resultados: {cnt}")
            # Proteínas por 100g
            if "proteins_100g" in df.columns:
                s = pd.to_numeric(df["proteins_100g"], errors="coerce").dropna()
                if not s.empty:
                    unit = None
                    if "proteins_unit" in df.columns and df["proteins_unit"].notna().any():
                        unit = str(df["proteins_unit"].dropna().iloc[0])
                    lines.append(
                        f"Proteína por 100 g: média {s.mean():.2f}{unit or ''} (mín {s.min():.2f}, máx {s.max():.2f})."
                    )
                    if set(["name", "brands"]).issubset(df.columns):
                        top = df.loc[df["proteins_100g"].notna(), ["name", "brands", "proteins_100g"]].sort_values("proteins_100g", ascending=False).head(3)
                        if not top.empty:
                            ex = "; ".join([f"{r.name} ({r.brands}): {r.proteins_100g:.2f}{unit or ''}" for r in top.itertuples(index=False)])
                            lines.append(f"Mais altos: {ex}.")
            # Calorias por 100g
            if "energy_kcal_100g" in df.columns:
                s = pd.to_numeric(df["energy_kcal_100g"], errors="coerce").dropna()
                if not s.empty:
                    unit = None
                    if "energy_kcal_unit" in df.columns and df["energy_kcal_unit"].notna().any():
                        unit = str(df["energy_kcal_unit"].dropna().iloc[0])
                    lines.append(
                        f"Calorias por 100 g: média {s.mean():.0f}{unit or ' kcal'} (mín {s.min():.0f}, máx {s.max():.0f})."
                    )
            # Quantidades
            if "quantity_g" in df.columns and df["quantity_g"].notna().any():
                sg = pd.to_numeric(df["quantity_g"], errors="coerce").dropna()
                if not sg.empty:
                    lines.append(f"Peso total: mediana {sg.median():.0f} g (mín {sg.min():.0f}, máx {sg.max():.0f}).")
            if "quantity_ml" in df.columns and df["quantity_ml"].notna().any():
                sv = pd.to_numeric(df["quantity_ml"], errors="coerce").dropna()
                if not sv.empty:
                    lines.append(f"Volume total: mediana {sv.median():.0f} ml (mín {sv.min():.0f}, máx {sv.max():.0f}).")
        except Exception:
            return None
        return " ".join(lines) if lines else None

    while True:
        try:
            q = input("\nPergunta ou comando: ").strip()
        except (EOFError, KeyboardInterrupt):
            printer.print_info("\nEncerrando...")
            break

        if not q:
            continue
        if q.lower() in {"sair", "exit", ":q"}:
            break

        # Comandos utilitários
        if q.startswith("/help"):
            help_text = (
                "Comandos:\n"
                "/schema                      - mostra colunas/tipos e amostras simples\n"
                "/sample [n]                  - retorna N linhas de amostra\n"
                "/sql <consulta>              - executa SELECT manual (com LIMIT automático)\n"
                "/limit <n>                   - altera LIMIT padrão\n"
                "/enrich on|off               - liga/desliga enriquecimento do resultado por code\n"
                "/format pretty|plain|csv|json- muda formato de exibição\n"
                "/save <path> [csv|json]      - salva o último resultado\n"
                "/cols                        - lista colunas do último resultado\n"
                "/select <c1> [c2 ...]        - exibe somente estas colunas do último resultado\n"
                "/head [n]                    - mostra topo do último resultado (padrão 10)\n"
                "sair                         - encerra"
            )
            print(help_text)
            continue

        if q.startswith("/schema"):
            agent.print_schema()
            continue
        if q.startswith("/sample"):
            try:
                n = int(q.split()[1]) if len(q.split()) > 1 else 5
            except Exception:
                n = 5
            try:
                df = agent.sample(n)
                printer.print_df(df)
                last_df, last_sql, last_elapsed_ms = df, f"SELECT * FROM {agent.table_name} LIMIT {n}", None
            except Exception as e:
                printer.print_err(f"Erro ao coletar amostra: {e}")
            continue
        if q.startswith("/limit"):
            try:
                n = int(q.split()[1])
                agent.default_limit = n
                printer.print_info(f"LIMIT padrão atualizado para {n}")
            except Exception:
                printer.print_warn("Uso: /limit <numero>")
            continue
        if q.startswith("/nlg"):
            parts = q.split()
            if len(parts) == 2 and parts[1].lower() in {"on", "off"}:
                nlg_enabled = parts[1].lower() == "on"
                printer.print_info(f"Resumo NLG: {'ligado' if nlg_enabled else 'desligado'}")
            else:
                printer.print_warn("Uso: /nlg on|off")
            continue
        if q.startswith("/sql "):
            raw_sql = q[5:].strip()
            if not _ensure_select_only(raw_sql):
                printer.print_warn("Apenas SELECT é permitido.")
                continue
            raw_sql = _add_limit_if_missing(raw_sql, agent.default_limit)
            try:
                df = agent.run_sql(raw_sql)
                # Enriquecer se habilitado
                if agent.enrich and df is not None and not df.empty:
                    raw_sql, df = agent._maybe_enrich(raw_sql, df)
                printer.print_df(df)
                if nlg_enabled:
                    summary = summarize_df(df)
                    if summary:
                        printer.print_info(summary)
                last_df, last_sql, last_elapsed_ms = df, raw_sql, None
            except Exception as e:
                printer.print_err(f"Erro ao executar SQL: {e}")
            continue

        if q.startswith("/enrich"):
            parts = q.split()
            if len(parts) == 2 and parts[1].lower() in {"on", "off"}:
                agent.enrich = parts[1].lower() == "on"
                printer.print_info(f"Enriquecimento: {'ligado' if agent.enrich else 'desligado'}")
            else:
                printer.print_warn("Uso: /enrich on|off")
            continue

        if q.startswith("/format"):
            parts = q.split()
            if len(parts) == 2 and parts[1].lower() in {"pretty", "plain", "csv", "json"}:
                printer.set_mode(parts[1].lower())
                printer.print_info(f"Formato de saída: {parts[1].lower()}")
            else:
                printer.print_warn("Uso: /format pretty|plain|csv|json")
            continue

        if q.startswith("/save"):
            parts = q.split()
            if last_df is None or last_df.empty:
                printer.print_warn("Sem resultado para salvar.")
                continue
            if len(parts) >= 2:
                path = parts[1]
                fmt = parts[2].lower() if len(parts) >= 3 else ("json" if path.lower().endswith(".json") else "csv")
                try:
                    if fmt == "json":
                        last_df.to_json(path, orient="records", force_ascii=False)
                    else:
                        last_df.to_csv(path, index=False)
                    printer.print_info(f"Salvo em {path} ({fmt}).")
                except Exception as e:
                    printer.print_err(f"Erro ao salvar: {e}")
            else:
                printer.print_warn("Uso: /save <path> [csv|json]")
            continue

        if q.startswith("/cols"):
            if last_df is None or last_df.empty:
                printer.print_warn("Sem resultado para inspecionar.")
            else:
                cols = ", ".join(map(str, last_df.columns))
                printer.print_info(f"Colunas: {cols}")
            continue

        if q.startswith("/select"):
            parts = q.split()
            if last_df is None or last_df.empty:
                printer.print_warn("Sem resultado para selecionar colunas.")
                continue
            if len(parts) < 2:
                printer.print_warn("Uso: /select <c1> [c2 ...]")
                continue
            sel_cols = parts[1:]
            try:
                sub = last_df.loc[:, sel_cols]
                printer.print_df(sub)
            except Exception as e:
                printer.print_err(f"Erro ao selecionar colunas: {e}")
            continue

        if q.startswith("/head"):
            parts = q.split()
            if last_df is None or last_df.empty:
                printer.print_warn("Sem resultado para exibir.")
                continue
            try:
                n = int(parts[1]) if len(parts) > 1 else 10
            except Exception:
                n = 10
            printer.print_df(last_df.head(n))
            continue

        # Fluxo IA (pergunta em linguagem natural)
        start = time.time()
        sql, df, err = agent.ask(q)
        elapsed = (time.time() - start) * 1000

        printer.print_info(f"SQL Gerado:\n{sql}\n")
        if err:
            printer.print_err(f"Erro: {err}")
            continue
        printer.print_df(df)
        if nlg_enabled:
            summary = summarize_df(df)
            if summary:
                printer.print_info(summary)
        last_df, last_sql, last_elapsed_ms = df, sql, elapsed
        rows = 0 if df is None else len(df)
        printer.print_info(f"Linhas: {rows} | Tempo: {elapsed:.0f} ms")


if __name__ == "__main__":
    main()

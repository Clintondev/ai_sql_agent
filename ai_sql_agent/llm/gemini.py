import os
from typing import List, Optional

try:
    import google.generativeai as genai
except Exception:
    genai = None

from ai_sql_agent.utils.sql_safety import _clean_sql


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
        - Ao filtrar por termo no nome, considere ambos: list_contains(list_transform(product_name, p -> lower(p.text)), lower('<termo>')) OU lower(categories) LIKE '%<termo>%'. Nas views, prefira colunas normalizadas name_norm, brands_norm, categories_norm com LIKE '%termo%'.
        - product_name é LIST<STRUCT>; para projetar use product_name[1].text; para filtrar use list_contains(list_transform(product_name, p -> lower(p.text)), '<termo>').
        - Para nutrientes detalhados na tabela original use UNNEST(nutriments) AS n e depois n.unnest."100g" (por 100g), n.unnest.serving (por porção), n.unnest.unit; nomes comuns: energy-kcal, proteins, fat, saturated-fat, sugars, carbohydrates, fiber, salt, sodium.
        - Alternativamente, a view {table_name}_flat já expõe colunas: energy_kcal_100g, proteins_100g, fat_100g, saturated_fat_100g, carbs_100g, sugars_100g, fiber_100g, salt_100g, sodium_100g, além de *_serving e *_unit quando aplicável, e quantity_g/quantity_ml; e ainda name_norm, brands_norm, categories_norm para filtragem robusta.

        Pergunta do usuário (PT-BR): {question}

        Responda SOMENTE com a consulta SQL (sem explicações).
        """
        if examples:
            prompt = prompt.replace("Responda SOMENTE com a consulta SQL (sem explicações).", "\nExemplos:\n" + "\n".join(examples) + "\n\nResponda SOMENTE com a consulta SQL (sem explicações).")
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


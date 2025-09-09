import re


def _clean_sql(sql: str) -> str:
    if not sql:
        return ""
    s = sql.strip()
    if s.lower().startswith("```sql"):
        s = s[6:].strip()
    if s.startswith("```"):
        s = s[3:].strip()
    if s.endswith("```"):
        s = s[:-3].strip()
    m = re.search(r"\bselect\b", s, flags=re.I)
    if m:
        s = s[m.start():].strip()
    s = s.rstrip("; ")
    return s


def _ensure_select_only(sql: str) -> bool:
    if not sql:
        return False
    banned = [r"\b(drop|alter|truncate|insert|update|delete|create|attach|copy)\b"]
    if sql.count(";") > 0:
        return False
    if not re.match(r"^\s*select\b", sql, flags=re.I):
        return False
    for pat in banned:
        if re.search(pat, sql, flags=re.I):
            return False
    return True


def _add_limit_if_missing(sql: str, default_limit: int = 50) -> str:
    if re.search(r"\blimit\s+\d+\b", sql, flags=re.I):
        return sql
    if re.search(r"\b(sample|using\s+sample)\b", sql, flags=re.I):
        return sql
    return f"{sql}\nLIMIT {default_limit}"


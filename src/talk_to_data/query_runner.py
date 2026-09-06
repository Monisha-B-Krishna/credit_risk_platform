"""Loads the Home Credit tables into DuckDB and executes SQL safely.

DuckDB is used instead of a full database server - it runs in-process,
needs no setup, and reads pandas DataFrames / CSVs directly. This is
what the chatbot actually queries against; it's a separate concern
from the ML pipeline's feature matrix (src/data/preprocessor.py).

Safety: only SELECT statements are allowed. Any query containing a
destructive keyword (DROP, DELETE, UPDATE, INSERT, ALTER, etc.) is
rejected before it ever reaches the database - this is the primary
hallucination/misuse control for the NL-to-SQL chatbot, since an LLM
could otherwise generate a destructive query either by mistake or in
response to an adversarial prompt.
"""

import os
import re

import duckdb
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# Only these tables are exposed to the chatbot. Kept to a manageable
# set rather than all 8 raw tables, so the schema shown to the LLM
# stays small (token cost) and the tables are genuinely useful for the
# kinds of questions a credit risk analyst would actually ask.
TABLE_FILES = {
    "applications": "application_train.csv",
    "bureau": "bureau.csv",
    "previous_applications": "previous_application.csv",
    "installments": "installments_payments.csv",
}

# The applications table has 122 raw columns, most of which (building
# detail fields, document flags, etc.) add no value to the kinds of
# questions a credit risk analyst would ask and would only inflate the
# token cost of every prompt sent to the LLM. Only these columns are
# shown in the schema summary - the underlying DuckDB table still has
# every column, so a query can reference any of them if the LLM
# genuinely needs to.
RELEVANT_COLUMNS = {
    "applications": [
        "SK_ID_CURR", "TARGET", "NAME_CONTRACT_TYPE", "CODE_GENDER",
        "FLAG_OWN_CAR", "FLAG_OWN_REALTY", "CNT_CHILDREN",
        "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE",
        "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS",
        "NAME_HOUSING_TYPE", "DAYS_BIRTH", "DAYS_EMPLOYED",
        "OCCUPATION_TYPE", "CNT_FAM_MEMBERS", "EXT_SOURCE_1",
        "EXT_SOURCE_2", "EXT_SOURCE_3",
    ],
    "bureau": [
        "SK_ID_CURR", "SK_ID_BUREAU", "CREDIT_ACTIVE", "DAYS_CREDIT",
        "CREDIT_DAY_OVERDUE", "AMT_CREDIT_SUM", "AMT_CREDIT_SUM_DEBT",
        "AMT_CREDIT_SUM_OVERDUE", "CREDIT_TYPE",
    ],
    "previous_applications": [
        "SK_ID_PREV", "SK_ID_CURR", "NAME_CONTRACT_TYPE", "AMT_ANNUITY",
        "AMT_APPLICATION", "AMT_CREDIT", "NAME_CONTRACT_STATUS",
        "NAME_CASH_LOAN_PURPOSE", "DAYS_DECISION", "CODE_REJECT_REASON",
    ],
    "installments": [
        "SK_ID_PREV", "SK_ID_CURR", "NUM_INSTALMENT_NUMBER",
        "DAYS_INSTALMENT", "DAYS_ENTRY_PAYMENT", "AMT_INSTALMENT",
        "AMT_PAYMENT",
    ],
}

# Any query containing one of these (case-insensitive, as a whole
# word) is rejected outright.
BLOCKED_KEYWORDS = [
    "DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "CREATE",
    "TRUNCATE", "REPLACE", "ATTACH", "COPY", "EXPORT", "PRAGMA",
]


class SQLSafetyError(Exception):
    """Raised when a query fails the safety validation checks."""
    pass


_cached_connection = None


def get_connection() -> duckdb.DuckDBPyConnection:
    """Creates (once) and reuses an in-memory DuckDB connection with all
    chatbot tables loaded. Cached at module level rather than reloaded
    per call - reloading ~13.6 million installment rows on every single
    question would make the chatbot noticeably slow in real use.
    """
    global _cached_connection

    if _cached_connection is not None:
        return _cached_connection

    con = duckdb.connect(database=":memory:")

    for table_name, filename in TABLE_FILES.items():
        file_path = os.path.join(DATA_DIR, filename)
        if not os.path.exists(file_path):
            print(f"[WARNING] {filename} not found, table '{table_name}' will be unavailable.")
            continue
        con.execute(f"""
            CREATE TABLE {table_name} AS
            SELECT * FROM read_csv_auto('{file_path}')
        """)

    _cached_connection = con
    return _cached_connection


def validate_sql(sql: str) -> None:
    """Raises SQLSafetyError if the query isn't a safe, read-only SELECT.

    Checks, in order:
        1. Query must start with SELECT (after stripping whitespace/comments)
        2. Query must not contain any blocked destructive keyword
        3. Query must not contain a semicolon followed by more SQL
           (blocks statement-stacking, e.g. "SELECT ...; DROP TABLE ...")
    """
    cleaned = sql.strip()

    # Strip leading SQL comments before checking the first keyword.
    cleaned_no_comments = re.sub(r"--.*?\n", "\n", cleaned)
    cleaned_no_comments = re.sub(r"/\*.*?\*/", "", cleaned_no_comments, flags=re.DOTALL)
    cleaned_no_comments = cleaned_no_comments.strip()

    upper_start = cleaned_no_comments.upper()
    if not (upper_start.startswith("SELECT") or upper_start.startswith("WITH")):
        raise SQLSafetyError(
            "Only SELECT queries are allowed (including WITH ... SELECT "
            "common table expressions). Query must start with SELECT or WITH."
        )

    upper_sql = cleaned_no_comments.upper()
    for keyword in BLOCKED_KEYWORDS:
        if re.search(rf"\b{keyword}\b", upper_sql):
            raise SQLSafetyError(
                f"Query contains a disallowed keyword: {keyword}. "
                f"Only read-only SELECT queries are permitted."
            )

    # Reject statement-stacking: a semicolon followed by more
    # non-whitespace content means multiple statements were submitted.
    stripped_trailing = cleaned_no_comments.rstrip(";").rstrip()
    if ";" in stripped_trailing:
        raise SQLSafetyError(
            "Multiple statements are not allowed. Submit one SELECT query at a time."
        )


def run_query(sql: str, max_rows: int = 20) -> pd.DataFrame:
    """Validates and executes a SQL query, returning at most max_rows.

    Row limiting happens here (not left to the LLM's discretion) to
    keep both execution time and the token cost of summarizing results
    back through the LLM bounded, regardless of what the query asks for.
    """
    validate_sql(sql)

    con = get_connection()
    try:
        result = con.execute(sql).fetchdf()
    except duckdb.Error as e:
        raise SQLSafetyError(f"Query failed to execute: {e}")

    if len(result) > max_rows:
        result = result.head(max_rows)

    return result


_cached_schema_summary = None


def get_schema_summary() -> str:
    """Returns a compact text description of every table's columns and
    a couple of sample rows - used to ground the LLM's SQL generation
    (see prompt_templates.py). Cached at module level since the schema
    doesn't change during a single run."""
    global _cached_schema_summary

    if _cached_schema_summary is not None:
        return _cached_schema_summary

    con = get_connection()
    lines = []

    for table_name in TABLE_FILES:
        relevant_cols = RELEVANT_COLUMNS.get(table_name)
        if not relevant_cols:
            continue

        try:
            col_list_sql = ", ".join(relevant_cols)
            sample = con.execute(
                f"SELECT {col_list_sql} FROM {table_name} LIMIT 2"
            ).fetchdf()
        except duckdb.Error:
            continue

        lines.append(f"Table: {table_name}")
        lines.append(f"Columns: {', '.join(relevant_cols)}")
        lines.append(f"Sample rows:\n{sample.to_string(index=False)}")
        lines.append("")

    _cached_schema_summary = "\n".join(lines)
    return _cached_schema_summary


if __name__ == "__main__":
    # Quick manual check: `python src/talk_to_data/query_runner.py`
    print("Schema summary:")
    print(get_schema_summary())

    print("\nTest query:")
    df = run_query("SELECT COUNT(*) AS total_applicants FROM applications")
    print(df)
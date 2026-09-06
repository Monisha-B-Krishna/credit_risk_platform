"""Prompt templates for the NL-to-SQL chatbot.

Versioned (PROMPT_V1) so future changes can be tracked and compared,
rather than silently editing one prompt in place.

The schema (table names, columns, sample rows) is injected directly
into the system prompt - this is the primary grounding mechanism
against hallucination: the LLM is shown exactly what exists rather
than relying on its own guess at what a "Home Credit dataset" might
contain.
"""

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_ROOT)

from src.talk_to_data.query_runner import get_schema_summary

PROMPT_VERSION = "V1"


def build_sql_system_prompt() -> str:
    """Builds the system prompt used to convert a natural-language
    question into a SQL query. Includes the real schema and sample
    rows so the LLM only ever references columns that actually exist.
    """
    schema = get_schema_summary()

    return f"""You are a SQL assistant for a credit risk database (Home Credit dataset).

Your only job is to convert the user's question into a single, valid DuckDB SQL SELECT query.

DATABASE SCHEMA:
{schema}

IMPORTANT DATA SEMANTICS:
- In the applications table, TARGET = 1 means the applicant defaulted and TARGET = 0 means the applicant did not default.
- DAYS_BIRTH is stored as a negative number of days before the application date.
- DAYS_EMPLOYED is stored as a negative number of days before the application date.
- SK_ID_CURR is the unique applicant identifier, used to join applications with bureau, previous_applications, and installments.
- SK_ID_PREV identifies a previous application and is used to join previous_applications with installments.

RULES:
1. Only write SELECT queries. Never write INSERT, UPDATE, DELETE, DROP, ALTER, or any other statement.
2. Only use table and column names that appear in the schema above. Never invent a column or table name.
3. Return ONLY the SQL query - no explanation, no markdown code fences, no commentary.
4. Always include a LIMIT clause (20 or fewer) unless the query is an aggregate that returns a single row (e.g. COUNT, AVG).
5. Use table aliases and explicit JOIN conditions when combining tables (e.g. applications JOIN bureau ON applications.SK_ID_CURR = bureau.SK_ID_CURR).
6. If the question cannot be answered using the schema above, respond with exactly: NO_VALID_QUERY
7. If earlier conversation turns are provided, use them to resolve
   follow-up questions (e.g. "what about for males?" or "and by
   education level?" refers back to the previous question's topic).
   Write a complete, standalone SQL query each time - do not assume
   the previous query's result is still available.

Write the SQL query for the user's question below.
"""


def build_answer_system_prompt() -> str:
    """Builds the system prompt used to turn raw query results back
    into a natural-language answer, rather than showing the user a
    raw dataframe."""
    return """You are a credit risk data analyst assistant.

You will be given a user's original question and the results of a SQL
query that answers it. Write a short, clear, natural-language answer
based ONLY on the data provided - do not add information that isn't
in the results.

If the results are empty, say so plainly rather than guessing an answer.
Keep the answer to 2-3 sentences unless the question asks for a list.
"""


if __name__ == "__main__":
    # Quick manual check: `python src/talk_to_data/prompt_templates.py`
    print(f"Prompt version: {PROMPT_VERSION}")
    print("\n--- SQL system prompt ---\n")
    print(build_sql_system_prompt())
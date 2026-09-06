"""Converts a natural-language question into SQL, runs it safely, and
returns a plain-language answer.

Pipeline:
    question -> Groq LLM generates SQL (grounded by the real schema,
    prompt_templates.py, and recent conversation history for follow-up
    questions) -> SQL validated (query_runner.py) -> SQL executed
    against DuckDB -> results summarized back into a readable answer
    by a second Groq call.

Two separate LLM calls are used (one for SQL generation, one for the
final answer) rather than one combined call, so each prompt stays
focused and small - this keeps token usage down and makes each step
easier to debug independently.

Run from the project root: `python src/talk_to_data/nl_to_sql.py` for
a demo covering 5+ example questions.
"""

import os
import re
import sys

from dotenv import load_dotenv
from groq import Groq

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_ROOT)
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

from src.talk_to_data.query_runner import run_query, SQLSafetyError
from src.talk_to_data.prompt_templates import (
    build_sql_system_prompt,
    build_answer_system_prompt,
)

LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")
MAX_RETRIES = 1  # one retry with the error fed back, if the first SQL attempt fails
MAX_HISTORY_TURNS = 3  # how many recent exchanges are fed back for follow-up questions


def get_groq_client() -> Groq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY not found. Copy .env.example to .env and add a "
            "free key from https://console.groq.com/keys"
        )
    return Groq(api_key=api_key)


def clean_sql_response(raw_response: str) -> str:
    """Strips markdown code fences and extra whitespace the LLM might
    add despite being asked not to - defensive cleanup, not a
    substitute for the actual safety validation in query_runner.py."""
    cleaned = raw_response.strip()
    cleaned = re.sub(r"^```(sql)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"```\s*$", "", cleaned)
    return cleaned.strip()


def _format_history(conversation_history: list) -> str:
    """Formats recent Q&A turns into a short context block for the
    prompt. Limited to the last MAX_HISTORY_TURNS exchanges to keep
    token usage bounded rather than growing unboundedly over a long
    conversation."""
    if not conversation_history:
        return ""

    recent = conversation_history[-MAX_HISTORY_TURNS:]
    lines = []
    for turn in recent:
        lines.append(f"Previous question: {turn['question']}")
        lines.append(f"Previous answer: {turn['answer']}")
    return "\n".join(lines) + "\n\n"


def generate_sql(client: Groq, question: str, conversation_history: list = None,
                  error_context: str = None) -> str:
    """Calls Groq to convert a question into SQL.

    conversation_history (a list of {"question": ..., "answer": ...}
    dicts, most recent last) is included so follow-up questions like
    "what about for males?" can be resolved using the prior topic.
    error_context, if given, lets the model see and correct a previous
    failed attempt.
    """
    system_prompt = build_sql_system_prompt()

    history_block = _format_history(conversation_history)
    user_message = f"{history_block}Current question: {question}" if history_block else question

    if error_context:
        user_message += (
            f"\n\nYour previous SQL attempt failed with this error:\n{error_context}\n"
            f"Please generate a corrected query."
        )

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=0,  # deterministic output for SQL generation
        max_tokens=500,
    )

    return clean_sql_response(response.choices[0].message.content)


def generate_answer(client: Groq, question: str, sql: str, results_text: str) -> str:
    """Calls Groq to turn raw query results into a natural-language answer."""
    system_prompt = build_answer_system_prompt()
    user_message = (
        f"Question: {question}\n\n"
        f"SQL query used: {sql}\n\n"
        f"Query results:\n{results_text}"
    )

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=0.3,
        max_tokens=300,
    )

    return response.choices[0].message.content.strip()


def ask(question: str, conversation_history: list = None) -> dict:
    """Main entry point: question in, answer out.

    Parameters
    ----------
    question : str
    conversation_history : list of {"question": str, "answer": str}, optional
        Recent prior exchanges in this conversation, most recent last.
        Used to resolve follow-up questions - not persisted anywhere,
        the caller (e.g. the Streamlit UI) is responsible for keeping
        this across turns.

    Returns
    -------
    dict with keys: question, sql, results (as a string), answer, error
    """
    client = get_groq_client()

    sql = generate_sql(client, question, conversation_history=conversation_history)

    if sql.strip().upper() == "NO_VALID_QUERY":
        return {
            "question": question,
            "sql": None,
            "results": None,
            "answer": "This question can't be answered with the data available "
                      "in this database.",
            "error": "NO_VALID_QUERY",
        }

    last_error = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            result_df = run_query(sql)
            break
        except SQLSafetyError as e:
            last_error = str(e)
            if attempt < MAX_RETRIES:
                sql = generate_sql(client, question, conversation_history=conversation_history,
                                    error_context=last_error)
            else:
                return {
                    "question": question,
                    "sql": sql,
                    "results": None,
                    "answer": "I wasn't able to generate a safe, valid query "
                              "for this question.",
                    "error": last_error,
                }

    results_text = result_df.to_string(index=False) if not result_df.empty else "(no rows returned)"
    answer = generate_answer(client, question, sql, results_text)

    return {
        "question": question,
        "sql": sql,
        "results": results_text,
        "answer": answer,
        "error": None,
    }


DEMO_QUESTIONS = [
    "What is the average income of applicants who defaulted versus those who didn't?",
    "How many applicants have more than 2 previous applications that were refused?",
    "What is the default rate for self-employed applicants?",
    "What percentage of applicants are above age 50?",
    "Show the average credit amount by education level.",
]


def run_demo():
    """Runs the 5 built-in example questions - demonstrates the
    required query patterns without needing manual input."""
    for question in DEMO_QUESTIONS:
        print("=" * 70)
        print(f"Q: {question}")
        print("=" * 70)

        result = ask(question)

        if result["error"] and result["error"] != "NO_VALID_QUERY":
            print(f"Error: {result['error']}")
        else:
            print(f"SQL: {result['sql']}")
            print(f"\nAnswer: {result['answer']}")
        print()


def run_interactive():
    """Lets you type your own questions and get real answers, with
    conversation memory across turns (e.g. 'what about for males?'
    resolves using the previous question's topic). Type 'exit' to quit."""
    print("Talk to your credit risk data. Type 'exit' to quit.\n")
    history = []
    while True:
        question = input("Ask a question: ").strip()
        if question.lower() in ("exit", "quit", ""):
            print("Goodbye.")
            break

        result = ask(question, conversation_history=history)

        if result["error"] and result["error"] != "NO_VALID_QUERY":
            print(f"Error: {result['error']}\n")
        else:
            print(f"SQL: {result['sql']}")
            print(f"Answer: {result['answer']}\n")
            history.append({"question": question, "answer": result["answer"]})


if __name__ == "__main__":
    import sys as _sys

    if "--demo" in _sys.argv:
        run_demo()
    else:
        run_interactive()
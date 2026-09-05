"""Loads all Home Credit CSV files into a dict of DataFrames.

Table relationships:
    application_train.csv / application_test.csv
        Main table, one row per loan application. Primary key: SK_ID_CURR.
        Only application_train has TARGET (1=defaulted, 0=repaid).

    bureau.csv
        Applicant credit history at other lenders. Joins via SK_ID_CURR.

    bureau_balance.csv
        Monthly balance history per bureau record. Joins to bureau.csv
        via SK_ID_BUREAU.

    previous_application.csv
        Applicant's past applications with Home Credit. Joins via SK_ID_CURR.

    POS_CASH_balance.csv / installments_payments.csv / credit_card_balance.csv
        Monthly/payment-level history for past loans. Join to
        previous_application.csv via SK_ID_PREV.
"""

import pandas as pd
from pathlib import Path

FILE_MAP = {
    "app_train": "application_train.csv",
    "app_test": "application_test.csv",
    "bureau": "bureau.csv",
    "bureau_balance": "bureau_balance.csv",
    "previous_application": "previous_application.csv",
    "pos_cash": "POS_CASH_balance.csv",
    "installments": "installments_payments.csv",
    "credit_card": "credit_card_balance.csv",
}


def load_all_tables(data_dir: str = "data") -> dict:
    """Load every Home Credit CSV into a dict of DataFrames.

    Parameters
    ----------
    data_dir : str
        Folder containing the raw CSVs.

    Returns
    -------
    dict[str, pd.DataFrame]
        Keyed by short table name (see FILE_MAP).
    """
    data_dir = Path(data_dir)
    tables = {}

    for short_name, filename in FILE_MAP.items():
        file_path = data_dir / filename
        if not file_path.exists():
            print(f"[WARNING] {filename} not found at {file_path}, skipping.")
            continue
        tables[short_name] = pd.read_csv(file_path)
        print(f"Loaded {short_name}: {tables[short_name].shape}")

    return tables


if __name__ == "__main__":
    tables = load_all_tables()
    print("\nTables loaded:", list(tables.keys()))
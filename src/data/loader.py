"""
loader.py

I wrote this to load all the Home Credit tables from the data/ folder.
The Home Credit dataset isn't one flat table - it's 8 CSVs that connect
to each other through ID columns, similar to tables in a database.

Here's how they connect (I worked this out during EDA):

    application_train.csv / application_test.csv
        - The main table. One row = one loan application.
        - Primary key: SK_ID_CURR
        - application_train has the TARGET column (1 = defaulted, 0 = repaid).
          application_test does NOT have TARGET - that's what we'd predict
          for a real submission, but for this assignment I train/validate
          only on application_train since it's the only labeled data.

    bureau.csv
        - Applicant's credit history at OTHER banks/lenders.
        - Joins to application via SK_ID_CURR (one applicant -> many bureau rows)

    bureau_balance.csv
        - Monthly balance snapshots for each bureau record.
        - Joins to bureau.csv via SK_ID_BUREAU (one bureau record -> many monthly rows)

    previous_application.csv
        - The applicant's past loan applications with THIS lender (Home Credit).
        - Joins to application via SK_ID_CURR

    POS_CASH_balance.csv
        - Monthly snapshots of past POS/cash loans.
        - Joins to previous_application via SK_ID_PREV

    installments_payments.csv
        - Repayment history for past loans - what was due vs what was paid.
        - Joins to previous_application via SK_ID_PREV

    credit_card_balance.csv
        - Monthly credit card balance snapshots.
        - Joins to previous_application via SK_ID_PREV

I load everything as-is here. Cleaning and feature engineering happen
separately in preprocessor.py - I like keeping "read the file" and
"transform the data" as two different steps so each one is easier to
test and debug on its own.
"""

import pandas as pd
from pathlib import Path


def load_all_tables(data_dir: str = "data") -> dict:
    """
    Loads every Home Credit CSV into a dictionary of DataFrames.

    Parameters
    ----------
    data_dir : str
        Folder where the raw CSVs live (default: "data")

    Returns
    -------
    dict[str, pd.DataFrame]
        Keys are short table names, values are the loaded DataFrames.
    """
    data_dir = Path(data_dir)

    # Mapping of a short, easy name -> actual filename on disk.
    # I use short names everywhere else in the code so it's less typing
    # and less chance of typos vs writing the full filename every time.
    file_map = {
        "app_train": "application_train.csv",
        "app_test": "application_test.csv",
        "bureau": "bureau.csv",
        "bureau_balance": "bureau_balance.csv",
        "previous_application": "previous_application.csv",
        "pos_cash": "POS_CASH_balance.csv",
        "installments": "installments_payments.csv",
        "credit_card": "credit_card_balance.csv",
    }

    tables = {}
    for short_name, filename in file_map.items():
        file_path = data_dir / filename
        if not file_path.exists():
            # I don't want the whole pipeline to crash if one optional
            # file is missing - I just warn and skip it, since not every
            # table is strictly required for a first working version.
            print(f"[WARNING] {filename} not found at {file_path}, skipping.")
            continue
        tables[short_name] = pd.read_csv(file_path)
        print(f"Loaded {short_name}: {tables[short_name].shape}")

    return tables


if __name__ == "__main__":
    # Quick manual test - run this file directly to sanity check loading works
    # e.g. `python src/data/loader.py` from the project root
    tables = load_all_tables()
    print("\nTables loaded:", list(tables.keys()))

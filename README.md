# AI-Powered Credit Risk Intelligence Platform

A credit risk scoring platform built on the Home Credit Default Risk
dataset. Includes data analysis, a trained risk model, model
explainability, a talk-to-data chatbot, and a deployable app.

## Setup

```bash
git clone <this-repo-url>
cd credit_risk_platform
python -m venv venv
venv\Scripts\Activate.ps1        # Windows
# source venv/bin/activate       # Mac/Linux
pip install -r requirements.txt
```

Place the Home Credit CSV files in the `data/` folder (these are not
committed to git - see `.gitignore`).

Copy `.env.example` to `.env` and add a free Groq API key
(https://console.groq.com/keys) once the talk-to-data chatbot is set up.

## Project Structure

```
credit_risk_platform/
├── data/                   # Raw CSV files (not committed)
├── notebooks/
│   ├── eda.ipynb           # Exploratory data analysis
│   └── eda.py              # Same analysis, as a plain script
├── src/
│   ├── data/
│   │   ├── loader.py       # Loads all 8 Home Credit tables
│   │   └── preprocessor.py # Cleans data and builds features
│   ├── ml/
│   │   ├── train.py        # Trains and saves the model
│   │   ├── evaluate.py     # Checks how well the model performs
│   │   ├── predict.py      # Scores a single applicant (in progress)
│   │   └── compare_models.py  # Tests class weighting on/off
│   ├── talk_to_data/       # NL-to-SQL chatbot (not yet built)
│   └── utils/
├── models/                 # Saved model and metrics (model file not committed)
├── outputs/                # Run logs, charts, comparison results
├── documents/               # Final presentation PDF (not yet built)
└── requirements.txt
```

## Architecture Overview

```
                    +----------------------+
                    |   Raw CSV files       |
                    |   (data/)              |
                    +-----------+-----------+
                                |
                    +-----------v-----------+
                    |   loader.py             |
                    |   Loads all 8 tables    |
                    +-----------+-----------+
                                |
                    +-----------v-----------+
                    |   preprocessor.py       |
                    |   Cleans data and        |
                    |   builds features         |
                    +-----------+-----------+
                                |
              +------------------+------------------+
              |                                       |
   +----------v-----------+            +-------------v----------+
   |   train.py              |            |   predict.py             |
   |   Trains the model,      |            |   Scores a new            |
   |   saves it to disk        |----------->|   applicant using the     |
   |                           |  loads      |   saved model              |
   +----------+-----------+  model       +-------------+----------+
              |                                          |
   +----------v-----------+            +-------------v----------+
   |   evaluate.py            |            |   Streamlit app            |
   |   Reports ROC-AUC,         |            |   (planned)                 |
   |   PR-AUC, precision,        |            |                             |
   |   recall                    |            |                             |
   +----------------------+            +------------------------+
```

A separate talk-to-data chatbot (planned) will answer questions by
querying the raw tables directly, independent of the model pipeline
shown above.

## 1. Exploratory Data Analysis

Full analysis is in `notebooks/eda.ipynb`. It covers the dataset
overview, target imbalance, data quality issues, feature groups,
individual feature patterns, how each feature relates to default, and
credit history/repayment behaviour from the extra tables.

**Main findings:**
- Default rate is 8.07% - a highly imbalanced dataset. Accuracy is not
  a useful way to measure the model here; ROC-AUC and PR-AUC are used
  instead.
- The strongest predictors are EXT_SOURCE_1/2/3 (external credit
  scores), age, and how long someone has been employed.
- Applicants with no credit history at other banks actually default
  *more* often than those with some history (known as the "thin file"
  effect in credit risk).
- A history of late payments predicts a higher chance of default on
  the current loan.
- Two data problems were found and fixed: a placeholder value in
  DAYS_EMPLOYED (365243, showing up in about 18% of rows) and one
  applicant with an impossible income entry (117,000,000).

## 2. Data Pipeline

`src/data/loader.py` loads all 8 Home Credit tables and explains how
they connect to each other through shared ID columns.

`src/data/preprocessor.py` does two jobs:
- **Cleaning**: fixes the DAYS_EMPLOYED placeholder value, caps the
  extreme income outlier, removes a handful of rows with an unclear
  gender value, and drops columns that are mostly empty (more than 50%
  missing).
- **Feature engineering**: builds new columns like age in years,
  employment length in years, and two ratios (loan-to-income,
  repayment-to-income). It also summarizes the bureau, previous
  application, and installment payment history tables into one row
  per applicant, since those files originally have many rows per
  person.

The final, cleaned table has 307,507 applicants and 97 columns.

Any remaining missing values are left as-is on purpose rather than
manually filled in - the model (LightGBM) can handle missing values on
its own during training.

## 3. Model

**Algorithm used:** LightGBM, a tree-based model. It was chosen
because it handles categorical columns and missing values without
extra setup, and performs well on this type of tabular data.

**Handling the imbalance:** `class_weight="balanced"` was used instead
of a technique like SMOTE. This avoids creating fake, made-up
applicants, which is a real risk when a dataset has this many
categorical and engineered columns.

**How the model was tested:** the training data was split 80/20, with
the same ~8% default rate kept in both parts. `application_test.csv`
could not be used for testing, since it has no answer column
(`TARGET`) to check predictions against.

### Results

| Metric | Score |
|---|---|
| ROC-AUC | 0.7657 |
| PR-AUC | 0.2562 |
| Precision | 0.1804 |
| Recall | 0.6483 |
| F1 Score | 0.2822 |

### Class Weighting: Tested and Compared

Two identical models were trained on the exact same data and the exact
same validation split. The only difference between them was whether
class weighting was turned on:

| Metric | No Weighting | class_weight="balanced" |
|---|---|---|
| ROC-AUC | 0.7656 | 0.7657 |
| PR-AUC | 0.2578 | 0.2562 |
| Precision | 0.5749 | 0.1804 |
| Recall | 0.0286 | 0.6483 |
| F1 Score | 0.0545 | 0.2822 |

Without class weighting, the model only caught 2.9% of actual
defaulters - even though its precision looked better on paper, it was
barely predicting "default" at all, which makes it close to useless
for this task. ROC-AUC and PR-AUC stayed almost the same between both
versions, which shows both models learned a similar sense of who is
risky - class weighting just changes where the cutoff line sits for
calling someone "risky," not how well the model can tell people apart
in the first place.

Since missing an actual defaulter is a costlier mistake for a bank
than double-checking a safe applicant, `class_weight="balanced"` was
kept as the final choice. The comparison script is in
`src/ml/compare_models.py`.

## 4. Explainability

Not yet built. Planned: SHAP will be used to explain individual
predictions in plain language (e.g. "this applicant is high-risk
mainly because of a low external credit score").

## 5. Talk-to-Data Chatbot

Not yet built. Planned: a chatbot that converts plain-English questions
into SQL queries against the dataset, using Groq's free LLM API.

## 6. Business Rules

Not yet built. Planned: simple, readable rules extracted from the
model (e.g. "if external score is low AND debt ratio is high, flag as
high risk") for a non-technical audience like a credit policy team.

## 7. User Interface

Not yet built. Planned: a Streamlit app with tabs for EDA charts, risk
prediction, explanations, business rules, and the chatbot.

## 8. Docker Deployment

Not yet built. Planned: a Dockerfile and docker-compose.yml so the
whole app runs with a single command.

## Known Limitations

- `application_test.csv` is not used for testing the model, since it
  has no answer column (`TARGET`). A split of `application_train.csv`
  is used instead.
- `class_weight="balanced"` favors catching more defaulters over
  precision (see the comparison table above). This is a deliberate
  choice, not an oversight, based on the higher cost of missing a real
  defaulter.
- The bureau, previous application, and installment features assume
  that history happened before the current loan decision, which is
  reasonable for this dataset but worth stating clearly.

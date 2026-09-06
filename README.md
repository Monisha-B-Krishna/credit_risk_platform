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
│   │   ├── predict.py      # Scores a single applicant, returns probability + risk score/band
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

### Scoring a New Applicant

`src/ml/predict.py` takes a single applicant's details (as a plain
Python dictionary, using the same field names as
`application_train.csv`) and returns a default probability, a risk
score out of 100, and a risk band (Low / Medium / High).

Risk bands are business-rule thresholds chosen for this project, not
statistically derived from the validation set:

| Band | Probability |
|---|---|
| Low | below 10% |
| Medium | 10% - 30% |
| High | 30% and above |

A new applicant naturally won't have every one of the 97 fields the
model was trained on (e.g. no bureau/previous application history yet
for a first-time applicant). Any missing fields are filled the same
way "no history" was handled during training - 0 for count-based
fields, left blank for LightGBM to treat as missing otherwise. The
output includes an `input_completeness` count so it's transparent how
much of the assessment relied on provided information versus defaults.

Demo run (`python src/ml/predict.py`), using an illustrative sample
applicant:

```
Default Probability: 12.93%
Risk Score:          12.93 / 100
Risk Band:           MEDIUM
Provided features:   23
Filled with defaults: 74
```

## 4. Explainability

`src/ml/explain.py` uses SHAP (TreeExplainer) to explain individual
predictions - which features pushed the risk score up or down, and by
how much, in plain language rather than raw SHAP numbers.

Two deliberate exclusions from the displayed explanation:
- Features the applicant didn't provide a real value for (LightGBM can
  treat missingness as informative, but "this increased your risk
  (value: unknown)" isn't a useful explanation for a human reader)
- Protected/sensitive attributes (gender, marital status) - the model
  may use them internally, but showing them as a "reason" for a credit
  decision is a fair-lending concern regardless of what SHAP
  calculated mathematically

Demo run (`python src/ml/explain.py`), same sample applicant as above:

```
Risk Assessment: MEDIUM (12.93/100)

Factors increasing risk:
  - Price of goods being financed: 450000.0
  - Car ownership: N

Factors decreasing risk:
  - External credit score 3: 0.6
  - Education level: Higher education
```

## 5. Talk-to-Data Chatbot

Three files, matching the required structure:
- `src/talk_to_data/query_runner.py` - loads 4 tables (applications,
  bureau, previous_applications, installments) into DuckDB, validates
  and executes SQL safely
- `src/talk_to_data/prompt_templates.py` - the schema-grounded system
  prompt (versioned as V1), including column meanings and join keys
- `src/talk_to_data/nl_to_sql.py` - question -> SQL -> answer, using
  Groq's free API (`openai/gpt-oss-120b`)

**SQL safety validation:** only SELECT (or WITH...SELECT) statements
are allowed; any query containing a destructive keyword (DROP, DELETE,
UPDATE, INSERT, ALTER, etc.) is rejected before it reaches the
database; statement-stacking (multiple queries separated by `;`) is
blocked; results are capped at 20 rows.

**Token optimization:** the schema shown to the LLM is limited to
~20 relevant columns per table (not all 122 raw application columns),
and both the DuckDB connection and schema summary are cached rather
than rebuilt on every question.

**5 example queries tested and working:**
1. Average income of applicants who defaulted vs. didn't
2. Applicants with more than 2 previously refused applications
3. Default rate for self-employed applicants
4. Percentage of applicants above age 50
5. Average credit amount by education level

Query 3 demonstrates a hallucination-control property worth noting
directly: "self-employed" isn't an actual category in this dataset.
Rather than inventing a plausible-sounding but fake percentage, the
system honestly reported that no matching applicants were found.

Run `python src/talk_to_data/nl_to_sql.py` for an interactive prompt to
ask your own questions, or `python src/talk_to_data/nl_to_sql.py --demo`
to re-run the 5 example queries above.

## 6. Business Rules

`src/ml/business_rules.py` trains a shallow decision tree (max depth
3) to approximate the real LightGBM model's predictions - a
"surrogate model" simple enough for a non-technical credit policy
team to review and understand. It is not a replacement for the actual
model, and is not as accurate; validation-set agreement with the real
model's predictions is reported honestly (81.6%) rather than
presenting the simplified rules as equivalent to the real model.

Protected/sensitive attributes (gender, marital status) are excluded
from the surrogate, same as in the SHAP explanations. Raw day-count
columns (DAYS_BIRTH, DAYS_EMPLOYED) are also excluded in favor of
their readable year-based versions already built during feature
engineering.

The resulting rules confirm what EDA and SHAP both found throughout
this project - the two external credit scores (EXT_SOURCE_2/3)
dominate the decision, consistent with them being the strongest
correlated features found back in exploratory analysis.

Sample output (`outputs/business_rules.txt`):
```
|--- External credit score 3 <= 0.44
|   |--- External credit score 2 <= 0.49
|   |   |--- class: 1
|--- External credit score 3 >  0.44
|   |--- External credit score 2 >  0.40
|   |   |--- class: 0

Agreement with the real model on validation data: 81.6%
```

## 7. User Interface

`app.py` (Streamlit) brings every module together into one app, four tabs:

- **EDA** - charts from `outputs/eda_charts/`
- **Assess Risk** - a form collecting applicant details (external credit
  scores are optional), showing the risk score, band, and SHAP
  explanation together in one flow
- **Business Rules** - the simplified decision tree, with the real
  model's agreement percentage shown as a metric
- **Chat** - the talk-to-data chatbot, with clickable example questions
  and a running conversation history

Includes basic input validation (loan amount/income must be positive,
warns if annuity exceeds income), a Responsible AI disclaimer shown
with every prediction, and a custom theme (`.streamlit/config.toml`).

Run with `streamlit run app.py`, or via Docker (see below).

## 8. Docker Deployment

```bash
docker compose up --build
```

Then open `http://localhost:8501`.

**What's included in the image:** all code and the trained model
artifacts (`models/credit_risk_model.joblib` and its metadata) - these
are committed to the repo (unlike the raw dataset) so the EDA,
Assess Risk, and Business Rules tabs work immediately with no setup.

**What's mounted at runtime, not baked into the image:**
- `data/` - the raw CSVs, required only for the Chat tab (DuckDB reads
  them directly). Not committed to git per the assignment's instructions.

**Environment variables:** copy `.env.example` to `.env` and add a free
Groq API key (https://console.groq.com/keys) for the Chat tab to work.

Verified working end-to-end: all four tabs tested successfully running
inside the container.

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
- A new applicant scored through `predict.py` may have many fields
  filled with defaults (e.g. no prior bureau/installment history yet).
  The output reports exactly how many fields were provided versus
  defaulted, so this is transparent rather than hidden.
- Protected/sensitive attributes (gender, marital status) are excluded
  from the displayed SHAP explanation and from the business rules
  surrogate tree, though the underlying model may still use them
  internally. A full fair-lending audit of which features the model
  relies on is outside this assignment's scope, but worth flagging as
  a real consideration for any production credit system.
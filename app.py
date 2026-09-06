"""Streamlit UI for the AI-Powered Credit Risk Intelligence Platform.

Five pages, navigated via a left sidebar rail, each backed by a module
already built and tested independently:
    - Home: landing page, overview
    - EDA: charts saved during exploratory analysis
    - Assess Risk: src/ml/predict.py + src/ml/explain.py, combined
      into one flow so the result and its explanation appear together
    - Business Rules: outputs/business_rules_structured.json
      (precomputed, not recalculated live - training the surrogate
      tree reloads the full dataset and would make every page load slow)
    - Chat: src/talk_to_data/nl_to_sql.py

Run from the project root: `streamlit run app.py`
"""

import glob
import json
import os
import re
import sys

import streamlit as st
import streamlit.components.v1 as components

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.append(PROJECT_ROOT)

from src.ml.predict import predict_risk
from src.ml.explain import explain_prediction

st.set_page_config(
    page_title="Credit Risk Intelligence Platform",
    layout="wide",
)

BRAND = {
    "primary": "#4A9130",
    "primary_light": "#8FCB6B",
    "bg_light": "#EAF3E4",
    "bg_page": "#FAFAF7",
    "text": "#2B2B2B",
    "danger": "#C0392B",
    "danger_bg": "#FCE8E6",
    "warning": "#B8860B",
    "warning_bg": "#FFF6E5",
    "success": "#4A9130",
    "success_bg": "#E8F5E0",
}

st.markdown(f"""
<style>
.kpi-card {{
    background-color: {BRAND['bg_light']};
    border: 1px solid {BRAND['primary_light']};
    border-radius: 12px;
    padding: 18px 10px;
    text-align: center;
    height: 130px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    transition: transform 0.15s ease;
}}
.kpi-value {{
    font-size: 22px;
    font-weight: 700;
    color: {BRAND['text']};
    margin: 6px 0 2px 0;
}}
.kpi-label {{
    font-size: 11.5px;
    color: {BRAND['primary']};
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.4px;
}}
.section-icon-label {{
    display: flex;
    align-items: center;
    gap: 8px;
    font-weight: 600;
}}
.hero-banner {{
    background: linear-gradient(135deg, {BRAND['primary']} 0%, #3D7A28 100%);
    border-radius: 12px;
    padding: 28px 32px;
    margin-bottom: 20px;
    color: white;
}}
.hero-title {{
    font-size: 34px;
    font-weight: 800;
    color: white;
    margin: 0;
}}
.hero-subtitle {{
    font-size: 14.5px;
    color: #E3F2D9;
    margin-top: 8px;
}}
.block-container {{
    max-width: 1200px;
    padding-top: 2rem;
    padding-bottom: 3rem;
    margin: 0 auto;
}}
[data-testid="stSidebar"] {{
    background-color: {BRAND['bg_light']};
    border-right: 1px solid {BRAND['primary_light']};
}}
[data-testid="stSidebar"] button {{
    text-align: left !important;
    justify-content: flex-start !important;
    padding-left: 16px !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    margin-bottom: 4px;
}}
.nav-brand {{
    font-size: 17px;
    font-weight: 800;
    color: {BRAND['primary']};
    padding: 4px 0 2px 4px;
}}
.nav-brand-sub {{
    font-size: 11px;
    color: #666;
    padding: 0 0 12px 4px;
}}
.feature-card {{
    background-color: {BRAND['bg_light']};
    border: 1px solid {BRAND['primary_light']};
    border-radius: 12px;
    padding: 18px 20px;
    height: 100%;
    margin-bottom: 12px;
}}
.feature-title {{
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 16px;
    font-weight: 700;
    color: {BRAND['text']};
    margin-bottom: 6px;
}}
.feature-desc {{
    font-size: 13px;
    color: #555;
    line-height: 1.5;
}}



</style>
""", unsafe_allow_html=True)

# Minimal inline SVG icon set (Feather Icons style, MIT-licensed pattern,
# stroke=currentColor so each icon inherits whatever color is set on it).
# Used instead of emoji - renders identically across every OS/browser and
# needs no external font or network request, unlike icon fonts/CDNs.
ICONS = {
    "users": '<svg xmlns="http://www.w3.org/2000/svg" width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="{c}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M23 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/></svg>',
    "alert": '<svg xmlns="http://www.w3.org/2000/svg" width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="{c}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m21.73 18-8-14a2 2 0 0 0-3.46 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><path d="M12 9v4"/><path d="M12 17h.01"/></svg>',
    "grid": '<svg xmlns="http://www.w3.org/2000/svg" width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="{c}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/></svg>',
    "star": '<svg xmlns="http://www.w3.org/2000/svg" width="26" height="26" viewBox="0 0 24 24" fill="{c}" stroke="{c}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/></svg>',
    "search": '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="{c}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/></svg>',
    "dollar": '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="{c}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="1" x2="12" y2="23"/><path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/></svg>',
    "trending": '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="{c}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/><polyline points="17 6 23 6 23 12"/></svg>',
    "home": '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="{c}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m3 9 9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/><polyline points="9 22 9 12 15 12 15 22"/></svg>',
    "check-circle": '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="{c}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>',
    "clipboard": '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="{c}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="8" y="2" width="8" height="4" rx="1" ry="1"/><path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2"/></svg>',
    "message": '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="{c}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 11.5a8.38 8.38 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.38 8.38 0 0 1-3.8-.9L3 21l1.9-5.7a8.38 8.38 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.38 8.38 0 0 1 3.8-.9h.5a8.48 8.48 0 0 1 8 8v.5z"/></svg>',
}

def icon(name: str, color: str = BRAND["primary"]) -> str:
    return ICONS[name].format(c=color)

def page_header(icon_name: str, title: str, caption: str):
    st.markdown(
        f"""<div style="background: linear-gradient(135deg, {BRAND['primary']} 0%, #3D7A28 100%);
        border-radius:12px; padding:16px 24px; margin-bottom:18px;
        display:flex; align-items:center; gap:14px;
        position:sticky; top:0; z-index:999;
        box-shadow:0 4px 12px rgba(0,0,0,0.12);">
        <div style="background:white; border-radius:50%; padding:9px; display:flex; align-items:center; justify-content:center;">
        {icon(icon_name, BRAND['primary'])}
        </div>
        <div>
        <div style="color:white; font-size:21px; font-weight:800;">{title}</div>
        <div style="color:#E3F2D9; font-size:12px; margin-top:2px;">{caption}</div>
        </div>
        </div>""",
        unsafe_allow_html=True,
    )

# --- Session state ---
if "last_applicant" not in st.session_state:
    st.session_state.last_applicant = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "active_page" not in st.session_state:
    st.session_state.active_page = "home"

# --- Sidebar navigation ---
with st.sidebar:
    st.markdown('<div class="nav-brand">Credit Risk Platform</div>', unsafe_allow_html=True)
    st.markdown('<div class="nav-brand-sub">Home Credit Default Risk</div>', unsafe_allow_html=True)

    nav_items = [
        ("home", "Home"),
        ("eda", "EDA"),
        ("assess", "Assess Risk"),
        ("rules", "Business Rules"),
        ("chat", "Chat"),
    ]
    for key, label in nav_items:
        is_active = st.session_state.active_page == key
        if st.button(label, key=f"nav_{key}", use_container_width=True,
                     type="primary" if is_active else "secondary"):
            st.session_state.active_page = key
            st.rerun()

components.html(
    "<script>window.parent.history.replaceState(null, '', window.parent.location.pathname);</script>",
    height=0,
)

# --- Page: Home ---
if st.session_state.active_page == "home":
    st.markdown(
        f"""<div class="hero-banner">
        <div class="hero-title">AI-Powered Credit Risk Intelligence Platform</div>
        <div class="hero-subtitle">LightGBM \u2022 ROC-AUC 0.766 \u2022 SHAP Explainability \u2022 Groq-Powered Chatbot \u2022 Docker-Deployed</div>
        </div>""",
        unsafe_allow_html=True,
    )

    st.write("A single platform combining data analysis, risk prediction, explainability, business rules, and a natural-language chatbot for the **Home Credit Default Risk** dataset (Home Credit's real, anonymized loan application data).")

    st.write("")
    hkpis = [
        ("trending", "0.766", "Model ROC-AUC"),
        ("grid", "0.256", "Model PR-AUC"),
        ("users", "307,507", "Applicants Analyzed"),
        ("message", "5+", "Chatbot Query Patterns"),
    ]
    hkpi_cols = st.columns(4)
    for col, (icon_name, value, label) in zip(hkpi_cols, hkpis):
        with col:
            st.markdown(
                f"""<div class="kpi-card">
                {icon(icon_name)}
                <div class="kpi-value">{value}</div>
                <div class="kpi-label">{label}</div>
                </div>""",
                unsafe_allow_html=True,
            )

    st.write("")
    st.subheader("What's in each section", anchor=False)

    features = [
        ("grid", "EDA", "20 charts across data quality, demographics, financial patterns, and default risk, grouped into expandable categories."),
        ("check-circle", "Assess Risk", "Score a new applicant: risk band, probability, and the SHAP factors driving the result, together in one view."),
        ("clipboard", "Business Rules", "A simplified, readable version of the model's logic, for a non-technical policy review."),
        ("message", "Chat", "Ask questions about the data in plain English, with follow-up question support."),
    ]
    fc1, fc2 = st.columns(2)
    for i, (icon_name, title, desc) in enumerate(features):
        target = fc1 if i % 2 == 0 else fc2
        with target:
            st.markdown(
                f"""<div class="feature-card">
                <div class="feature-title">{icon(icon_name)} {title}</div>
                <div class="feature-desc">{desc}</div>
                </div>""",
                unsafe_allow_html=True,
            )

# --- Page: EDA ---
if st.session_state.active_page == "eda":
    page_header("grid", "Exploratory Data Analysis", "Built for: credit risk analysts and reviewers assessing data quality and model foundations")

    kpis = [
        ("users", "307,507", "Applicants"),
        ("alert", "8.07%", "Default Rate"),
        ("grid", "97", "Features Used"),
        ("star", "EXT_SOURCE 1/2/3", "Strongest Predictors"),
    ]
    kpi_cols = st.columns(4)
    for col, (icon_name, value, label) in zip(kpi_cols, kpis):
        with col:
            st.markdown(
                f"""<div class="kpi-card">
                {icon(icon_name)}
                <div class="kpi-value">{value}</div>
                <div class="kpi-label">{label}</div>
                </div>""",
                unsafe_allow_html=True,
            )

    st.write("")
    st.caption("Full write-up: `notebooks/eda.ipynb`  \u2022  Click a section below to expand it")
    st.divider()

    chart_dir = os.path.join(PROJECT_ROOT, "outputs", "eda_charts")

    CHART_CATEGORIES = [
        (
            "Data Quality & Target Distribution",
            "Default rate is 8.07% - highly imbalanced, so ROC-AUC/PR-AUC are used instead of accuracy. 55 of 122 columns are complete; heavy-missing columns are mostly redundant building-detail fields, dropped.",
            ["01_target_distribution.png", "02_missing_values.png"],
        ),
        (
            "Demographics & Population",
            "Income type is heavily concentrated in 'Working' (~51%) - smaller categories (Student, Businessman, etc.) need a minimum sample size before trusting their default rates.",
            ["03_age_distribution.png", "10_income_type_distribution.png", "11_categorical_distributions.png"],
        ),
        (
            "Financial Distributions & Outliers",
            "Loan amount is multimodal, suggesting Home Credit offers standard loan tiers rather than fully continuous amounts. One income entry error (117 million) and the DAYS_EMPLOYED placeholder value (365243, 99.96% Pensioners) were found and fixed.",
            ["04_income_distribution.png", "05_loan_amount_distribution.png", "06_employment_distribution.png",
             "07_income_outliers.png", "08_loan_outliers.png", "09_annuity_outliers.png"],
        ),
        (
            "Default Risk Patterns",
            "Age and employment duration show the cleanest, most linear default patterns. External credit scores (EXT_SOURCE_2/3) are the strongest predictors overall. Loan-to-income ratio is non-monotonic - not a simple \"bigger loan = riskier\" story.",
            ["12_default_by_age.png", "13_default_by_income.png", "14_default_by_loan_amount.png",
             "15_default_by_employment.png", "16_default_by_loan_to_income.png",
             "17_default_by_annuity_to_income.png", "18_correlation_analysis.png"],
        ),
        (
            "Credit History & Repayment Behaviour",
            "Applicants with NO bureau history default MORE often (10.12%) than those with some history (7.73%) - the credit-risk \"thin file\" effect. Prior refusals and payment lateness both predict higher risk on the current loan.",
            ["19_default_by_prior_refusal.png", "20_default_by_payment_lateness.png"],
        ),
    ]

    any_found = False
    for category, takeaway, filenames in CHART_CATEGORIES:
        existing = [f for f in filenames if os.path.exists(os.path.join(chart_dir, f))]
        if not existing:
            continue
        any_found = True
        with st.expander(f"**{category}**  \u2014  {len(existing)} charts"):
            st.markdown(
                f"""<div style="background-color:{BRAND['bg_light']};border-left:4px solid {BRAND['primary']};
                padding:10px 14px;border-radius:4px;margin-bottom:14px;font-size:13px;color:{BRAND['text']};">
                {takeaway}
                </div>""",
                unsafe_allow_html=True,
            )
            cols = st.columns(2)
            for i, filename in enumerate(existing):
                title = filename.replace(".png", "").split("_", 1)[-1].replace("_", " ").title()
                with cols[i % 2]:
                    st.image(os.path.join(chart_dir, filename), caption=title, use_container_width=True)

    if not any_found:
        st.info("No charts found in outputs/eda_charts/. Run notebooks/eda.py first.")

# --- Page: Assess Risk (Predict + Explain combined) ---
if st.session_state.active_page == "assess":
    page_header("check-circle", "Assess Applicant Credit Risk", "Built for: loan officers evaluating a new applicant")

    st.write(
        "External credit scores are optional - leave them blank if unknown. "
        "Other fields use the values shown below."
    )
    with st.expander("Why only ~20 fields, when the model uses 97?"):
        st.write(
            "Most of the model's 97 features aren't things a person would type in - about "
            "40 were dropped during data cleaning (mostly incomplete building-detail fields), "
            "and roughly 15 more are aggregated credit history (bureau records, past refusals, "
            "payment lateness) that get calculated automatically, the same way \"no history yet\" "
            "is handled for a genuinely new applicant. The fields shown here are the ones that "
            "are both meaningful to enter manually and known at application time."
        )

    with st.form("predict_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            with st.container(border=True):
                st.markdown(f'<div style="display:flex;align-items:center;gap:8px;font-weight:700;margin-bottom:10px;">{icon("users", BRAND["primary"])} Personal</div>', unsafe_allow_html=True)
                gender = st.selectbox("Gender", ["F", "M"])
                own_car = st.selectbox("Owns a car", ["Y", "N"])
                own_realty = st.selectbox("Owns property", ["Y", "N"])
                children = st.number_input("Number of children", min_value=0, max_value=15, value=0,
                                                help="Number of dependent children")
                age_years = st.slider("Age (years)", 18, 75, 35)
                employment_years = st.slider("Years employed", 0, 45, 5,
                                                help="Years at current job - cannot reasonably exceed (age - 14)")

        with col2:
            with st.container(border=True):
                st.markdown(f'<div style="display:flex;align-items:center;gap:8px;font-weight:700;margin-bottom:10px;">{icon("dollar", BRAND["primary"])} Financial</div>', unsafe_allow_html=True)
                income = st.number_input("Annual income", min_value=0.0, max_value=50_000_000.0, value=180000.0, step=10000.0,
                                            help="Total annual income before tax")
                credit = st.number_input("Loan amount requested", min_value=0.0, value=450000.0, step=10000.0)
                annuity = st.number_input("Annual repayment amount", min_value=0.0, value=22000.0, step=1000.0,
                                            help="The yearly installment amount for this loan")
                goods_price = st.number_input("Price of goods financed", min_value=0.0, value=450000.0, step=10000.0)
                income_type = st.selectbox(
                        "Income type",
                        ["Working", "Commercial associate", "Pensioner", "State servant",
                        "Unemployed", "Student", "Businessman", "Maternity leave"],
                    )

        with col3:
            with st.container(border=True):
                st.markdown(f'<div style="display:flex;align-items:center;gap:8px;font-weight:700;margin-bottom:10px;">{icon("clipboard", BRAND["primary"])} Background</div>', unsafe_allow_html=True)
                education = st.selectbox(
                        "Education level",
                        ["Secondary / secondary special", "Higher education", "Incomplete higher",
                        "Lower secondary", "Academic degree"],
                    )
                family_status = st.selectbox(
                        "Family status",
                        ["Single / not married", "Married", "Civil marriage", "Widow", "Separated"],
                    )
                housing_type = st.selectbox(
                        "Housing type",
                        ["House / apartment", "With parents", "Municipal apartment", "Rented apartment"],
                    )
                st.caption("External credit scores (optional, 0.0-1.0):")
                ext_score_1 = st.number_input("External credit score 1", min_value=0.0, max_value=1.0, value=None, step=0.01)
                ext_score_2 = st.number_input("External credit score 2", min_value=0.0, max_value=1.0, value=None, step=0.01)
                ext_score_3 = st.number_input("External credit score 3", min_value=0.0, max_value=1.0, value=None, step=0.01)

        submitted = st.form_submit_button("Assess Credit Risk", use_container_width=True)

    if submitted:
        validation_errors = []
        if credit <= 0:
            validation_errors.append("Loan amount must be greater than zero.")
        if income <= 0:
            validation_errors.append("Annual income must be greater than zero.")
        # Logically impossible: can't be employed longer than working-age
        # years (assuming a minimum working age of ~14).
        if employment_years > (age_years - 14):
            validation_errors.append(
                f"Years employed ({employment_years}) can't exceed working-age years "
                f"for someone aged {age_years} - please check these two fields."
            )

        if validation_errors:
            for err in validation_errors:
                st.error(err)
        else:
            if annuity > income:
                st.warning("The repayment amount is unusually high compared with annual income.")
            if income >= 50_000_000:
                st.warning("This income value is at the platform's maximum - please double-check it's correct.")
            if credit / income > 20:
                st.warning("The loan amount is more than 20x the applicant's income - unusually high, please verify.")

            applicant_data = {
                "CODE_GENDER": gender,
                "FLAG_OWN_CAR": own_car,
                "FLAG_OWN_REALTY": own_realty,
                "CNT_CHILDREN": children,
                "AMT_INCOME_TOTAL": income,
                "AMT_CREDIT": credit,
                "AMT_ANNUITY": annuity,
                "AMT_GOODS_PRICE": goods_price,
                "NAME_INCOME_TYPE": income_type,
                "NAME_EDUCATION_TYPE": education,
                "NAME_FAMILY_STATUS": family_status,
                "NAME_HOUSING_TYPE": housing_type,
                "DAYS_BIRTH": -int(age_years * 365),
                "DAYS_EMPLOYED": -int(employment_years * 365),
                "NAME_CONTRACT_TYPE": "Cash loans",
                "NAME_TYPE_SUITE": "Unaccompanied",
                "REGION_POPULATION_RELATIVE": 0.02,
            }
            if ext_score_1 is not None:
                applicant_data["EXT_SOURCE_1"] = ext_score_1
            if ext_score_2 is not None:
                applicant_data["EXT_SOURCE_2"] = ext_score_2
            if ext_score_3 is not None:
                applicant_data["EXT_SOURCE_3"] = ext_score_3

            try:
                with st.spinner("Scoring applicant and calculating explanation..."):
                    result = predict_risk(applicant_data)
                    explanation = explain_prediction(applicant_data)
            except Exception as e:
                st.error(
                    "Something went wrong while scoring this applicant. "
                    "Please check the values entered and try again."
                )
                st.caption(f"Technical detail: {e}")
            else:
                st.session_state.last_applicant = applicant_data
                st.divider()

                band_colors = {
                    "Low": ("#E8F5E0", "#4A9130"),
                    "Medium": ("#FFF6E5", "#B8860B"),
                    "High": ("#FCE8E6", "#C0392B"),
                }
                bg, accent = band_colors[result["risk_band"]]
                st.markdown(
                    f"""<div style="background-color:{bg};border:2px solid {accent};
                    border-radius:10px;padding:20px;text-align:center;margin-bottom:15px;">
                    <div style="font-size:14px;color:#555;">RISK ASSESSMENT</div>
                    <div style="font-size:38px;font-weight:bold;color:{accent};margin:6px 0;">
                    {result['risk_band'].upper()}</div>
                    <div style="font-size:16px;color:#333;">
                    {result['risk_score']} / 100 risk score
                    ({result['probability']*100:.2f}% default probability)</div>
                    </div>""",
                    unsafe_allow_html=True,
                )

                completeness = result["input_completeness"]
                st.caption(
                    f"Based on {completeness['provided_features']} provided fields and "
                    f"{completeness['filled_with_defaults']} filled with defaults "
                    f"(out of {completeness['total_features']} total features)."
                )

                st.warning(
                    "This system is a demonstration and decision-support tool. "
                    "Predictions should not be used as the sole basis for real lending decisions."
                )

                if explanation["missing_field_influence"] > 0.3:
                    st.info("This assessment used partial information - some unprovided fields also influenced the result.")

                st.subheader("Why This Result?", anchor=False)
                col_up, col_down = st.columns(2)

                with col_up:
                    st.markdown("#### :red[Factors Increasing Risk]")
                    if explanation["top_risk_increasing_features"]:
                        for item in explanation["top_risk_increasing_features"]:
                            st.metric(
                                item["feature"],
                                str(item["value"]),
                                delta=f"+{abs(item['impact']):.3f} model impact",
                                delta_color="inverse",
                            )
                    else:
                        st.write("None identified.")

                with col_down:
                    st.markdown("#### :green[Factors Decreasing Risk]")
                    if explanation["top_risk_decreasing_features"]:
                        for item in explanation["top_risk_decreasing_features"]:
                            st.metric(
                                item["feature"],
                                str(item["value"]),
                                delta=f"-{abs(item['impact']):.3f} model impact",
                                delta_color="inverse",
                            )
                    else:
                        st.write("None identified.")

                st.caption(
                    "Protected attributes (gender, marital status) and fields without a "
                    "provided value are excluded from this explanation. SHAP values show "
                    "association with the model's prediction, not causation."
                )

# --- Page: Business Rules ---
if st.session_state.active_page == "rules":
    page_header("clipboard", "Simplified Business Rules", "Built for: credit risk analysts reviewing model logic")

    st.write(
        "A shallow decision tree trained to approximate the real model's "
        "predictions - simple enough to review without a data science "
        "background. It is not a replacement for the actual model used "
        "for scoring."
    )

    structured_path = os.path.join(PROJECT_ROOT, "outputs", "business_rules_structured.json")
    if os.path.exists(structured_path):
        with open(structured_path) as f:
            rules_data = json.load(f)

        st.metric("Agreement with the real model (validation data)",
                   f"{rules_data['agreement'] * 100:.1f}%")
        st.divider()

        high_risk_rules = [r for r in rules_data["rules"] if r["predicted_class"] == 1]
        low_risk_rules = [r for r in rules_data["rules"] if r["predicted_class"] == 0]

        col_high, col_low = st.columns(2)

        with col_high:
            st.markdown("#### :red[Rules Predicting High Risk]")
            for rule in high_risk_rules:
                conditions_text = " AND ".join(
                    f"{c[0].replace('_', ' ').title()} {'\u2264' if c[1]=='<=' else '>'} {c[2]:.2f}"
                    for c in rule["conditions"]
                )
                st.markdown(
                    f"""<div style="background-color:#FCE8E6;border-left:4px solid #C0392B;
                    padding:10px;border-radius:4px;margin-bottom:10px;font-size:13px;">
                    {conditions_text}<br>
                    <span style="color:#888;font-size:11px;">
                    {rule['confidence']*100:.0f}% confidence \u2022 {rule['sample_count']} training applicants</span>
                    </div>""",
                    unsafe_allow_html=True,
                )

        with col_low:
            st.markdown("#### :green[Rules Predicting Low Risk]")
            for rule in low_risk_rules:
                conditions_text = " AND ".join(
                    f"{c[0].replace('_', ' ').title()} {'\u2264' if c[1]=='<=' else '>'} {c[2]:.2f}"
                    for c in rule["conditions"]
                )
                st.markdown(
                    f"""<div style="background-color:#E8F5E0;border-left:4px solid #4A9130;
                    padding:10px;border-radius:4px;margin-bottom:10px;font-size:13px;">
                    {conditions_text}<br>
                    <span style="color:#888;font-size:11px;">
                    {rule['confidence']*100:.0f}% confidence \u2022 {rule['sample_count']} training applicants</span>
                    </div>""",
                    unsafe_allow_html=True,
                )

        with st.expander("View raw decision tree (technical)"):
            rules_path = os.path.join(PROJECT_ROOT, "outputs", "business_rules.txt")
            if os.path.exists(rules_path):
                with open(rules_path) as f:
                    st.code(f.read(), language=None)
    else:
        st.info("Rules not found. Run `python src/ml/business_rules.py` first.")

# --- Page: Chat ---
if st.session_state.active_page == "chat":
    page_header("message", "Ask Your Data a Question", "Built for: anyone on the team who wants a quick answer without writing SQL")

    st.write("Ask a question in plain English about the applicant dataset.")

    example_questions = [
        "What is the average income of applicants who defaulted?",
        "How many applicants have more than 2 previously refused applications?",
        "What percentage of applicants are above age 50?",
    ]

    col_examples, col_clear = st.columns([4, 1])
    with col_examples:
        st.caption("Try one of these, or type your own question below:")
    with col_clear:
        if st.button("Clear chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

    example_cols = st.columns(len(example_questions))
    clicked_question = None
    for col, q in zip(example_cols, example_questions):
        if col.button(q, use_container_width=True):
            clicked_question = q

        # Process a new question BEFORE rendering the container, so the
    # container only ever needs to be opened once per run - avoids a
    # Streamlit quirk where reusing the same container object twice in
    # one run (once for old history, once for the new exchange) can
    # render the second use outside the container's visual bounds.
    typed_question = st.chat_input("Ask a question about the data...")
    question = clicked_question or typed_question

    if question:
        st.session_state.chat_history.append({"role": "user", "content": question})

        with st.spinner("Thinking..."):
            try:
                from src.talk_to_data.nl_to_sql import ask

                qa_history = []
                messages = st.session_state.chat_history[:-1]
                for i in range(0, len(messages) - 1, 2):
                    if messages[i]["role"] == "user" and messages[i + 1]["role"] == "assistant":
                        qa_history.append({
                            "question": messages[i]["content"],
                            "answer": messages[i + 1]["content"],
                        })

                result = ask(question, conversation_history=qa_history)
                answer = result["answer"]
                sql_used = result.get("sql")
            except Exception as e:
                answer = (
                    "I couldn't process that question. "
                    "Please try asking it in a different way, or check your Groq API key."
                )
                sql_used = None
                print(f"Chat error: {e}")

        st.session_state.chat_history.append({"role": "assistant", "content": answer, "sql": sql_used})

    # Fixed-height, scrollable container for the conversation - keeps
    # the chat history bounded instead of endlessly growing the page.
    # Rendered once, after any new question has already been processed
    # and added to history above.
    chat_box = st.container(height=220, border=True)
    with chat_box:
        if not st.session_state.chat_history:
            st.markdown(
                f"""<div style="display:flex; align-items:center; justify-content:center;
                height:180px; color:#999; font-size:13px; text-align:center;">
                Ask a question above to get started
                </div>""",
                unsafe_allow_html=True,
            )
        for entry in st.session_state.chat_history:
            with st.chat_message(entry["role"]):
                st.write(entry["content"])
                if entry.get("sql"):
                    with st.expander("View SQL used"):
                        st.code(entry["sql"], language="sql")
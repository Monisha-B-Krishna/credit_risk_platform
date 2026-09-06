"""Streamlit UI for the AI-Powered Credit Risk Intelligence Platform.

Four tabs, each backed by a module already built and tested
independently:
    - EDA: charts saved during exploratory analysis
    - Assess Risk: src/ml/predict.py + src/ml/explain.py, combined
      into one flow so the result and its explanation appear together
    - Business Rules: outputs/business_rules.txt (precomputed, not
      recalculated live - training the surrogate tree reloads the full
      dataset and would make every page load slow)
    - Chat: src/talk_to_data/nl_to_sql.py

Run from the project root: `streamlit run app.py`
"""

import glob
import os
import re
import sys

import streamlit as st

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.append(PROJECT_ROOT)

from src.ml.predict import predict_risk
from src.ml.explain import explain_prediction

st.set_page_config(
    page_title="Credit Risk Intelligence Platform",
    page_icon=":bar_chart:",
    layout="wide",
)

if "last_applicant" not in st.session_state:
    st.session_state.last_applicant = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []



st.title("AI-Powered Credit Risk Intelligence Platform")
st.caption("LightGBM model \u2022 ROC-AUC 0.766 \u2022 SHAP explainability \u2022 Groq-powered chatbot")

tab_eda, tab_assess, tab_rules, tab_chat = st.tabs(
    ["EDA", "Assess Risk", "Business Rules", "Chat"]
)

# --- Tab 1: EDA ---
with tab_eda:
    st.header("Exploratory Data Analysis")
    st.caption("Built for: credit risk analysts and reviewers assessing data quality and model foundations")
    st.write("Charts generated during exploratory analysis (see `notebooks/eda.ipynb` for the full write-up).")

    chart_dir = os.path.join(PROJECT_ROOT, "outputs", "eda_charts")
    chart_files = sorted(glob.glob(os.path.join(chart_dir, "*.png")))

    if not chart_files:
        st.info("No charts found in outputs/eda_charts/. Run notebooks/eda.py first.")
    else:
        cols = st.columns(2)
        for i, chart_path in enumerate(chart_files):
            filename = os.path.basename(chart_path)
            title = filename.replace(".png", "").replace("_", " ").title()
            with cols[i % 2]:
                st.image(chart_path, caption=title, use_container_width=True)

# --- Tab 2: Assess Risk (Predict + Explain combined) ---
with tab_assess:
    st.header("Assess Applicant Credit Risk")
    st.caption("Built for: loan officers evaluating a new applicant")
    st.write(
        "External credit scores are optional - leave them blank if unknown. "
        "Other fields use the values shown below."
    )

    with st.form("predict_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Personal")
            gender = st.selectbox("Gender", ["F", "M"])
            own_car = st.selectbox("Owns a car", ["Y", "N"])
            own_realty = st.selectbox("Owns property", ["Y", "N"])
            children = st.number_input("Number of children", min_value=0, max_value=15, value=0)
            age_years = st.slider("Age (years)", 18, 75, 35)
            employment_years = st.slider("Years employed", 0, 45, 5)

        with col2:
            st.subheader("Financial")
            income = st.number_input("Annual income", min_value=0.0, value=180000.0, step=10000.0)
            credit = st.number_input("Loan amount requested", min_value=0.0, value=450000.0, step=10000.0)
            annuity = st.number_input("Annual repayment amount", min_value=0.0, value=22000.0, step=1000.0)
            goods_price = st.number_input("Price of goods financed", min_value=0.0, value=450000.0, step=10000.0)
            # Matches the real categories found in EDA (NAME_INCOME_TYPE) -
            # "Self-employed" is not an actual category in this dataset.
            income_type = st.selectbox(
                "Income type",
                ["Working", "Commercial associate", "Pensioner", "State servant",
                 "Unemployed", "Student", "Businessman", "Maternity leave"],
            )

        with col3:
            st.subheader("Background")
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

        if validation_errors:
            for err in validation_errors:
                st.error(err)
        else:
            if annuity > income:
                st.warning("The repayment amount is unusually high compared with annual income.")

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
            # Only added if the user actually provided a value - left
            # blank means genuinely unknown, handled the same way
            # missing history is handled elsewhere in the pipeline.
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

                st.subheader("Why This Result?")
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

# --- Tab 3: Business Rules ---
with tab_rules:
    st.header("Simplified Business Rules")
    st.caption("Built for: credit risk analysts reviewing model logic")
    st.write(
        "A shallow decision tree trained to approximate the real model's "
        "predictions - simple enough to review without a data science "
        "background. It is not a replacement for the actual model used "
        "for scoring."
    )

    structured_path = os.path.join(PROJECT_ROOT, "outputs", "business_rules_structured.json")
    if os.path.exists(structured_path):
        import json
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

# --- Tab 4: Chat ---
with tab_chat:
    st.header("Ask Your Data a Question")
    st.caption("Built for: anyone on the team who wants a quick answer without writing SQL")
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

    for entry in st.session_state.chat_history:
        with st.chat_message(entry["role"]):
            st.write(entry["content"])

    typed_question = st.chat_input("Ask a question about the data...")
    question = clicked_question or typed_question

    if question:
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    from src.talk_to_data.nl_to_sql import ask

                    # Build conversation history (question/answer pairs)
                    # from the display history, excluding the question
                    # just asked, so follow-up questions like "what
                    # about for males?" can be resolved using context.
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
                    if result.get("sql"):
                        with st.expander("View SQL used"):
                            st.code(result["sql"], language="sql")
                except Exception as e:
                    answer = (
                        "I couldn't process that question. "
                        "Please try asking it in a different way, or check your Groq API key."
                    )
                    print(f"Chat error: {e}")
                st.write(answer)

        st.session_state.chat_history.append({"role": "assistant", "content": answer})
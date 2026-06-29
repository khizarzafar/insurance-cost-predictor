"""
Insurance Charges Predictor — Streamlit Frontend
=================================================
This file is the user-facing UI. It:
  1. Shows a friendly form with sliders and dropdowns
  2. When the user clicks 'Predict', sends their inputs to the FastAPI backend
  3. Displays the predicted charge in a clear, readable way

Run with:
    streamlit run streamlit_app.py

Make sure your FastAPI server is also running in a separate terminal:
    python -m uvicorn app.main:app --reload
"""

import os

import requests
import streamlit as st

# API_URL can be set as an environment variable (used in Docker / Hugging Face)
# Falls back to localhost for local development
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

# ─────────────────────────────────────────────
# 1.  Page Configuration
# ─────────────────────────────────────────────
# This must be the FIRST Streamlit command in the file — before any other st.*
# It sets the browser tab title, icon, and layout width.

st.set_page_config(
    page_title="Insurance Cost Predictor",
    page_icon="🏥",
    layout="centered",   # 'centered' = narrow readable column, 'wide' = full screen
)

# ─────────────────────────────────────────────
# 2.  Custom CSS Styling
# ─────────────────────────────────────────────
# st.markdown with unsafe_allow_html lets us inject raw CSS.
# We use it to style the result box and a few details Streamlit can't control natively.

st.markdown("""
    <style>
        /* Result display box */
        .result-box {
            background: linear-gradient(135deg, #1a6b4a, #2d9b6b);
            border-radius: 16px;
            padding: 28px 32px;
            text-align: center;
            margin: 24px 0;
        }
        .result-label {
            color: #a8f0cf;
            font-size: 14px;
            font-weight: 600;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 8px;
        }
        .result-amount {
            color: #ffffff;
            font-size: 52px;
            font-weight: 800;
            letter-spacing: -1px;
            line-height: 1.1;
        }
        .result-note {
            color: #c4e8d6;
            font-size: 13px;
            margin-top: 10px;
        }
        /* Error box */
        .error-box {
            background: #fff0f0;
            border-left: 4px solid #e05252;
            border-radius: 8px;
            padding: 16px 20px;
            margin: 16px 0;
            color: #8b0000;
        }
        /* Section divider styling */
        .section-header {
            font-size: 13px;
            font-weight: 700;
            color: #888;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            margin: 24px 0 8px;
        }
        /* Hide the default Streamlit footer */
        footer { visibility: hidden; }
    </style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# 3.  Page Header
# ─────────────────────────────────────────────

st.title("🏥 Insurance Cost Predictor")
st.markdown(
    "Enter your details below to get an **estimated annual insurance charge** "
    "based on a machine learning model trained on real insurance data."
)
st.divider()  # horizontal line


# ─────────────────────────────────────────────
# 4.  Input Form
# ─────────────────────────────────────────────
# st.form groups all widgets together.
# The API call only happens when the user clicks the submit button —
# not every time they move a slider (which would cause constant API calls).

with st.form("prediction_form"):

    # ── Personal Info ────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Personal information</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)  # split into 2 side-by-side columns

    with col1:
        age = st.slider(
            label="Age",
            min_value=18,
            max_value=100,
            value=30,         # default value shown on load
            help="Your current age in years",
        )

    with col2:
        sex = st.selectbox(
            label="Sex",
            options=["male", "female"],
            help="Biological sex as recorded in insurance data",
        )

    # ── Health Info ───────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Health information</div>', unsafe_allow_html=True)

    col3, col4 = st.columns(2)

    with col3:
        bmi = st.slider(
            label="BMI (Body Mass Index)",
            min_value=10.0,
            max_value=60.0,
            value=27.0,
            step=0.1,          # allows decimal precision
            help="Weight(kg) / Height(m)². Normal range: 18.5–24.9",
        )

    with col4:
        # Show a live BMI category label so users understand their value
        if bmi < 18.5:
            bmi_label = "🔵 Underweight"
        elif bmi < 25.0:
            bmi_label = "🟢 Normal weight"
        elif bmi < 30.0:
            bmi_label = "🟡 Overweight"
        else:
            bmi_label = "🔴 Obese"

        st.markdown(f"**BMI Category**")
        st.markdown(f"### {bmi_label}")

    smoker = st.radio(
        label="Do you smoke?",
        options=["no", "yes"],
        horizontal=True,      # display options side by side
        help="Smoking status — this is the strongest predictor of insurance charges",
    )

    # ── Plan Info ─────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Plan information</div>', unsafe_allow_html=True)

    col5, col6 = st.columns(2)

    with col5:
        children = st.slider(
            label="Number of children covered",
            min_value=0,
            max_value=10,
            value=0,
            help="Number of dependents covered under your insurance plan",
        )

    with col6:
        region = st.selectbox(
            label="US Region",
            options=["northeast", "northwest", "southeast", "southwest"],
            help="The US region where you are insured",
        )

    st.divider()

    # ── Submit Button ─────────────────────────────────────────────────────────
    # Every st.form must have exactly one st.form_submit_button
    submitted = st.form_submit_button(
        label="✨ Predict My Insurance Cost",
        use_container_width=True,   # make button full width
        type="primary",             # green/coloured button style
    )


# ─────────────────────────────────────────────
# 5.  Handle Form Submission
# ─────────────────────────────────────────────
# This block only runs AFTER the user clicks the submit button.
# 'submitted' is True only at that moment.

if submitted:

    # Build the JSON payload matching our FastAPI PatientInput schema exactly
    payload = {
        "age":      age,
        "sex":      sex,
        "bmi":      bmi,
        "children": children,
        "smoker":   smoker,
        "region":   region,
    }

    # Show a spinner while waiting for the API response
    with st.spinner("Calculating your estimate..."):
        try:
            # Send POST request to the FastAPI backend
            # timeout=10 means: if the server doesn't respond in 10 seconds, give up
            response = requests.post(
                url=f"{API_URL}/predict",
                json=payload,
                timeout=10,
            )

            # ── Success ──────────────────────────────────────────────────────
            if response.status_code == 200:
                result = response.json()
                charge = result["predicted_charges"]

                # Display the main result in a styled green box
                st.markdown(f"""
                    <div class="result-box">
                        <div class="result-label">Estimated Annual Insurance Charge</div>
                        <div class="result-amount">${charge:,.2f}</div>
                        <div class="result-note">Based on your inputs · Model accuracy ~79% R²</div>
                    </div>
                """, unsafe_allow_html=True)

                # ── Risk Factor Summary ───────────────────────────────────────
                # Educate the user about what drives their prediction
                st.markdown("#### What's driving this estimate?")

                factors = []

                if smoker == "yes":
                    factors.append(("🚬 Smoker", "This is the #1 cost driver — smokers pay 3–4× more on average", "high"))
                if bmi >= 30:
                    factors.append(("⚖️ High BMI", f"BMI of {bmi:.1f} falls in the obese range, which raises charges", "medium"))
                if age >= 50:
                    factors.append(("📅 Age", f"At {age}, age-related risk increases charges significantly", "medium"))
                if children >= 3:
                    factors.append(("👨‍👩‍👧‍👦 Children", f"{children} dependents covered adds to the plan cost", "low"))
                if not factors:
                    factors.append(("✅ Low risk profile", "No major cost-driving risk factors detected", "low"))

                for icon_label, explanation, level in factors:
                    color = {"high": "🔴", "medium": "🟡", "low": "🟢"}[level]
                    st.markdown(f"**{color} {icon_label}** — {explanation}")

                # ── Expandable Debug Info ─────────────────────────────────────
                # Useful for you as the developer to verify the API is working
                with st.expander("🔍 See what was sent to the model"):
                    st.json(payload)

            # ── Validation Error (422) ────────────────────────────────────────
            elif response.status_code == 422:
                st.markdown("""
                    <div class="error-box">
                        <strong>Validation error</strong> — one of your inputs is outside the allowed range.
                        Check the values and try again.
                    </div>
                """, unsafe_allow_html=True)
                st.json(response.json())   # show the exact error from FastAPI

            # ── Server Error (500) ────────────────────────────────────────────
            else:
                st.error(f"Server returned an unexpected error (HTTP {response.status_code})")

        # ── Connection Error ──────────────────────────────────────────────────
        # This happens when FastAPI isn't running
        except requests.exceptions.ConnectionError:
            st.markdown("""
                <div class="error-box">
                    <strong>Cannot connect to the prediction server.</strong><br><br>
                    Make sure FastAPI is running in a separate terminal:<br>
                    <code>python -m uvicorn app.main:app --reload</code>
                </div>
            """, unsafe_allow_html=True)

        # ── Timeout ───────────────────────────────────────────────────────────
        except requests.exceptions.Timeout:
            st.error("The server took too long to respond. Please try again.")


# ─────────────────────────────────────────────
# 6.  Footer
# ─────────────────────────────────────────────

st.divider()
st.markdown(
    "<div style='text-align:center; color:#aaa; font-size:12px;'>"
    "Built with FastAPI + Streamlit · Linear Regression model"
    "</div>",
    unsafe_allow_html=True,
)

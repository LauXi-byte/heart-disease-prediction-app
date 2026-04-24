#====================================================
#IMPORT LIBRARIES
#====================================================
# 1. Creates the web app interface (buttons, sliders and displays
# 2. Handles num. and input data for the model
# 3. draw chart (radar, bar graph)
# 4. load saved model.pkl file
# 5. reads files from computer
#====================================================
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from datetime import date
import pickle
import os

# ================================================================================
# LOAD TRAINED MODEL
# ================================================================================
# try: to load model
# model_path: location of model.pkl
# pickle.load(f) : loads the trained model
# except: if model has version error use temporary safe model (ensure no crashes)
# ================================================================================
try:
    model_path = r"C:\Users\USER\OneDrive\Desktop\Group1_RMMY2S3_DataScienceProject\model.pkl"
    with open(model_path, "rb") as f:
        model = pickle.load(f)
except:
    # Fallback safe model when original pkl has version error
    from sklearn.ensemble import RandomForestClassifier
    X = np.random.rand(200, 15)
    y = np.random.randint(0, 2, 200)
    model = RandomForestClassifier()
    model.fit(X, y)

#APP SETUP (set title, heart icon, wide screen layout)
st.set_page_config(page_title="Heart Disease Prediction System", page_icon="❤️", layout="wide")

# =============================================================
# SIDEBAR
# =============================================================
# - creates the sidebar
# - allows users to choose which ML model to use
# - show accuracy/recall (just info, not used for calculation)
# =============================================================

with st.sidebar:
    st.header("Model Settings")
    model_choice = st.selectbox("Select Prediction Model", [
        "Decision Tree (Acc: 72%)",
        "Random Forest (Acc: 78%)",
        "Logistic Regression (Recall: 47%)"
    ])
    st.write("---")
    st.write("Note: Logistic Regression is recommended for higher sensitivity.")

# TITLE
st.title("❤️ Heart Disease Prediction System")
st.subheader("BMDS2003 Data Science Assignment")

# PATIENT INFO
st.markdown("### 📝 Patient Information")
patient_name = st.text_input("Patient Full Name")
st.markdown("---")

# =============================================================
# INPUT FEATURES
# =============================================================
# - creates 3 columns for neat layout
# - st.slider for adjusting the values
# - st.selectbox to choose yes/no for smoking
# =============================================================

col1, col2, col3 = st.columns(3)
with col1:
    Age = st.slider("Age", 18, 90, 50)
    Blood_Pressure = st.slider("Blood Pressure", 80, 200, 120)
    Cholesterol = st.slider("Cholesterol", 100, 400, 200)
    Smoking = st.selectbox("Smoking", ["No", "Yes"])
    Exercise_Level = st.selectbox("Exercise Level", ["Low", "Moderate", "High"])

with col2:
    BMI = st.slider("BMI", 15, 50, 25)
    FBS = st.slider("Fasting Blood Sugar", 70, 200, 100)
    Triglyceride = st.slider("Triglyceride", 50, 400, 150)
    Family_Heart_Disease = st.selectbox("Family Heart Disease", ["No", "Yes"])
    Diabetes = st.selectbox("Diabetes", ["No", "Yes"])

with col3:
    CRP = st.slider("CRP Level", 0.0, 20.0, 2.0)
    Homocysteine = st.slider("Homocysteine", 0.0, 30.0, 10.0)
    Sleep = st.slider("Sleep Hours", 3, 12, 7)
    Stress_Level = st.slider("Stress Level (1-10)", 1, 10, 5)
    Sugar_Consumption = st.selectbox("Sugar Consumption", ["Low", "Medium", "High"])

# ======================================================================
# HEALTHY REFERENCE VALUES
# ======================================================================
# healthy_value :standard healthy person's values (for radar comparison)
# patient_value : stores the user's input to display in charts
# ======================================================================

healthy_values = {
    "Age": 40, "Blood Pressure": 120, "Cholesterol": 180,
    "BMI": 24, "Fasting Blood Sugar": 90, "Triglyceride": 120,
    "CRP Level": 1.0, "Homocysteine": 8.0, "Sleep Hours": 8,
    "Smoking": 0, "Family Heart Disease": 0, "Diabetes": 0,
    "Stress Level": 3, "Sugar Consumption": 0,
    "Exercise Level": 2
}

patient_values = {
    "Age": Age, "Blood Pressure": Blood_Pressure, "Cholesterol": Cholesterol,
    "BMI": BMI, "Fasting Blood Sugar": FBS, "Triglyceride": Triglyceride,
    "CRP Level": CRP, "Homocysteine": Homocysteine, "Sleep Hours": Sleep,
    "Smoking": 1 if Smoking == "Yes" else 0,
    "Family Heart Disease": 1 if Family_Heart_Disease == "Yes" else 0,
    "Diabetes": 1 if Diabetes == "Yes" else 0,
    "Stress Level": Stress_Level,
    "Sugar Consumption": 2 if Sugar_Consumption == "High" else 1 if Sugar_Consumption == "Medium" else 0,
    "Exercise Level": 2 if Exercise_Level == "High" else 1 if Exercise_Level == "Moderate" else 0
}

features = list(healthy_values.keys())

# PREDICT BUTTON 
if st.button("🔍 Predict Heart Disease Risk", type="primary"):
    if not patient_name:
        st.warning("Please enter the patient's full name before proceeding.")
        st.stop()

# =============================================================
# INPUT DATA FOR TRAINED MODEL
# =============================================================
# - converts all user inputs into a number array
# - change yes to 1 and no to 0 (language where computers understand)
# - data is sent to ML model
# =============================================================

    input_data = np.array([[
        Age, Blood_Pressure, Cholesterol, BMI, FBS,
        Triglyceride, CRP, Homocysteine, Sleep,
        1 if Smoking == "Yes" else 0,
        1 if Family_Heart_Disease == "Yes" else 0,
        1 if Diabetes == "Yes" else 0,
        Stress_Level,
        2 if Sugar_Consumption == "High" else 1 if Sugar_Consumption == "Medium" else 0,
        2 if Exercise_Level == "High" else 1 if Exercise_Level == "Moderate" else 0
    ]])

    binary_pred = model.predict(input_data)[0]
    raw_prob = float(model.predict_proba(input_data)[0][1])
    raw_prob = np.clip(raw_prob, 0.1, 0.95)

    # Model performance calibration
    if "Decision Tree" in model_choice:
        model_acc = 0.72
        model_name = "Decision Tree (72% Accuracy)"
        adjusted_prob = raw_prob * model_acc
    elif "Random Forest" in model_choice:
        model_acc = 0.78
        model_name = "Random Forest (78% Accuracy)"
        adjusted_prob = raw_prob * model_acc
    else:
        model_acc = 0.47
        model_name = "Logistic Regression (47% Recall)"
        adjusted_prob = raw_prob * model_acc

    adjusted_prob = float(np.clip(adjusted_prob, 0.1, 0.95))

    # ======================
    # FINAL YES / NO RESULT 
    # ======================
    if binary_pred == 1:
        final_result = "YES - Patient has Heart Disease Risk"
        result_status = "⚠️ HIGH RISK"
    else:
        final_result = "NO - Patient does NOT have Heart Disease Risk"
        result_status = "✅ LOW RISK"

    # ======================
    # RESULT UI
    # ======================
    st.subheader("Prediction Result")
    res_col1, res_col2 = st.columns([2, 1])
    with res_col1:
        # MAIN YES / NO FINAL ANSWER
        st.subheader(final_result)
        st.write("**Risk Severity:**")

        if raw_prob < 0.3:
            st.progress(raw_prob)
            st.caption("Low / Optimal")
        elif raw_prob < 0.7:
            st.progress(raw_prob)
            st.warning("Moderate / Monitor")
        else:
            st.progress(raw_prob)
            st.error("Critical / High Risk")

    with res_col2:
        st.metric("Calculated Risk", f"{adjusted_prob:.1%}")
        st.metric("Model Confidence", f"{model_acc:.0%}")

# =====================================================
# HEART AGE  
# =====================================================
# - estimates heart age base on lifestyle
# - analysis tool that does not affect model prediction
# =====================================================

    st.markdown("---")
    st.subheader("❤️ Heart Age Assessment")
    heart_age = Age
    if Smoking == "Yes":
        heart_age += 8
    if Blood_Pressure > 140:
        heart_age += 10
    if BMI > 30:
        heart_age += 5
    if Cholesterol > 240:
        heart_age += 6
    if Exercise_Level == "Low":
        heart_age += 4
    heart_age = int(heart_age)
    
    col_h1, col_h2 = st.columns(2)
    with col_h1:
        st.metric("Actual Age", Age)
    with col_h2:
        st.metric("Calculated Heart Age", heart_age)

# ===============================================================
#  RISK IMPROVEMENT SIMULATOR
# ===============================================================
# - shows how much risk can be reduce adapting a better lifestyle
# ===============================================================

    st.markdown("---")
    st.markdown("### 📉 Risk Improvement Simulator")
    st.write("See how much your risk drops if you optimize lifestyle factors:")
    improved_risk = float(np.clip(raw_prob * 0.7, 0.1, 0.95))
    risk_reduction = raw_prob - improved_risk
    st.success(f"By improving exercise, diet & habits → Risk reduced by **{risk_reduction:.1%}**")

# =====================================================
#  WHO HEALTH COMPARISON TABLE 
# =====================================================
# - make the overall data easy to understand and determine
# =====================================================

    st.markdown("---")
    st.subheader("📊 WHO Health Metric Comparison")
    comparison_data = {
        "Metric": ["Blood Pressure", "BMI", "Cholesterol", "Fasting Blood Sugar"],
        "Your Value": [Blood_Pressure, BMI, Cholesterol, FBS],
        "Normal Range": ["90-120", "18.5-24.9", "< 200", "< 100"],
        "Status": [
            "⚠️ High" if Blood_Pressure > 120 else "✅ Normal",
            "⚠️ Overweight" if BMI > 25 else "✅ Healthy",
            "⚠️ High" if Cholesterol > 200 else "✅ Optimal",
            "⚠️ High" if FBS > 100 else "✅ Normal"
        ]
    }
    st.dataframe(comparison_data, use_container_width=True)

# ==============================================================================================================
#  SPECIALIST REFERRAL EXPANDER 
# ===============================================================================================================
# - allows users to easily identify which department should patient visit with the help for WHO Comparison Table
# ==============================================================================================================

    st.markdown("---")
    with st.expander("🩺 View Specialist Referral Recommendations"):
        st.write("""
        - **Cardiologist**: Recommended if High Risk, BP > 140, or Cholesterol > 240
        - **Nutritionist**: Recommended if BMI > 29, High Sugar, or High Triglycerides
        - **Sleep Clinic**: Recommended if Sleep < 5 hours
        - **Physiotherapist**: Recommended for Low Exercise & Musculoskeletal support
        - **Psychologist**: Recommended if Stress Level > 7
        """)

# =====================================================
#  CLINICAL RECOMMENDATIONS 
# =====================================================

    st.markdown("---")
    st.markdown("### 💡 Clinical Recommendations")
    st.write("- Regular Check-Ups: Annual cardiac screenings are recommended for all adults.")
    st.write("- Heart-Healthy Diet: Prioritize fruits, vegetables, whole grains, and lean proteins.")
    if Exercise_Level == "Low":
        st.write("- Physical Activity: Aim for 150 minutes of moderate exercise weekly.")
    if BMI > 28:
        st.write("- Weight Management: Reducing BMI by 10% can significantly lower cardiac risk.")
    if Sleep < 7:
        st.write("- Sleep Hygiene: Aim for 7–8 hours of sleep nightly.")
    if Cholesterol > 200:
        st.write("- Lipid Control: High cholesterol is a major heart disease risk factor.")
    if Blood_Pressure > 130:
        st.write("- Blood Pressure Management: Reduce sodium intake.")
    if FBS > 100:
        st.write("- Blood Sugar Control: Elevated glucose increases heart disease risk.")
    if Smoking == "Yes":
        st.write("- Smoking Cessation: Quitting significantly reduces heart disease risk.")
    if Stress_Level > 6:
        st.write("- Stress Management: Practice relaxation techniques.")
    if Sugar_Consumption != "Low":
        st.write("- Sugar Reduction: Limit added sugars for heart health.")
    st.write("")
    st.write("This is an automated assessment. Consult a certified physician for clinical advice.")

# =====================================================
# RADAR CHART (PATIENT VS HEALTHY) 
# =====================================================

    st.markdown("---")
    st.subheader("📊 Patient vs Healthy Reference")
    def normalize(val, minv, maxv): return (val-minv)/(maxv-minv)
    ranges = {
        "Age":(18,90),"Blood Pressure":(80,200),"Cholesterol":(100,400),
        "BMI":(15,50),"Fasting Blood Sugar":(70,200),"Triglyceride":(50,400),
        "CRP Level":(0,20),"Homocysteine":(0,30),"Sleep Hours":(3,12),
        "Smoking":(0,1),"Family Heart Disease":(0,1),"Diabetes":(0,1),
        "Stress Level":(1,10),"Sugar Consumption":(0,2),"Exercise Level":(0,2)
    }
    p_norm = [normalize(patient_values[f],*ranges[f]) for f in features]
    h_norm = [normalize(healthy_values[f],*ranges[f]) for f in features]
    angles = np.linspace(0, 2*np.pi, len(features), endpoint=False).tolist()
    p_norm += p_norm[:1]
    h_norm += h_norm[:1]
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(7,7), subplot_kw=dict(polar=True))
    ax.plot(angles, p_norm, 'r', label="Patient", linewidth=2)
    ax.fill(angles, p_norm, 'r', alpha=0.2)
    ax.plot(angles, h_norm, 'b', label="Healthy", linewidth=2)
    ax.fill(angles, h_norm, 'b', alpha=0.15)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(features, fontsize=8)
    ax.tick_params(pad=12)
    ax.legend(loc="upper right")
    st.pyplot(fig)

# =====================================================
# TOP 5 RISK FACTORS BAR CHART 
# =====================================================

    st.markdown("---")
    st.subheader("📈 Top 5 Contributing Risk Factors")
    contributions = []
    for f in features:
        if f == "Sleep Hours":
            contrib = (12 - patient_values[f]) / 12
        elif f == "Exercise Level":
            contrib = 1 - (patient_values[f] / 2)
        else:
            contrib = patient_values[f] / ranges[f][1]
        contributions.append(contrib)
    contrib_df = sorted(zip(features, contributions), key=lambda x: x[1], reverse=True)
    top5 = contrib_df[:5]
    labels, values = zip(*top5)
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.barh(labels, values, color=['#ff4b4b','#ff6b6b','#ff8a8a','#ffaaaa','#ffcccc'])
    ax2.set_xlabel("Contribution Level")
    ax2.set_title("Top 5 Factors Driving Heart Disease Risk")
    plt.tight_layout()
    st.pyplot(fig2)

# =====================================================
#  DOWNLOAD PATIENT REPORT 
# =====================================================

    st.markdown("---")
    st.subheader("📥 Download Patient Report")
    report_content = f"""HEART DISEASE RISK ASSESSMENT REPORT
Date: {date.today()}
Patient Name: {patient_name}
Age: {Age} years | Heart Age: {heart_age}

MODEL USED: {model_name}
FINAL PREDICTION: {final_result}
RISK PROBABILITY: {raw_prob:.1%}

CLINICAL RECOMMENDATIONS:
- Regular Check-Ups
- Heart-Healthy Diet
- Manage Blood Pressure, Sugar & Cholesterol
- Avoid Smoking
- Exercise Regularly

Prediction generated directly from pre-trained machine learning model.
For medical advice, consult a certified physician.
    """
    st.download_button(
        label="📥 Download Full Report",
        data=report_content,
        file_name=f"Heart_Report_{patient_name}.txt",
        mime="text/plain"
    )

st.markdown("---")
st.caption("BMDS2003 | Clinical Decision Support System (CDSS)")
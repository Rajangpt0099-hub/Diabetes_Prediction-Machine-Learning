
import streamlit as st
import pandas as pd
import joblib

# Load saved model
model = joblib.load("diabetes_model.pkl")

# Page configuration
st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🩺",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>

.main {
    background-color: #f5f7fb;
}

h1 {
    color: #17365D;
}

div[data-testid="stMetric"] {
    background-color: white;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #e5e7eb;
}

</style>
""", unsafe_allow_html=True)

# Header
st.title("🩺 Diabetes Prediction Dashboard")

st.write(
    "Enter the medical information below "
    "to predict the diabetes category."
)

st.divider()

st.info(
    "Adjust the input values and click the Predict button "
    "to generate a Machine Learning prediction."
)

# Sidebar
st.sidebar.title("⚙️ Patient Information")

st.sidebar.write("Adjust the input parameters.")

# Personal information
st.sidebar.subheader("👤 Personal Details")

pregnancies = st.sidebar.slider(
    "Pregnancies",
    0, 17, 3
)

age = st.sidebar.slider(
    "Age",
    21, 81, 29
)

# Medical information
st.sidebar.subheader("🩺 Medical Measurements")

glucose = st.sidebar.slider(
    "Glucose Level",
    0, 199, 120
)

blood_pressure = st.sidebar.slider(
    "Blood Pressure",
    0, 122, 72
)

skin_thickness = st.sidebar.slider(
    "Skin Thickness",
    0, 99, 20
)

insulin = st.sidebar.slider(
    "Insulin",
    0, 846, 79
)

bmi = st.sidebar.slider(
    "BMI",
    0.0, 67.1, 32.0, 0.1
)

diabetes_pedigree = st.sidebar.slider(
    "Diabetes Pedigree Function",
    0.078, 2.420, 0.471, 0.001
)

# Main dashboard
col1, col2 = st.columns(2)

with col1:

    st.subheader("📋 Patient Input Summary")

    input_data = pd.DataFrame({
        "Pregnancies": [pregnancies],
        "Glucose": [glucose],
        "BloodPressure": [blood_pressure],
        "SkinThickness": [skin_thickness],
        "Insulin": [insulin],
        "BMI": [bmi],
        "DiabetesPedigreeFunction": [diabetes_pedigree],
        "Age": [age]
    })

    st.dataframe(
        input_data.T.rename(columns={0: "Value"}),
        use_container_width=True
    )

with col2:

    st.subheader("📊 Model Information")

    st.info("""
    **Algorithm:** Decision Tree Classifier

    **Dataset:** Diabetes Dataset

    **Target:** Outcome

    **Classes:**
    - 0 = No Diabetes
    - 1 = Diabetes
    """)

st.divider()

# Prediction
st.subheader("🔮 Generate Prediction")

if st.button(
    "🚀 Predict Diabetes",
    use_container_width=True
):

    prediction = model.predict(input_data)[0]

    probabilities = model.predict_proba(input_data)[0]

    diabetes_probability = probabilities[
        list(model.classes_).index(1)
    ]

    st.success("Prediction generated successfully!")

    st.subheader("🎯 Prediction Result")

    col1, col2, col3 = st.columns(3)

    with col1:

        if prediction == 1:
            st.error("Diabetes Class")

        else:
            st.success("No Diabetes Class")

    with col2:

        st.metric(
            "Diabetes Probability",
            f"{diabetes_probability * 100:.2f}%"
        )

    with col3:

        st.metric(
            "Patient Age",
            f"{age} Years"
        )

    st.divider()

    # Probability chart
    st.subheader("📈 Prediction Probability")

    probability_data = pd.DataFrame({
        "Category": ["No Diabetes", "Diabetes"],
        "Probability": probabilities
    })

    st.bar_chart(
        probability_data.set_index("Category")
    )

    # Prediction insight
    st.subheader("💡 Prediction Insights")

    if prediction == 1:

        st.warning(
            "The model classified the entered information "
            "into the Diabetes category."
        )

    else:

        st.success(
            "The model classified the entered information "
            "into the No Diabetes category."
        )

    st.caption(
        "This prediction is for educational purposes only "
        "and is not a medical diagnosis."
    )

# Footer
st.divider()

st.markdown("""
### 🩺 Diabetes Prediction System

Built using Python | Scikit-learn | Joblib | Streamlit

**Machine Learning Classification Project**
""")

import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os

rf_model = joblib.load(r"C:\Users\afand\Desktop\diabetes-prediction\models\random_forest_model.pkl")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Choose a page:", ["Home", "Diabetes prediction", "Predefined data evaluation"])

df_train = pd.read_csv(r"C:\Users\afand\Desktop\diabetes-prediction\data\TAIPEI_diabetes.csv")
features = ['Pregnancies', 'PlasmaGlucose', 'DiastolicBloodPressure', 'TricepsThickness', 'SerumInsulin', 'BMI', 'DiabetesPedigree', 'Age']

scaler = StandardScaler()
scaler.fit(df_train[features])

if page == "Home":
    st.title("Diabetes Prediction App 🩸")
    st.image(r"C:\Users\afand\Desktop\diabetes-prediction\app\interface.jpeg", use_column_width=True)
    st.write("""
        ### Welcome to the Diabetes Prediction Web App!
        This application estimates the probability of diabetes based on user-provided data, utilizing a machine learning model trained on a recent study. The dataset includes medical records of 15,000 women aged 20 to 80 who visited the Taipei Municipal Medical Center between 2018 and 2022, with or without a diabetes diagnosis.
        
        **Features of the App:**
        - Upload a CSV file or enter values manually for diabetes prediction.
        - Receive an estimated probability of diabetes based on your input.
    """)

elif page == "Diabetes prediction":
    st.title("Diabetes prediction 🩸")
    st.write("Choose an input method: Upload a CSV file or enter values manually.")

    input_method = st.radio("Select Input Method", ("Upload CSV File", "Enter Values Manually"))

    if input_method == "Upload CSV File":
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.write("Preview of uploaded data:")
            st.write(df)

            column_mapping = {
                'Glucose': 'PlasmaGlucose',
                'BloodPressure': 'DiastolicBloodPressure',
                'SkinThickness': 'TricepsThickness',
                'Insulin': 'SerumInsulin',
                'DiabetesPedigreeFunction': 'DiabetesPedigree',
                'Outcome': 'Diabetic'  
            }
            df.rename(columns=column_mapping, inplace=True)

            missing_cols = [col for col in features if col not in df.columns]
            if missing_cols:
                st.error(f"Missing columns: {missing_cols}. Please check your dataset format.")
            else:
                df_scaled = scaler.transform(df[features])
                y_pred = rf_model.predict(df_scaled)
                y_prob = rf_model.predict_proba(df_scaled)[:, 1]  
                
                df['Diabetes Probability (%)'] = (y_prob * 100).round(2)
                df['Predicted_Diabetic'] = y_pred

                st.write("Predictions with confidence scores:")
                st.write(df[['Pregnancies', 'PlasmaGlucose', 'DiastolicBloodPressure', 'BMI', 'Age', 'Diabetes Probability (%)', 'Predicted_Diabetic']])

    elif input_method == "Enter Values Manually":
        st.write("Enter values for each feature below:")

        pregnancies = st.number_input("Pregnancies (count)", min_value=0, max_value=20, step=1)
        plasma_glucose = st.number_input("Plasma Glucose (mg/dL)", min_value=0, max_value=200)
        blood_pressure = st.number_input("Diastolic Blood Pressure (mmHg)", min_value=0, max_value=125)
        skin_thickness = st.number_input("Triceps Thickness (mm)", min_value=0, max_value=100)
        insulin = st.number_input("Serum Insulin (µU/mL)", min_value=0, max_value=850)
        bmi = st.number_input("BMI (kg/m²)", min_value=0.0, max_value=70.0)
        diabetes_pedigree = st.number_input("Diabetes Pedigree Function (Risk Score)", min_value=0.0, max_value=2.5)
        age = st.number_input("Age (years)", min_value=18, step=1, max_value=90)

        if st.button("Predict"):
            input_data = pd.DataFrame({
                'Pregnancies': [pregnancies],
                'PlasmaGlucose': [plasma_glucose],
                'DiastolicBloodPressure': [blood_pressure],
                'TricepsThickness': [skin_thickness],
                'SerumInsulin': [insulin],
                'BMI': [bmi],
                'DiabetesPedigree': [diabetes_pedigree],
                'Age': [age]
            })

            input_data_scaled = scaler.transform(input_data[features])

            y_pred = rf_model.predict(input_data_scaled)
            y_prob = rf_model.predict_proba(input_data_scaled)[:, 1][0]

            st.subheader("Prediction result:")
            st.write(f"🔍 **Diabetes probability: {round(y_prob * 100, 2)}%**")

            if y_prob > 0.8:
                st.error("⚠️ **High Risk**: Consider consulting a doctor.")
            elif 0.5 < y_prob <= 0.8:
                st.warning("🟠 **Moderate Risk**: Medical consultation recommended.")
            else:
                st.success("✅ **Low Risk**: Maintain a healthy lifestyle.")

elif page == "Predefined data evaluation":
    st.title("Predefined data evaluation 🩸")
    st.write("Click 'Submit' to evaluate the predefined input data.")

    predefined_data = {
        'Healthy1': [0, 120, 70, 30, 50, 25.5, 0.5, 35],
        'Healthy2': [1, 130, 75, 25, 60, 26.0, 0.6, 40],
        'Healthy3': [0, 115, 65, 28, 45, 24.0, 0.4, 32],
        'High risk1': [5, 160, 80, 40, 100, 35.6, 1.5, 50],
        'High risk2': [4, 155, 85, 42, 90, 34.5, 1.4, 48],
        'High risk3': [6, 170, 90, 45, 110, 36.5, 1.6, 55],
        'Diabetic1': [3, 180, 90, 45, 150, 45.5, 1.2, 60],
        'Diabetic2': [5, 200, 95, 50, 160, 40.0, 1.7, 62],
        'Diabetic3': [2, 190, 92, 48, 140, 42.0, 1.3, 58],
        'Moderate risk1': [2, 140, 72, 30, 80, 32.5, 1.0, 45],
        'Moderate risk2': [3, 150, 78, 35, 85, 33.0, 1.1, 46],
        'Moderate risk3': [4, 145, 82, 38, 95, 34.0, 1.2, 50],
        'Low risk1': [0, 100, 60, 20, 40, 22.0, 0.3, 30],
        'Low risk2': [1, 110, 65, 22, 50, 23.0, 0.4, 34],
        'Low risk3': [0, 105, 62, 23, 45, 21.5, 0.3, 29],
    }

    st.write("### Predefined data overview:")
    predefined_df = pd.DataFrame(predefined_data).T
    predefined_df.columns = features
    st.write(predefined_df)

    selected_scenario = st.selectbox("Select a predefined scenario:", list(predefined_data.keys()))

    scenario_values = predefined_data[selected_scenario]
    st.write(f"Selected scenario: {selected_scenario}")
    st.write(f"Values: {scenario_values}")

    if st.button("Submit"):
        input_data = pd.DataFrame({
            'Pregnancies': [scenario_values[0]],
            'PlasmaGlucose': [scenario_values[1]],
            'DiastolicBloodPressure': [scenario_values[2]],
            'TricepsThickness': [scenario_values[3]],
            'SerumInsulin': [scenario_values[4]],
            'BMI': [scenario_values[5]],
            'DiabetesPedigree': [scenario_values[6]],
            'Age': [scenario_values[7]]
        })

        input_data_scaled = scaler.transform(input_data[features])

        y_pred = rf_model.predict(input_data_scaled)
        y_prob = rf_model.predict_proba(input_data_scaled)[:, 1][0]

        st.subheader("Prediction result:")
        st.write(f"🔍 **Diabetes probability: {round(y_prob * 100, 2)}%**")

        if y_prob > 0.8:
            st.error("⚠️ **High Risk**: Consider consulting a doctor.")
        elif 0.5 < y_prob <= 0.8:
            st.warning("🟠 **Moderate Risk**: Medical consultation recommended.")
        else:
            st.success("✅ **Low Risk**: Maintain a healthy lifestyle.")

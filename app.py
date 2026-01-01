import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load the model
try:
    model = joblib.load('model.joblib')
except FileNotFoundError:
    st.error("Model file 'model.joblib' not found. Please make sure the model file is in the same directory.")
    st.stop()


# Load the data
try:
    teleco_base_data = pd.read_csv('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    teleco_data_dummies = pd.read_csv('data/tel_data.csv')
except FileNotFoundError:
    st.error("Data files not found. Please make sure 'data/WA_Fn-UseC_-Telco-Customer-Churn.csv' and 'data/tel_data.csv' are available.")
    st.stop()


st.set_page_config(layout="wide", page_title="Telco Churn Prediction", page_icon="📶")

st.title("📱 Telco Customer Churn Prediction")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Prediction", "Exploratory Data Analysis"])

def get_user_input():
    st.header("Enter Customer Details")
    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox('Gender', ('Male', 'Female'))
        partner = st.selectbox('Partner', ('Yes', 'No'))
        dependents = st.selectbox('Dependents', ('Yes', 'No'))
        phone_service = st.selectbox('Phone Service', ('Yes', 'No'))
        paperless_billing = st.selectbox('Paperless Billing', ('Yes', 'No'))
        senior_citizen = st.checkbox('Senior Citizen')


    with col2:
        tenure = st.slider('Tenure (months)', min_value=0, max_value=72, value=1)
        multiple_lines = st.selectbox('Multiple Lines', ('Yes', 'No', 'No phone service'))
        internet_service = st.selectbox('Internet Service', ('DSL', 'Fiber optic', 'No'))
        online_security = st.selectbox('Online Security', ('Yes', 'No', 'No internet service'))
        online_backup = st.selectbox('Online Backup', ('Yes', 'No', 'No internet service'))
        
    with col3:
        device_protection = st.selectbox('Device Protection', ('Yes', 'No', 'No internet service'))
        tech_support = st.selectbox('Tech Support', ('Yes', 'No', 'No internet service'))
        streaming_tv = st.selectbox('Streaming TV', ('Yes', 'No', 'No internet service'))
        streaming_movies = st.selectbox('Streaming Movies', ('Yes', 'No', 'No internet service'))
        contract = st.selectbox('Contract', ('Month-to-month', 'One year', 'Two year'))
        payment_method = st.selectbox('Payment Method', ('Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'))

    monthly_charges = st.number_input('Monthly Charges', min_value=0.0, value=50.0)
    total_charges = st.number_input('Total Charges', min_value=0.0, value=100.0)
    
    input_data = {
        'gender': gender,
        'SeniorCitizen': 1 if senior_citizen else 0,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges
    }

    return pd.DataFrame([input_data])

def preprocess_input(df):
    # Create tenure_group
    labels = ["{0}-{1}".format(i, i + 11) for i in range(1, 72, 12)]
    df['tenure_group'] = pd.cut(df.tenure, range(1, 80, 12), right=False, labels=labels)
    df.drop(columns=['tenure'], inplace=True)

    # Create dummy variables
    df_dummies = pd.get_dummies(df)

    # Align columns with the training data
    training_cols = teleco_data_dummies.drop(columns=['Churn', 'Unnamed: 0']).columns
    
    for col in training_cols:
        if col not in df_dummies.columns:
            df_dummies[col] = False

    df_dummies = df_dummies[training_cols]
    
    return df_dummies

if page == "Prediction":
    user_input = get_user_input()

    if st.button("Predict Churn"):
        processed_input = preprocess_input(user_input)
        prediction = model.predict(processed_input)
        prediction_proba = model.predict_proba(processed_input)

        st.subheader("Prediction Result")
        if prediction[0] == 1:
            st.error("This customer is likely to churn. 😞")
            st.write(f"Probability of churn: {prediction_proba[0][1]:.2f}")
        else:
            st.success("This customer is not likely to churn. 🎉")
            st.write(f"Probability of not churning: {prediction_proba[0][0]:.2f}")


elif page == "Exploratory Data Analysis":
    st.header("📊 Exploratory Data Analysis")
    st.write("This page shows the exploratory data analysis from the notebook.")

    with st.expander("Churn Count"):
        fig, ax = plt.subplots(figsize=(8, 6))
        teleco_base_data['Churn'].value_counts().plot(kind='barh', ax=ax)
        ax.set_title("Churn Count")
        ax.set_xlabel("Count")
        ax.set_ylabel("Churn")
        st.pyplot(fig)

    with st.expander("Categorical Features vs. Churn"):
        teleco_data = teleco_base_data.copy()
        teleco_data.TotalCharges = pd.to_numeric(teleco_data.TotalCharges, errors='coerce')
        teleco_data.dropna(how='any', inplace=True)
        
        for i, predictor in enumerate(teleco_data.drop(columns=['Churn', 'TotalCharges', 'MonthlyCharges', 'tenure', 'customerID'])):
            fig, ax = plt.subplots()
            sns.countplot(data=teleco_data, x=predictor, hue='Churn', ax=ax)
            ax.set_title(f"{predictor} vs Churn")
            plt.xticks(rotation=45)
            st.pyplot(fig)

    with st.expander("Charges Distribution by Churn"):
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        sns.kdeplot(teleco_data_dummies[teleco_data_dummies['Churn']==0]['MonthlyCharges'], color='Red', fill=True, ax=axes[0])
        sns.kdeplot(teleco_data_dummies[teleco_data_dummies['Churn']==1]['MonthlyCharges'], color="Blue", fill=True, ax=axes[0])
        axes[0].legend(["No Churn","Churn"],loc='upper right')
        axes[0].set_ylabel('Density')
        axes[0].set_xlabel('Monthly Charges')
        axes[0].set_title('Monthly charges by churn')

        sns.kdeplot(teleco_data_dummies[teleco_data_dummies['Churn']==0]['TotalCharges'], color='Red', fill=True, ax=axes[1])
        sns.kdeplot(teleco_data_dummies[teleco_data_dummies['Churn']==1]['TotalCharges'], color="Blue", fill=True, ax=axes[1])
        axes[1].legend(["No Churn","Churn"],loc='upper right')
        axes[1].set_ylabel('Density')
        axes[1].set_xlabel('Total Charges')
        axes[1].set_title('Total charges by churn')
        st.pyplot(fig)

    with st.expander("Correlation Heatmap"):
        fig, ax = plt.subplots(figsize=(20, 20))
        sns.heatmap(teleco_data_dummies.corr(), cmap='Paired', ax=ax, annot=False)
        st.pyplot(fig)

    with st.expander("Correlation with Churn"):
        fig, ax = plt.subplots(figsize=(20, 8))
        teleco_data_dummies.corr()['Churn'].sort_values(ascending=False).plot(kind='bar', ax=ax)
        ax.set_title("Correlation with Churn")
        st.pyplot(fig)

# FUTURE_ML_02
Future_Interns Task 2 - Churn Analysis

# 📉 Customer Churn Prediction System

## 📋 Project Overview
In industries like telecommunications, retaining customers is often more profitable than acquiring new ones. This project is a **Churn Prediction System** designed to identify customers who are likely to stop using a service.

Using historical customer data, I built a machine learning model to predict churn probability and identifying key indicators (churn drivers). The final output is an interactive dashboard that allows business decision-makers to assess risk and take proactive retention measures.

## 🚀 Key Features
* **Data Analysis:** Comprehensive EDA (Exploratory Data Analysis) to understand customer demographics and service usage patterns.
* **Predictive Modeling:** Trained classification models (Logistic Regression, Random Forest, XGBoost) to predict customer churn.
* **Imbalanced Data Handling:** Utilized techniques like SMOTE (Synthetic Minority Over-sampling Technique) to handle class imbalance.
* **Model Evaluation:** Focused on **Recall** and **F1-Score** to minimize false negatives (missing at-risk customers).
* **Interactive Dashboard:** A user-friendly web app built with **Streamlit** to demo the model in real-time.

## 🛠️ Technologies Used
* **Language:** Python
* **Data Manipulation:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn
* **Machine Learning:** Scikit-learn, XGBoost, Imbalanced-learn (SMOTE)
* **Model Deployment:** Streamlit, Pickle

## 📂 Dataset
The project uses the **Telco Customer Churn Dataset** (sourced from Kaggle).
* **Rows:** 7,043 customers
* **Features:** 21 columns including:
    * *Demographics:* Gender, Senior Citizen, Partner, Dependents
    * *Services:* Phone, Internet, Online Security, Tech Support
    * *Account:* Contract, Payment Method, Monthly Charges, Tenure
    * *Target:* Churn (Yes/No)

## 📊 Project Workflow
1.  **Data Preprocessing:** Handling missing values (`TotalCharges`), encoding categorical variables (One-Hot Encoding), and feature scaling.
2.  **Exploratory Data Analysis (EDA):** Visualizing churn rates across different contract types, payment methods, and tenure.
3.  **Model Building:** Training multiple classifiers and tuning hyperparameters.
4.  **Evaluation:** Analyzing the Confusion Matrix to maximize Recall.
5.  **Deployment:** Saving the best model using `pickle` and building the Streamlit interface.

## ⚙️ Installation & Usage

1.  **Clone the repository:**
    ```bash
    git clone <"https://github.com/derrickrajkumar10/FUTURE_ML_02.git">
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Streamlit App:**
    ```bash
    streamlit run app.py
    ```

## 📈 Model Performance
* **Primary Metric:** Recall (Sensitivity)
* *Why Recall?* In churn analysis, a **False Negative** (predicting a customer will stay when they actually leave) is the most costly error. Therefore, the model is optimized to capture as many potential churners as possible.



## 🤝 Contributing
Contributions, issues, and feature requests are welcome!

## 📜 License
This project is licensed under the MIT License.
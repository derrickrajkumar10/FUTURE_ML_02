<div align="center">
  <img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=200&section=header&text=Telecom%20Churn%20Analysis&fontSize=46&fontColor=fff&animation=twinkling&fontAlignY=38&desc=Predicting%20customer%20churn%20before%20it%20happens&descAlignY=62&descSize=18"/>
</div>

<div align="center">

![Python](https://img.shields.io/badge/Python-0D1117?style=for-the-badge&logo=python&logoColor=3776AB)
![scikit-learn](https://img.shields.io/badge/scikit--learn-0D1117?style=for-the-badge&logo=scikit-learn&logoColor=F7931E)
![Streamlit](https://img.shields.io/badge/Streamlit-0D1117?style=for-the-badge&logo=streamlit&logoColor=FF4B4B)
![Jupyter](https://img.shields.io/badge/Jupyter-0D1117?style=for-the-badge&logo=jupyter&logoColor=F37626)
![License](https://img.shields.io/badge/License-MIT-0D1117?style=for-the-badge)

</div>

<img width="100%" src="https://capsule-render.vercel.app/api?type=rect&color=gradient&customColorList=6,11,20&height=2"/>

## 📡 About

A full ML pipeline for predicting customer churn in the telecom sector — from raw data exploration to trained models deployed behind a Streamlit app.

Three production-ready models trained and saved: Decision Tree, Gradient Boosting, and Random Forest. Load any of them and predict churn in real time.

<img width="100%" src="https://capsule-render.vercel.app/api?type=rect&color=gradient&customColorList=6,11,20&height=2"/>

## ✨ Features

- 🔍 **Deep EDA** — exploration of churn drivers, patterns, and correlations
- 🤖 **3 trained models** — Decision Tree, Gradient Boosting, Random Forest (`.pkl` files included)
- 🌐 **Streamlit app** — real-time churn prediction via a clean UI
- 📊 **Feature importance** — understand what actually drives customers to leave
- 🔄 **End-to-end pipeline** — raw CSV → cleaned features → model → live predictions

<img width="100%" src="https://capsule-render.vercel.app/api?type=rect&color=gradient&customColorList=6,11,20&height=2"/>

## 🛠 Tech Stack

| Layer | Technology |
|-------|-----------|
| **Language** | Python 3.x |
| **ML** | scikit-learn (Decision Tree, Gradient Boosting, Random Forest) |
| **App** | Streamlit |
| **Data** | Pandas · NumPy |
| **Visualisation** | Matplotlib · Seaborn |
| **Notebooks** | Jupyter |

<img width="100%" src="https://capsule-render.vercel.app/api?type=rect&color=gradient&customColorList=6,11,20&height=2"/>

## 🚀 Getting Started

```bash
git clone https://github.com/derrickrajkumar10/Churn-Analysis-in-Telecom-Sector.git
cd Churn-Analysis-in-Telecom-Sector

pip install -r requirements.txt

# Launch the Streamlit churn predictor
streamlit run app.py
```

The app loads the saved models automatically — no retraining needed.

<img width="100%" src="https://capsule-render.vercel.app/api?type=rect&color=gradient&customColorList=6,11,20&height=2"/>

## 📁 Project Structure

```
├── Churn_Analysis_EDA.ipynb      # Exploratory Data Analysis
├── Churn_Prediction.ipynb        # Model training and evaluation
├── app.py                        # Streamlit prediction app
├── churn_model_dt.pkl            # Saved Decision Tree model
├── churn_model_gb.pkl            # Saved Gradient Boosting model
├── churn_model_rf.pkl            # Saved Random Forest model
├── model.joblib                  # Primary model (joblib format)
├── data/                         # Raw and processed datasets
└── requirements.txt
```

<div align="center">
  <img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=100&section=footer"/>
</div>

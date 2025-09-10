# 📊 Customer Churn Prediction

## 📌 Overview
This project predicts **customer churn** using an **Artificial Neural Network (ANN)**.  
It processes telecom customer records, performs feature engineering, and provides real-time predictions via a Flask API and Streamlit dashboard.

---

## 🛠️ Tech Stack
- **Languages:** Python  
- **Libraries:** NumPy, Pandas, Scikit-learn, TensorFlow, Keras, Matplotlib, Seaborn  
- **Deployment:** Flask API, Streamlit  

---

## 🚀 Key Features
- Preprocessed **10,000+ customer records** (handling missing values & encoding).  
- Applied **GridSearchCV & Cross-Validation** for hyperparameter tuning.  
- Achieved **92% F1-score**, significantly improving baseline accuracy.  
- Added **dropout & batch normalization** to prevent overfitting.  
- Deployed with **Flask API** and **Streamlit dashboard** for real-time prediction.  

---

## 📈 Results
- **Model:** ANN  
- **Evaluation Metric:** F1-Score = **92%**  
- Outperformed baseline models (Logistic Regression, Random Forest).  

---

## ▶️ How to Run
```bash
git clone https://github.com/SakshamAsopa/churn-prediction.git
cd churn-prediction
pip install -r requirements.txt
python app.py   # Run Flask API
streamlit run dashboard.py

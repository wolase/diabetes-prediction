# diabetes-prediction
Machine learning-based diabetes prediction using with a Streamlit web app for easy data input and analysis.

#  Diabetes prediction using Machine Learning  

## Overview  
This project predicts the likelihood of diabetes onset in women using machine learning techniques. It is based on a dataset collected from the **Taipei Municipal Medical Center (2018–2022)**. The dataset includes key health indicators such as number of pregnancies, plasmaglucose,SerumInsulin,BMI,age
TricepsThickness,blood pressure and DiabetesPedigree.


A **web application** has been developed to allow users to:  
✅ **Upload a CSV file** for batch prediction.  
✅ **Manually input data** to get an instant probability score.  
✅ **Use predefined test values** to explore model performance.  

## 📊 Methodology  
1. **Exploratory Data Analysis (EDA)** – Analyzing feature distributions, handling duplicates, and visualizing correlations.  
2. **Feature Engineering** – Standardizing data with z-score normalization for improved model performance.  
3. **Model Training & Evaluation** – Comparing **Logistic Regression, Decision Tree, and Random Forest** models.  
4. **Best Model Selection** – **Random Forest** was chosen as the final model due to its highest accuracy (**93%**).  
5. **Deployment** – A **Streamlit web app** was built for easy user interaction.  

## 🏆 Results  
The **Random Forest model** achieved:  
- **Precision** – 93%  
- **Recall** – 92%  
- **F1-Score** – 92%  
- **Accuracy** – 93%  

📢 The model performed well on the **Taipei dataset**, but when tested on the **PIMA dataset**, accuracy dropped, highlighting dataset relevance challenges.  

## 🛠 Installation & Usage  

```bash
git clone https://github.com/your-repo-name.git
cd your-repo-name
pip install -r requirements.txt
streamlit run app.py # Make sure to update any necessary paths in app.py before running the app





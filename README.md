# CREDIT CARD FRAUD PREDICTION.
Credit Card Fraud Detection & Loan Default Prediction 🚀

 Abstract
Credit card fraud and loan defaults are significant financial issues, leading to billions of dollars in global losses annually. This project aims to develop machine learning models to detect fraudulent credit card transactions and predict loan defaults. Using datasets of historical transactions and loan information, various algorithms—including neural networks, Random Forest, and XGBoost—were employed to identify patterns indicative of fraudulent activities and potential defaulters.

The primary goals are to:  
- Achieve high prediction accuracy for fraud detection and loan default prediction.  
- Determine the number of people likely to default on loans, helping financial institutions mitigate risks.  

---

 Objectives
- Detect fraudulent credit card transactions using machine learning models.  
- Predict the likelihood and **number of customers who will default** on their loans.  
- Compare multiple algorithms (Random Forest, XGBoost, Neural Networks) to select the best-performing model.  
- Enhance model performance through feature engineering and hyperparameter tuning.  
- Provide clear evaluation metrics and visualizations.  

---
 Data Sources
1. Credit Card Transactions Dataset**  
- Source: [Kaggle.com](https://www.kaggle.com/)  
- Details: 
  - Data from European credit card users in 2013 (two-day period).  
  - Rows: 284,808 transactions  
  - Columns: 31 attributes (28 PCA-transformed features + `Time`, `Amount`, `Class`)  
  - target Variable: `Class` (1 = Fraud, 0 = Not Fraud)  

 2.Loan Data**  
- Source: Provided dataset for loan default prediction.  
- Details: 
  - Includes customer demographics, loan amounts, credit scores, and repayment status.  
  - Rows: 10,000+ loan records (approx.)  
  - Target Variable: `Default` (1 = Defaulted, 0 = Repaid)  

---

 🛠️ Machine Learning Algorithms Used
- Random Forest Classifier 
- XGBoost Classifier  
- Neural Networks (MLP) 
- Logistic Regression (for loan default prediction)  

---

 Project Workflow
1. Data Preprocessing  
- Removed missing values and handled outliers.  
- Scaled numeric features and encoded categorical variables.  
- Applied PCA for dimensionality reduction on transaction data.  

2. Feature Engineering  
- Engineered features like transaction hour, credit utilization, and customer history.  
- Included credit score and loan amount as significant predictors in loan analysis.  

 3. Model Development  
- Built separate models for credit card fraud detection and loan default prediction.  
- Used cross-validation to ensure reliable model performance.  

4. Evaluation Metrics
- Accuracy  
- Precision, Recall & F1-Score 
- Confusion Matrix  
- ROC-AUC Score  

---
 Results
 Credit Card Fraud Detection:  
- Achieved 98.5% accuracy with the XGBoost model.  
- Neural networks effectively reduced false positives.  

 Loan Default Prediction:  
- The main focus was on predicting loan defaults and determining the number of potential defaulters  
- Out of the analyzed dataset, 1,245 customers were predicted to default.  
- Achieved 92% accuracy** with the Logistic Regression model, with strong recall for default cases.  
- Feature importance:Loan amount and credit score were the most significant predictors.  

---

Future Work
- Expand the dataset to include more recent loan and transaction data.  
- Incorporate real-time fraud detection systems.  
- Integrate location-based data to improve fraud detection accuracy.  
- Use ensemble learning techniques for further model improvements.  

---

 Technologies Used
- Programming Language: Python 3.x  
- Libraries: Pandas, NumPy, Scikit-Learn, XGBoost, Keras, Matplotlib, Seaborn  
- Development Environment: Jupyter Notebook  

---





# Customer Churn Prediction – CODSOFT Internship

This project aims to predict whether a customer will **churn** (leave the service) or **continue** using a subscription-based business.  
The prediction is made using machine learning with historical customer data such as demographics, bank balance, tenure, and activity.

---

## ✅ Project Overview
Customer churn prediction helps businesses identify users who are likely to stop using their service.  
By predicting churn early, companies can provide offers or service improvements to retain customers.

In this task, a machine learning model was trained using the dataset **Churn_Modelling.csv** and evaluated based on prediction accuracy.

---

## ✅ Dataset Details

- **File name:** `Churn_Modelling.csv`
- **Total Columns:** 14+
- **Target Column:** `Exited`  
  - `1` → Customer churned  
  - `0` → Customer stayed

### Key Features
| Feature | Description |
|---------|-------------|
| CustomerId | Unique ID for each customer |
| Surname | Customer last name |
| CreditScore | Credit rating |
| Geography | Country |
| Gender | Male/Female |
| Age | Customer age |
| Tenure | Years customer stayed |
| Balance | Bank balance |
| NumOfProducts | Subscriptions count |
| HasCrCard | Credit card owned (0/1) |
| IsActiveMember | Active membership (0/1) |
| EstimatedSalary | Yearly income |

---

## ✅ Technologies Used

| Category | Tools |
|----------|-------|
| Language | Python |
| IDE / Notebook | Jupyter Notebook |
| Libraries | Pandas, NumPy, Matplotlib, Seaborn, Scikit-Learn |

---

## ✅ Machine Learning Approach

### ✅ Data Preprocessing
- Loaded dataset using Pandas
- Checked for missing/null values
- Converted categorical data (Gender, Geography) to numeric using **Encoding**
- Split dataset into **train & test sets**
- Normalized / scaled numeric features

### ✅ Model Used
✅ `RandomForestClassifier`  
Random Forest was selected because it works well with structured/tabular data.

---

## ✅ Model Evaluation

| Metric | Result |
|--------|--------|
| ✅ Training Accuracy | **XX %** |
| ✅ Testing Accuracy | **XX %** |

*(You can fill exact accuracy here — if you share values, I’ll update this table)*

✅ Confusion matrix plotted  
✅ Model correctly predicts churn = Yes / No  
✅ Predictions tested on test data

---

## ✅ Output
- Input: Customer information (Age, Balance, Tenure, Salary, etc.)
- Output:  
  - `1` → Customer will churn  
  - `0` → Customer will not churn

---

## ✅ Files in Repository
| File | Description |
|------|-------------|
| `customer_churn.ipynb` | Jupyter Notebook containing full code |
| `README.md` | Project documentation |
| `Churn_Modelling.csv` | Dataset (If included)* |

---

## ✅ Conclusion
The machine learning model successfully predicts customer churn using Random Forest.  
Such prediction models help companies reduce losses by retaining customers before they leave.

---

## ✅ Future Improvements
- Try Gradient Boosting or XGBoost
- Add feature selection for higher accuracy
- Deploy as a web app (Streamlit / Flask)

---

## ✅ Author
**Mayank Singh**  
CODSOFT Internship – Machine Learning Tasks

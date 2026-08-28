# AI-Powered Student Expense Analyzer

## AI Fundamentals Project

This project is developed as part of the AI Fundamentals course by Edunet Foundation in collaboration with AICTE.

The AI-Powered Student Expense Analyzer uses Artificial Intelligence and Machine Learning techniques to analyze student spending patterns, categorize expenses, identify unusual spending, and provide simple budgeting insights.

## Features

- Automatic expense categorization
- Spending analysis
- Category-wise spending visualization
- Unusual spending detection
- AI-based spending insights
- Interactive Streamlit dashboard
- Downloadable analyzed expense data

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Streamlit

## Machine Learning Techniques

### 1. TF-IDF

TF-IDF is used to convert transaction descriptions into numerical features that can be processed by the machine-learning model.

### 2. Logistic Regression

Logistic Regression is used to classify transactions into categories such as Food, Travel, Education, Shopping, Entertainment, and Utilities.

### 3. Isolation Forest

Isolation Forest is used to identify transactions that are noticeably different from the normal spending pattern.

An unusual spending alert does not mean that a transaction is fraudulent. It only indicates that the transaction should be checked by the user.

## Project Structure

```text
AI-Student-Expense-Analyzer/
│
├── app.py
├── requirements.txt
├── README.md
│
├── data/
│   └── expenses.csv
│
└── src/
    ├── _init_.py
    ├── generate_data.py
    ├── analysis.py
    ├── train_classifier.py
    ├── anomaly_detection.py
    ├── insights.py
    └── main.py

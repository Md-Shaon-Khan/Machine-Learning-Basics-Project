# Machine Learning Basics Project

This repository is a collection of beginner-level Machine Learning projects built using Python and Jupyter Notebooks. Each project focuses on a real-world problem and walks through the full ML pipeline — from data loading and preprocessing through model training, evaluation, and interpretation. The goal is to build a solid practical foundation in core ML concepts by working on datasets that reflect actual use cases.

All projects are self-contained within their own folders and can be run independently.

---

## Projects

### Bank Loan Prediction
Predicts whether a loan application will be approved or rejected based on applicant details such as income, credit history, and loan amount. This is a binary classification problem and explores algorithms commonly used in financial risk assessment.

### Boston House Price Prediction
Estimates the median value of homes in the Boston area using features such as crime rate, number of rooms, and proximity to employment centers. A classic regression problem used to demonstrate linear and polynomial regression techniques.

### Breast Cancer Detection
Classifies tumors as malignant or benign using features extracted from digitized images of fine needle aspirate (FNA) of breast mass. This project covers binary classification with an emphasis on model accuracy and sensitivity given the medical context.

### Car Price Prediction
Predicts the selling price of used cars based on attributes such as brand, year of manufacture, fuel type, transmission, and mileage. Covers regression techniques with categorical feature encoding and feature importance analysis.

### Customer Churn Prediction
Determines which customers are likely to leave a service or subscription. This is a classification problem that involves handling imbalanced datasets and using metrics beyond plain accuracy to evaluate model performance.

### Diabetes Prediction
Predicts the likelihood of a patient having diabetes based on diagnostic measurements such as glucose levels, BMI, age, and blood pressure. Covers binary classification and demonstrates the importance of data preprocessing and feature scaling in medical datasets.

### Email Spam Detection
Classifies emails as spam or not spam using natural language processing techniques and text feature extraction. Introduces concepts like TF-IDF vectorization and Naive Bayes classification for text data.

### House Price Prediction
Predicts residential property prices based on structural and neighborhood features. Similar in scope to the Boston dataset but uses a different dataset and explores additional feature engineering and regression methods.

### Movies Recommendation
Builds a basic recommendation system that suggests movies to users based on similarity. Covers content-based filtering and introduces how similarity metrics are used to compare items.

### Profile Prediction
Predicts a user or student profile category based on input features. Focuses on classification and demonstrates how ML models can be applied to profiling and segmentation problems.

### Project Cardio Train
Uses cardiovascular health data to predict the presence or absence of heart disease. Covers data cleaning, feature selection, and classification model comparison on health-related datasets.

### Rock vs Mine Classification
Classifies sonar signals as either rocks or mines using a dataset of sonar readings. A binary classification problem that demonstrates how ML can be applied to signal processing data.

---

## Topics Covered

The projects in this repository collectively cover the following core Machine Learning concepts:

- Data loading and exploratory data analysis (EDA)
- Data cleaning and handling missing values
- Feature engineering and encoding categorical variables
- Feature scaling and normalization
- Supervised learning: regression and classification
- Model training and hyperparameter selection
- Model evaluation using accuracy, precision, recall, F1-score, and confusion matrix
- Text feature extraction (TF-IDF, bag of words)
- Content-based recommendation systems
- Imbalanced dataset handling

---

## Technologies Used

- **Python** — primary programming language
- **Jupyter Notebook** — all projects are written as interactive notebooks
- **NumPy** — numerical computation
- **Pandas** — data manipulation and analysis
- **Matplotlib / Seaborn** — data visualization
- **Scikit-learn** — machine learning models and evaluation utilities

---

## Getting Started

Clone the repository:

```bash
git clone https://github.com/Md-Shaon-Khan/Machine-Learning-Basics-Project.git
cd Machine-Learning-Basics-Project
```

Install the required dependencies:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Navigate into any project folder and open the `.ipynb` file to run it. Each notebook is self-contained and includes its own dataset or instructions on where to obtain the data.

---

## Repository Structure

```
Machine-Learning-Basics-Project/
│
├── Bank Loan/
├── Boston House price/
├── Breast Cancer Detection/
├── Car Price Prediction/
├── Customer Churn Prediction/
├── Diabetes Prediction/
├── Email Spam/
├── House Price/
├── Movies Recomendation/
├── Profile Prediction/
├── Project Cardio Train/
├── Rock_VS_Mine/
├── test.ipynb
└── test.py
```

---

## Notes

This repository is intended as a learning resource. The projects here are not production systems but rather structured exercises designed to reinforce understanding of standard ML workflows. The datasets used are either publicly available or derived from well-known sources such as Kaggle and the UCI Machine Learning Repository.

The `test.ipynb` and `test.py` files at the root level are scratch files used during development and are not part of any specific project.

---

## Author

**Md Shaon Khan**  
GitHub: [Md-Shaon-Khan](https://github.com/Md-Shaon-Khan)

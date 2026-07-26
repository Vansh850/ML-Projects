# 🏥 Medical Insurance Cost Predictor

A Machine Learning project that predicts an individual's medical insurance charges based on demographic and health-related features such as age, BMI, smoking status, number of children, and region. This project demonstrates the complete machine learning workflow, from exploratory data analysis (EDA) and data preprocessing to model training, evaluation, and prediction using Linear Regression.

---

## 📌 Project Overview

Healthcare costs vary significantly depending on various personal and lifestyle factors. The objective of this project is to build a regression model capable of estimating medical insurance charges based on historical data.

The project covers:

- Data loading and cleaning
- Exploratory Data Analysis (EDA)
- Data visualization
- Feature encoding
- Model training
- Model evaluation
- Insurance cost prediction for new individuals

---

## 📂 Dataset

The project uses the **Medical Insurance Dataset**, which contains the following features:

| Feature | Description |
|----------|-------------|
| age | Age of the insured individual |
| sex | Gender |
| bmi | Body Mass Index |
| children | Number of dependent children |
| smoker | Smoking status |
| region | Residential region |
| charges | Medical insurance cost (Target Variable) |

---

## 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

---

## 📊 Exploratory Data Analysis

The notebook includes several visualizations to understand the dataset:

- Distribution of insurance charges
- Gender distribution
- Smoker vs Non-Smoker analysis
- Region-wise analysis
- Boxplots
- Histograms
- Correlation Heatmap
- Scatter plots

These visualizations help identify relationships between variables before model training.

---

## ⚙ Data Preprocessing

The following preprocessing steps were performed:

- Checked for missing values
- Statistical summary of the dataset
- Converted categorical variables into numerical values
- Selected relevant features
- Train-Test Split

---

## 🤖 Machine Learning Model

### Algorithm Used

- Linear Regression

The dataset was split into training and testing sets using `train_test_split()` from Scikit-learn.

The model was trained to predict the insurance charges.

---

## 📈 Model Evaluation

The model performance was evaluated using:

- R² Score

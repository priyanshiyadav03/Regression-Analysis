# Medical Insurance Expense Prediction

This project predicts **medical insurance expenses** based on features like **age, BMI, and number of children**.

We explore the dataset, understand feature relationships, detect outliers, and build a **Linear Regression model** to predict expenses.

---

##  Dataset

**File:** `insurance.csv`

### Features

- **age** → Age of the individual  
- **bmi** → Body Mass Index  
- **children** → Number of children  

###  Target

- **expenses** → Medical insurance cost

---

##  Part A: Exploratory Data Analysis (EDA)

###  Objective
Understand the data and feature relationships before modeling.

### ⚙️ Implementation

- Loaded dataset and checked:
  - First 5 rows
  - Columns
  - Data types

- Calculated summary statistics:
  - Mean
  - Standard deviation
  - Min / Max
  - Quartiles

- Checked for missing values → **None found**

- Detected outliers using **IQR method**

- Calculated **Pearson correlation**

###  Results

- Most people are aged **30–40**
- BMI mostly between **20–30**
- Most people have **0–2 children**
- Expenses are **right-skewed**
- **Age & BMI** show positive correlation with expenses

###  Visualizations

- **Histograms & KDE plots** → Feature distributions  
- **Boxplots** → Median, IQR, outliers  
- **Correlation heatmap** → Feature relationships  

---

##  Part B: Linear Regression

###  Objective
Predict insurance expenses using dataset features.

###  Implementation

- **Features selected:** `bmi` (Simple Linear Regression)
- **Model:** Linear Regression (scikit-learn)
- **Notebook:** `part_b_linear_regression.ipynb`

###  Results

Regression Equation:

```text
expenses = (slope × bmi) + intercept

This equation can be used to predict expenses for any BMI value.

## Visualization

- Scatter plot showing relationship between BMI and expenses
- Regression line fitted to the data points

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

## Key Learnings

- Explored data distribution, skewness, and density using histograms and KDE plots
- Detected outliers using IQR method
- Measured feature relationships with Pearson correlation
- Built and interpreted a linear regression model
- Learned how to predict expenses using regression equation



This project predicts medical insurance expenses based on features like age, BMI, and number of children.
We explore the dataset, understand feature relationships, detect outliers, and build a linear regression model to predict expenses.

Dataset

  File: insurance.csv

Features:

  age → Age of the individual

  bmi → Body Mass Index

  children → Number of children

Target: expenses → Medical insurance cost

Part A: Exploratory Data Analysis (EDA)

Objective: Understand the data and feature relationships before modeling.

Implementation:

  Loaded dataset and checked first 5 rows, columns, and data types.

  Calculated summary statistics (mean, std, min, max, quartiles).

  Checked for missing values → none found.

  Detected outliers using IQR method.

  Calculated Pearson correlation to see which features affect expenses.

Results:

  Most people are aged 30–40.

  BMI mostly between 20–30.

  Most people have 0–2 children.

  Expenses are right-skewed (few people have very high costs).

  Age and BMI show a positive correlation with expenses.

Visualization:

  Histograms & KDE plots → show feature distributions and density peaks.

  Boxplots → show median, spread (IQR), and outliers.

  Correlation heatmap → shows strength of feature relationships.

Part B: Linear Regression

  Objective: Predict insurance expenses using features from the dataset.

Implementation:

  Features selected: bmi (for simple linear regression)

  Model: Linear Regression using scikit-learn

  Notebook: part_b_linear_regression.ipynb

Results:

  Fitted the regression model to calculate slope and intercept from your dataset.

Regression Equation:

  expenses = (slope × bmi) + intercept

  This equation can be used to predict expenses for any BMI value.

Visualization:

  Scatter plot showing relationship between BMI and expenses.

  Regression line fitted to the data points.

🛠 Technologies Used

  Python

  Pandas

  NumPy

  Matplotlib

  Seaborn

  Scikit-learn

  Jupyter Notebook

Key Learnings:

  Explored data distribution, skewness, and density using histograms and KDE plots.

  Detected outliers using IQR method.

  Measured feature relationships with Pearson correlation.

  Built and interpreted a linear regression model.

  Learned how to predict expenses using regression equation.

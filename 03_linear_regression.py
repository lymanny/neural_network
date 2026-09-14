# -----------------------------------
# SIMPLE LINEAR REGRESSION
# Predict Salary from Years of Experience
# -----------------------------------

# Data source:
# Kaggle - Salary Dataset - Simple Linear Regression
# https://www.kaggle.com/datasets/abhishek14398/salary-dataset-simple-linear-regression


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


# -----------------------------------
# 1. LOAD DATA
# -----------------------------------

# Read CSV file
data = pd.read_csv("Salary_dataset.csv")

# Show first 5 rows by default
print(data.head())


# -----------------------------------
# 2. INPUT AND OUTPUT
# -----------------------------------

# X = training input / independent variable
X = data[["YearsExperience"]]

# y = target / dependent variable
y = data["Salary"]


# -----------------------------------
# 3. CREATE MODEL
# -----------------------------------

# Create Linear Regression model
model = LinearRegression()


# -----------------------------------
# 4. TRAIN MODEL
# -----------------------------------

# Train the model to learn the best Weight (slope) and Bias (intercept) from data
model.fit(X, y)


# -----------------------------------
# 5. WEIGHT AND BIAS
# -----------------------------------

# Get the learned Weight (slope)
# model.coef_ stores the learned weight(s)
# [0] gets the first weight because we have only one input: YearsExperience
weight = model.coef_[0]

# Get the learned Bias (intercept)
# Bias is the predicted value when YearsExperience = 0
bias = model.intercept_

print("\nWeight =", weight)
print("Bias =", bias)


# -----------------------------------
# FORMULA
# -----------------------------------

# Linear Regression formula:
# Prediction = (Input * Weight) + Bias -> y = (x * w) + b
# x = Years of Experience
# w = Weight / slope
# b = Bias / intercept
# y = Predicted Salary


# -----------------------------------
# 6. MAKE A PREDICTION
# -----------------------------------

# Ask user to enter years of experience
experience = float(
    input("\nEnter years of experience: ")
)


# Put the new input into the same format as X
new_data = pd.DataFrame(
    {"YearsExperience": [experience]}
)


# Prediction formula: Salary = (Experience * Weight) + Bias -> y = (x * w) + b


# Use the trained model to predict salary
prediction = model.predict(new_data)


print("\nYears of Experience =", experience)
print("Predicted Salary =", prediction[0]) # [0] gets the first predicted value


# -----------------------------------
# 7. SHOW GRAPH
# -----------------------------------

# Show actual data points
plt.scatter(
    X,
    y,
    label="Actual Data"
)


# Show regression line
# X["YearsExperience"] = x-axis values
# model.predict(X) = predicted Salary values (predicted y)
plt.plot(
    X["YearsExperience"],
    model.predict(X),
    label="Regression Line"
)


# Show user's input + prediction
# as one extra point on the graph
plt.scatter(
    experience,
    prediction[0],
    label="Your Prediction"
)


# Graph labels
plt.xlabel("Years of Experience")
plt.ylabel("Salary")

plt.title(
    "Linear Regression: Experience vs Salary"
)

plt.legend()

plt.show()


# -----------------------------------
# SUMMARY
# -----------------------------------

# Linear Regression:
#
# CSV Data
#    ↓
# X = YearsExperience
# y = Salary
#    ↓
# model.fit(X, y)
#    ↓
# Learn Weight + Bias
#    ↓
#
# User enters experience
#    ↓
# new_data
#    ↓
# model.predict(new_data)
#    ↓
# Predicted Salary
#    ↓
# Show prediction as one point on graph
#
#
# Formula:
#
# y = (x * w) + b
#
# x = input
# w = Weight / slope
# b = Bias / intercept
# y = predicted value
#
#
# Easy summary:
#
# Linear Regression =
# Input -> Learn Weight + Bias -> Predict a number
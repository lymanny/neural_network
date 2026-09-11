import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


# -----------------------------------
# REFERENCES
# -----------------------------------

# Dataset:
# Salary Dataset - Simple Linear Regression
# Source: Kaggle
# https://www.kaggle.com/datasets/abhishek14398/salary-dataset-simple-linear-regression

# Linear Regression:
# scikit-learn LinearRegression documentation


# -----------------------------------
# 1. LOAD DATA
# -----------------------------------

data = pd.read_csv("Salary_dataset.csv")

# Show first 5 rows
print(data.head())


# -----------------------------------
# 2. INPUT AND TARGET
# -----------------------------------

# X = input
X = data[["YearsExperience"]]

# y = correct answer / real salary
y = data["Salary"]


# -----------------------------------
# 3. CREATE MODEL
# -----------------------------------

model = LinearRegression()


# -----------------------------------
# 4. TRAIN MODEL
# -----------------------------------

# Learn Weight and Bias from the data
model.fit(X, y)


# -----------------------------------
# 5. SHOW WEIGHT AND BIAS
# -----------------------------------

print("\nWeight:", model.coef_[0])
print("Bias:", model.intercept_)

# Formula:
# Prediction = (Input × Weight) + Bias


# -----------------------------------
# 6. USER INPUT
# -----------------------------------

experience = float(
    input("\nEnter years of experience: ")
)


# -----------------------------------
# 7. PREDICT SALARY
# -----------------------------------

new_data = pd.DataFrame({
    "YearsExperience": [experience]
})

prediction = model.predict(new_data)[0]

print(
    "Predicted Salary:",
    round(prediction, 2)
)


# -----------------------------------
# 8. DRAW GRAPH
# -----------------------------------

# Real salary data = dots
plt.scatter(
    X["YearsExperience"],
    y,
    label="Real Salary"
)

# Regression line = model prediction
plt.plot(
    X["YearsExperience"],
    model.predict(X),
    label="Regression Line"
)

# Your new prediction = one dot
plt.scatter(
    experience,
    prediction,
    label="Your Prediction"
)

plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Linear Regression - Experience vs Salary")

plt.legend()

# Show graph
plt.show()


# -----------------------------------
# SUMMARY
# -----------------------------------

# X = Years of Experience
# y = Real Salary

# LinearRegression()
# → creates the model

# model.fit(X, y)
# → learns Weight + Bias

# model.predict()
# → predicts Salary

# plt.scatter()
# → shows real data as dots

# plt.plot()
# → shows the regression line

# plt.show()
# → opens the graph
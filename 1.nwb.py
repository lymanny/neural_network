# -----------------------------------
# SIMPLE NEURON EXAMPLE
# -----------------------------------

# Data: [study_hours, sleep_hours, real_exam_score]
data = [
    [2, 6, 30],
    [4, 7, 50],
    [5, 7, 60]
]


# -----------------------------------
# INPUTS
# -----------------------------------

# Inputs = data given to the neuron
x1 = float(input("Enter study hours: "))
x2 = float(input("Enter sleep hours: "))


# -----------------------------------
# WEIGHTS
# -----------------------------------

# Weights = how strongly each input affects the prediction
w1 = 10   # Study has a strong effect
w2 = 0    # Sleep has no effect in this simple example


# -----------------------------------
# BIAS
# -----------------------------------

# Bias = extra adjustment
bias = 10


# -----------------------------------
# NEURON CALCULATION
# -----------------------------------

# Formula:
# z = (x1 × w1) + (x2 × w2) + bias

z = (x1 * w1) + (x2 * w2) + bias


# z is the neuron output / prediction
print("Prediction:", z)


# -----------------------------------
# CHECK CORRECT ANSWER
# -----------------------------------

found = False

for row in data:

    # Check if the user's inputs exist in the dataset
    if x1 == row[0] and x2 == row[1]:

        found = True

        # Real correct answer from the dataset
        correct = row[2]

        # Error = difference between prediction and correct answer
        error = abs(correct - z)

        print("Correct answer:", correct)
        print("Error:", error)

        # Check prediction
        if error == 0:
            print("Prediction is correct ✅")
        else:
            print("Prediction is wrong ❌")


# If input is not in the dataset,
# there is no correct answer available to compare
if not found:
    print("No matching data found.")
    print("Cannot check if the prediction is correct.")


# -----------------------------------
# SUMMARY
# -----------------------------------

# Neuron = a small calculation unit that takes inputs
#          and produces an output

# x1, x2 = Inputs
# w1, w2 = Weights
# bias   = Extra adjustment
# z      = Neuron output / prediction

# Weight:
# Controls how strongly each input affects the result

# Bias:
# Adds an extra adjustment to the result

# Good weights + bias → better prediction
# Bad weights + bias  → bigger error

# In this example:
# We manually choose w1, w2, and bias

# In real Deep Learning:
# The neural network learns the best weights
# and bias automatically during training

# Learning process:
#
# Input
#   ↓
# Weights + Bias
#   ↓
# Prediction
#   ↓
# Compare with correct answer
#   ↓
# Error / Loss
#   ↓
# Training adjusts Weights + Bias
#   ↓
# Better Prediction
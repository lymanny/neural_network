# Gradient Descent and Learning Rate
# Example: Study Hours -> Exam Score


# X = input data
# X means Study Hours
X = [1, 2, 3, 4, 5]

# y = output data / actual answers
# y means Exam Scores
y = [50, 60, 70, 80, 90]


# Start with Weight = 0
# The model does not know the correct Weight yet
weight = 0

# Start with Bias = 0
# The model does not know the correct Bias yet
bias = 0


# Learning Rate
# Controls how big each update step is
learning_rate = 0.01


# The model will practice with the same training data 2000 times.
epochs = 2000


# Count how many input data values are inside X
n = len(X)


# Train the model
# Repeat the training process many times (epochs)
for epoch in range(epochs):

    # Start gradients at 0 for each training round
    weight_gradient = 0
    bias_gradient = 0


    # zip(X, y) matches each Study Hour with its Exam Score
    #
    # X = [1, 2, 3, 4, 5]
    # y = [50, 60, 70, 80, 90]
    #
    # So Python reads:
    # 1 -> 50
    # 2 -> 60
    # 3 -> 70
    # 4 -> 80
    # 5 -> 90

    for study_hours, actual_score in zip(X, y):

        # Make a prediction
        #
        # Linear Regression formula:
        # Prediction = (Input * Weight) + Bias
        prediction = (study_hours * weight) + bias


        # Calculate the error
        #
        # Example:
        # Prediction = 30
        # Actual Score = 60
        #
        # Error = 30 - 60
        # Error = -30
        error = prediction - actual_score


        # Calculate how Weight should change
        weight_gradient += (2 / n) * error * study_hours


        # Calculate how Bias should change
        bias_gradient += (2 / n) * error


    # Gradient Descent
    #
    # Update Weight
    weight = weight - (learning_rate * weight_gradient)

    # Update Bias
    bias = bias - (learning_rate * bias_gradient)


# Show what the model learned
print("Learned Weight:", weight)
print("Learned Bias:", bias)


# Test the trained model with new input
study_hours = 5


# Predict the Exam Score for 5 Study Hours
predicted_score = (study_hours * weight) + bias


# Show result
print("Study Hours:", study_hours)
print("Predicted Exam Score:", predicted_score)
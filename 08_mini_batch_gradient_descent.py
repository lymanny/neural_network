# Mini-Batch Gradient Descent
# Example: Study Hours -> Exam Score

X = [1, 2, 3, 4, 5, 6]
y = [50, 60, 70, 80, 90, 100]

weight = 0
bias = 0

learning_rate = 0.01
epochs = 100

# Use 2 data points before each update
batch_size = 2

n = len(X)

for epoch in range(epochs):

    # Move through data in groups of 2
    for start in range(0, n, batch_size):

        end = start + batch_size

        batch_X = X[start:end]
        batch_y = y[start:end]

        weight_gradient = 0
        bias_gradient = 0

        # Calculate gradient using this mini-batch
        for study_hours, actual_score in zip(batch_X, batch_y):

            prediction = (study_hours * weight) + bias

            error = prediction - actual_score

            weight_gradient += error * study_hours
            bias_gradient += error

        # Number of data points in this batch
        current_batch_size = len(batch_X)

        # Average the gradients
        weight_gradient = (2 / current_batch_size) * weight_gradient
        bias_gradient = (2 / current_batch_size) * bias_gradient

        # Update Weight and Bias
        weight -= learning_rate * weight_gradient
        bias -= learning_rate * bias_gradient


print("Weight:", weight)
print("Bias:", bias)

# Test with new input
study_hours = 7

predicted_score = (study_hours * weight) + bias

print("Study Hours:", study_hours)
print("Predicted Exam Score:", predicted_score)
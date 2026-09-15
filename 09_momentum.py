# Momentum
# Example: Study Hours -> Exam Score

X = [1, 2, 3, 4, 5]
y = [50, 60, 70, 80, 90]

weight = 0
bias = 0

learning_rate = 0.01
momentum = 0.9
epochs = 100

# Remember previous movement
weight_velocity = 0
bias_velocity = 0

for epoch in range(epochs):

    for study_hours, actual_score in zip(X, y):

        # Prediction
        prediction = (study_hours * weight) + bias

        # Error
        error = prediction - actual_score

        # Gradients
        weight_gradient = 2 * error * study_hours
        bias_gradient = 2 * error

        # Momentum
        weight_velocity = (
            momentum * weight_velocity
            - learning_rate * weight_gradient
        )

        bias_velocity = (
            momentum * bias_velocity
            - learning_rate * bias_gradient
        )

        # Update Weight and Bias
        weight = weight + weight_velocity
        bias = bias + bias_velocity


print("Weight:", weight)
print("Bias:", bias)
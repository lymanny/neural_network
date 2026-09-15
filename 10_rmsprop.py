# RMSProp
# Example: Study Hours -> Exam Score

X = [1, 2, 3, 4, 5]
y = [50, 60, 70, 80, 90]

weight = 0
bias = 0

learning_rate = 0.01
decay = 0.9
epsilon = 0.00000001
epochs = 100

# Store recent squared gradients
weight_squared_gradient = 0
bias_squared_gradient = 0


for epoch in range(epochs):

    for study_hours, actual_score in zip(X, y):

        # Prediction
        prediction = (study_hours * weight) + bias

        # Error
        error = prediction - actual_score

        # Gradients
        weight_gradient = 2 * error * study_hours
        bias_gradient = 2 * error

        # RMSProp remembers recent squared gradients
        weight_squared_gradient = (
            decay * weight_squared_gradient
            + (1 - decay) * weight_gradient ** 2
        )

        bias_squared_gradient = (
            decay * bias_squared_gradient
            + (1 - decay) * bias_gradient ** 2
        )

        # RMSProp updates Weight and Bias
        weight = weight - (
            learning_rate * weight_gradient
            / ((weight_squared_gradient ** 0.5) + epsilon)
        )

        bias = bias - (
            learning_rate * bias_gradient
            / ((bias_squared_gradient ** 0.5) + epsilon)
        )


print("Weight:", weight)
print("Bias:", bias)
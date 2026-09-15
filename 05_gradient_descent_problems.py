# Problems with Gradient Descent
# Example: Study Hours -> Exam Score

X = [1, 2, 3, 4, 5]
y = [50, 60, 70, 80, 90]

n = len(X)


def train_model(learning_rate, epochs):

    weight = 0
    bias = 0

    for epoch in range(epochs):

        weight_gradient = 0
        bias_gradient = 0

        for study_hours, actual_score in zip(X, y):

            # Make prediction
            prediction = (study_hours * weight) + bias

            # Calculate error
            error = prediction - actual_score

            # Calculate gradients
            weight_gradient += (2 / n) * error * study_hours
            bias_gradient += (2 / n) * error

        # Gradient Descent
        weight = weight - (learning_rate * weight_gradient)
        bias = bias - (learning_rate * bias_gradient)

    return weight, bias


# 1. Good Learning Rate
weight, bias = train_model(0.01, 2000)

print("Good Learning Rate")
print("Weight:", weight)
print("Bias:", bias)


# 2. Learning Rate Too Small
weight, bias = train_model(0.000001, 2000)

print("\nLearning Rate Too Small")
print("Weight:", weight)
print("Bias:", bias)


# 3. Learning Rate Too Large
weight, bias = train_model(0.1, 2000)

print("\nLearning Rate Too Large")
print("Weight:", weight)
print("Bias:", bias)
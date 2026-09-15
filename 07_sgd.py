# SGD: update Weight and Bias after each data point

X = [1, 2, 3, 4, 5]
y = [50, 60, 70, 80, 90]

weight = 0
bias = 0
learning_rate = 0.01
epochs = 100

for epoch in range(epochs):
    for study_hours, actual_score in zip(X, y):

        prediction = (study_hours * weight) + bias
        error = prediction - actual_score

        weight_gradient = 2 * error * study_hours
        bias_gradient = 2 * error

        weight -= learning_rate * weight_gradient
        bias -= learning_rate * bias_gradient

print("Weight:", weight)
print("Bias:", bias)
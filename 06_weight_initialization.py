import random

# 1. Weight Initialization
weight = random.uniform(-0.1, 0.1)

learning_rate = 0.01

print("Starting Weight:", weight)

# Example gradient
weight_gradient = -5

# 2. Gradient Descent improves the Weight
weight = weight - (learning_rate * weight_gradient)

print("Updated Weight:", weight)
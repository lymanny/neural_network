# -----------------------------------
# SIMPLE MLP
# Structure: 3 Inputs -> 4 Hidden -> 1 Output
# -----------------------------------


# -----------------------------------
# INPUT LAYER
# -----------------------------------

x1 = 4      # Study hours
x2 = 7      # Sleep hours
x3 = 90     # Attendance


# ===================================
# HIDDEN LAYER
# 4 neurons
# ===================================


# Hidden Neuron 1

w1 = 1
w2 = 1
w3 = 0.1

b1 = 1

# Formula:
# z1 = (x1 * w1) + (x2 * w2) + (x3 * w3) + b1

z1 = (x1 * w1) + (x2 * w2) + (x3 * w3) + b1

# ReLU
h1 = max(0, z1)


# Hidden Neuron 2

w4 = 2
w5 = 1
w6 = 0.1

b2 = 1

z2 = (x1 * w4) + (x2 * w5) + (x3 * w6) + b2

h2 = max(0, z2)


# Hidden Neuron 3

w7 = 1
w8 = 2
w9 = 0.1

b3 = 1

z3 = (x1 * w7) + (x2 * w8) + (x3 * w9) + b3

h3 = max(0, z3)


# Hidden Neuron 4

w10 = 1
w11 = 1
w12 = 0.2

b4 = 1

z4 = (x1 * w10) + (x2 * w11) + (x3 * w12) + b4

h4 = max(0, z4)


# ===================================
# OUTPUT LAYER
# 1 neuron
# ===================================

w13 = 0.5
w14 = 0.5
w15 = 0.5
w16 = 0.5

b5 = 1


# Formula:
# output = (h1 * w13)
#        + (h2 * w14)
#        + (h3 * w15)
#        + (h4 * w16)
#        + b5

output = (
    (h1 * w13)
    + (h2 * w14)
    + (h3 * w15)
    + (h4 * w16)
    + b5
)


# -----------------------------------
# RESULT
# -----------------------------------

print("h1 =", h1)
print("h2 =", h2)
print("h3 =", h3)
print("h4 =", h4)

print("Prediction =", output)


# -----------------------------------
# SUMMARY
# -----------------------------------

# MLP = Multi-Layer Perceptron
#
# Structure:
#
# 3 Inputs -> 1 Hidden Layer -> 1 Output
#
# x1 = Study hours
# x2 = Sleep hours
# x3 = Attendance
#
# Hidden Layer:
# h1, h2, h3, h4
#
# Hidden neuron formula:
#
# z = (inputs * weights) + bias
#
# h = ReLU(z)
#
# ReLU(z) = max(0, z)
#
# Output:
#
# Prediction = (hidden outputs * weights) + bias
#
# In real deep learning,
# training learns the weights and biases automatically.
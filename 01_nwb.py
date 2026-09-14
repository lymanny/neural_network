# -----------------------------------
# SIMPLE NEURAL NETWORK
# 3 Inputs -> 4 Hidden -> 4 Hidden -> 1 Output
# -----------------------------------

# GENERAL NEURON FORMULA:
#
# z = (input1 * weight1)
#   + (input2 * weight2)
#   + ...
#   + bias
#
# Hidden neuron output:
# h = ReLU(z)
#
# ReLU formula:
# ReLU(z) = max(0, z)
#
# Positive value -> keep it
# Negative value -> change to 0


# ===================================
# 1. INPUT LAYER
# 3 INPUTS
# ===================================

x1 = 4      # Study hours
x2 = 7      # Sleep hours
x3 = 90     # Attendance


# ===================================
# 2. HIDDEN LAYER 1
# 4 NEURONS
# ===================================

# Formula for each neuron:
#
# z = (x1 * weight)
#   + (x2 * weight)
#   + (x3 * weight)
#   + bias
#
# h = ReLU(z)


# -----------------------------------
# Neuron 1
# -----------------------------------

# Weights
w1 = 1
w2 = 1
w3 = 0.1

# Bias
b1 = 1

# Formula:
# z1 = (x1 * w1) + (x2 * w2) + (x3 * w3) + b1

z1 = (x1 * w1) + (x2 * w2) + (x3 * w3) + b1

# ReLU formula:
# h1 = max(0, z1)

h1 = max(0, z1)


# -----------------------------------
# Neuron 2
# -----------------------------------

# Weights
w4 = 2
w5 = 1
w6 = 0.1

# Bias
b2 = 1

# Formula:
# z2 = (x1 * w4) + (x2 * w5) + (x3 * w6) + b2

z2 = (x1 * w4) + (x2 * w5) + (x3 * w6) + b2

# ReLU
h2 = max(0, z2)


# -----------------------------------
# Neuron 3
# -----------------------------------

# Weights
w7 = 1
w8 = 2
w9 = 0.1

# Bias
b3 = 1

# Formula:
# z3 = (x1 * w7) + (x2 * w8) + (x3 * w9) + b3

z3 = (x1 * w7) + (x2 * w8) + (x3 * w9) + b3

# ReLU
h3 = max(0, z3)


# -----------------------------------
# Neuron 4
# -----------------------------------

# Weights
w10 = 1
w11 = 1
w12 = 0.2

# Bias
b4 = 1

# Formula:
# z4 = (x1 * w10) + (x2 * w11) + (x3 * w12) + b4

z4 = (x1 * w10) + (x2 * w11) + (x3 * w12) + b4

# ReLU
h4 = max(0, z4)


print("Hidden Layer 1:")
print("h1 =", h1)
print("h2 =", h2)
print("h3 =", h3)
print("h4 =", h4)


# ===================================
# 3. HIDDEN LAYER 2
# 4 NEURONS
# ===================================

# Hidden Layer 2 uses the outputs:
#
# h1, h2, h3, h4
#
# Formula:
#
# z = (h1 * weight)
#   + (h2 * weight)
#   + (h3 * weight)
#   + (h4 * weight)
#   + bias
#
# h = ReLU(z)


# -----------------------------------
# Neuron 1
# -----------------------------------

# Weights
w13 = 0.5
w14 = 0.5
w15 = 0.5
w16 = 0.5

# Bias
b5 = 1

# Formula:
# z5 = (h1 * w13)
#    + (h2 * w14)
#    + (h3 * w15)
#    + (h4 * w16)
#    + b5

z5 = (
    (h1 * w13)
    + (h2 * w14)
    + (h3 * w15)
    + (h4 * w16)
    + b5
)

# ReLU
h5 = max(0, z5)


# -----------------------------------
# Neuron 2
# -----------------------------------

# Weights
w17 = 0.4
w18 = 0.4
w19 = 0.4
w20 = 0.4

# Bias
b6 = 1

# Formula:
# z6 = (h1 * w17)
#    + (h2 * w18)
#    + (h3 * w19)
#    + (h4 * w20)
#    + b6

z6 = (
    (h1 * w17)
    + (h2 * w18)
    + (h3 * w19)
    + (h4 * w20)
    + b6
)

# ReLU
h6 = max(0, z6)


# -----------------------------------
# Neuron 3
# -----------------------------------

# Weights
w21 = 0.3
w22 = 0.3
w23 = 0.3
w24 = 0.3

# Bias
b7 = 1

# Formula:
# z7 = (h1 * w21)
#    + (h2 * w22)
#    + (h3 * w23)
#    + (h4 * w24)
#    + b7

z7 = (
    (h1 * w21)
    + (h2 * w22)
    + (h3 * w23)
    + (h4 * w24)
    + b7
)

# ReLU
h7 = max(0, z7)


# -----------------------------------
# Neuron 4
# -----------------------------------

# Weights
w25 = 0.2
w26 = 0.2
w27 = 0.2
w28 = 0.2

# Bias
b8 = 1

# Formula:
# z8 = (h1 * w25)
#    + (h2 * w26)
#    + (h3 * w27)
#    + (h4 * w28)
#    + b8

z8 = (
    (h1 * w25)
    + (h2 * w26)
    + (h3 * w27)
    + (h4 * w28)
    + b8
)

# ReLU
h8 = max(0, z8)


print("\nHidden Layer 2:")
print("h5 =", h5)
print("h6 =", h6)
print("h7 =", h7)
print("h8 =", h8)


# ===================================
# 4. OUTPUT LAYER
# 1 NEURON
# ===================================

# Weights
w29 = 0.5
w30 = 0.5
w31 = 0.5
w32 = 0.5

# Bias
b9 = 1


# OUTPUT FORMULA:
#
# output = (h5 * w29)
#        + (h6 * w30)
#        + (h7 * w31)
#        + (h8 * w32)
#        + b9
#
# No ReLU here because this example
# uses a linear output for predicting a number.


output = (
    (h5 * w29)
    + (h6 * w30)
    + (h7 * w31)
    + (h8 * w32)
    + b9
)


# ===================================
# 5. FINAL OUTPUT
# ===================================

print("\nPrediction =", output)


# ===================================
# SUMMARY
# ===================================

# Input Layer:
# x1 = Study hours
# x2 = Sleep hours
# x3 = Attendance
#
# Hidden Layer 1:
# 4 neurons -> h1, h2, h3, h4
#
# Hidden Layer 2:
# 4 neurons -> h5, h6, h7, h8
#
# Output Layer:
# 1 neuron -> Prediction
#
#
# Hidden neuron formula:
#
# z = (inputs * weights) + bias
# h = ReLU(z)
#
# ReLU(z) = max(0, z)
#
#
# Output neuron formula:
#
# Prediction = (hidden outputs * weights) + bias
#
#
# Weight = controls how strongly a connection affects a neuron
#
# Bias = extra adjustment added to a neuron
#
# ReLU = activation function
#        positive -> keep
#        negative -> 0
#
#
# In this example:
#
# Hidden Layer 2 outputs:
# h5 = 53
# h6 = 42.6
# h7 = 32.2
# h8 = 21.8
#
# Weighted sum:
# (53 * 0.5)
# + (42.6 * 0.5)
# + (32.2 * 0.5)
# + (21.8 * 0.5)
# = 74.8
#
# Add output bias:
# 74.8 + 1 = 75.8
#
# Final Prediction = 75.8
#
#
# IMPORTANT:
# These weights and biases are manually chosen
# only to understand how a neural network works.
#
# In real Deep Learning, training learns
# the weights and biases automatically.
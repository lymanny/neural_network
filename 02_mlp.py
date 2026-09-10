# -----------------------------------
# BASIC MLP EXAMPLE
# -----------------------------------

# Inputs
x1 = 5   # Study hours
x2 = 7   # Sleep hours


# -----------------------------------
# HIDDEN NEURON 1
# -----------------------------------

w11 = 2   # Weight for study
w12 = 1   # Weight for sleep
b1 = 0    # Bias

# Neuron 1 calculation
z1 = (x1 * w11) + (x2 * w12) + b1


# -----------------------------------
# HIDDEN NEURON 2
# -----------------------------------

w21 = 1   # Weight for study
w22 = 2   # Weight for sleep
b2 = 0    # Bias

# Neuron 2 calculation
z2 = (x1 * w21) + (x2 * w22) + b2


# -----------------------------------
# OUTPUT NEURON
# -----------------------------------

w3 = 1    # Weight from neuron 1
w4 = 1    # Weight from neuron 2
b3 = 0    # Bias

# Final neuron calculation
output = (z1 * w3) + (z2 * w4) + b3


# Results
print("Neuron 1 output:", z1)
print("Neuron 2 output:", z2)
print("Final output:", output)
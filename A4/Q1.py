import numpy as np

def sigmoid(x):
    return 1.0/(1.0 + (np.e**(-1*x)))

if __name__ == "__main__":
    x1, x2, x3, x4 = 0.9, -1.1, -0.3, 0.8
    w1, w2, w3, w4, w5, w6 = 0.75, -0.63, 0.24, -1.7, 0.8, -0.2
    y = 0.5

z1 = w1*x1+w2*x2
print("z1:", z1)
h1 = sigmoid(z1)
print("h1:", h1)

z2 = w3*x3+w4*x4
print("z2:", z2)
h2 = sigmoid(z2)
print("h2:", h2)

h3 = w5*h1+w6*h2
print("h3:", h3)
y_hat = sigmoid(h3)
print("y_hat:", y_hat)
L = (y-y_hat)**2
print("L:", L)

dy_hat = 2 * abs(y-y_hat)
print("dy_hat:", dy_hat)

dh3 = dy_hat * sigmoid(h3) * (1-sigmoid(h3))
print("dh3:", dh3)

dh2 = dh3 * w6
print("dh2:", dh2)

dz2 = dh2 * sigmoid(z2)*(1-sigmoid(z2))
print("dz2:", dz2)

dw3 = dz2 * x3
print("dw3:", dw3)
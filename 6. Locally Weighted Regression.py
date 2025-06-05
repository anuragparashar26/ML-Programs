import numpy as np
import matplotlib.pyplot as plt

def lwr(x, y, xq, tau):
    W = np.diag(np.exp(-(x - xq)**2 / (2 * tau**2)))
    X = np.c_[np.ones(len(x)), x]
    theta = np.linalg.pinv(X.T @ W @ X) @ (X.T @ W @ y)
    return np.array([1, xq]) @ theta

x = np.linspace(0, 10, 100)
y = np.sin(x) + np.random.normal(0, 0.2, 100)
x_test = np.linspace(0, 10, 100)
taus = [0.1, 0.5, 1, 5]

plt.figure(figsize=(12, 8))
for tau in taus:
    y_pred = [lwr(x, y, xq, tau) for xq in x_test]
    plt.plot(x_test, y_pred, label=f'tau={tau}')

plt.scatter(x, y, c='k', alpha=0.5, label='Training Data')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Locally Weighted Regression (LWR)')
plt.legend()
plt.show()

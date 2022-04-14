from matplotlib import pyplot as plt
import numpy as np
from functions import step_function, sigmoid

x = np.arange(-5.0, 5.0, 0.1)
y1 = step_function(x)
y2 = sigmoid(x)
plt.plot(x, y1)
plt.plot(x, y2, linestyle="--")
plt.ylim(-0.1, 1.1)
plt.show()

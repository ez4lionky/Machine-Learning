import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0 + 1e-10, 1 - 1e-10, 1000)
y = -(x * np.log2(x) + (1 - x) * np.log2(1 - x))

plt.figure()
plt.plot(x, y, 'k-')
plt.xlabel('$p$')
plt.ylabel('Entropy')
plt.title('Information entropy curve')
plt.show()
import numpy as np
import matplotlib.pyplot as plt

mean_error = np.loadtxt('./Result/mean_error.txt')
mean_error_AP3P = mean_error[:,0]
mean_error_AP3P = mean_error_AP3P[mean_error_AP3P > -1]
times = np.size(mean_error_AP3P,0)
print(times)

mean = np.mean(mean_error_AP3P)

print(mean)

fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(111)
ax.scatter(range(1, times + 1), mean_error_AP3P, c='cyan', s=10, 
           edgecolor='k', linewidth=1, marker='o')
ax.text(-10, mean, f'AP3P:{mean:.2f}',
        verticalalignment="top", horizontalalignment="right", c='k', fontsize=10)
ax.axhline(mean, c='k')
plt.tight_layout()
plt.show()
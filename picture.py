import numpy as np
import matplotlib.pyplot as plt

mean_error = np.loadtxt('./Result/mean_error.txt')

mean_error_AP3P = np.mean(mean_error[:,0])
mean_error_P3P = np.mean(mean_error[:,1])
mean_error_EPnP = np.mean(mean_error[:,2])

print(mean_error_AP3P)
print(mean_error_P3P)
print(mean_error_EPnP)

times = np.size(mean_error,0)

fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(111)

# AP3P
ax.scatter(range(1, times + 1), mean_error[:,0], c='g', s=10, 
           edgecolor='g', linewidth=1, marker='s', label='AP3P')
ax.text(0, mean_error_AP3P, f'AP3P:{mean_error_AP3P:.2f}',
        verticalalignment="top", horizontalalignment="right", c='k', fontsize=10)
ax.axhline(mean_error_AP3P, c='g')

# P3P
ax.scatter(range(1, times + 1), mean_error[:,1], c='r', s=10, 
           edgecolor='r', linewidth=1, marker='^', label='P3P')
ax.text(0, mean_error_P3P, f'P3P:{mean_error_P3P:.2f}',
        verticalalignment="bottom", horizontalalignment="right", c='k', fontsize=10)
ax.axhline(mean_error_P3P, c='r')

# EPNP
ax.scatter(range(1, times + 1), mean_error[:,2], c='b', s=10, 
           edgecolor='b', linewidth=1, marker='o', label='EPnP')
ax.text(0, mean_error_EPnP, f'EPnP:{mean_error_EPnP:.2f}',
        verticalalignment="bottom", horizontalalignment="right", c='k', fontsize=10)
ax.axhline(mean_error_EPnP, c='k')



plt.legend()
ax.set_title('Reprojection error', fontsize=14)
ax.set_xlabel('times', fontsize=10)
ax.set_ylabel('Reprojection error/pixel', fontsize=10)
ax.set_ylim(-11, 1000)
ax.set_xlim((-10, times + 10))

plt.tight_layout()
plt.show()
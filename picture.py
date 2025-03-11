import numpy as np
import matplotlib.pyplot as plt

oringinal_points = np.loadtxt('./Result/original_points.txt')
transformed_points = np.loadtxt('./Result/transformed_points.txt')
RotationMatrix = np.loadtxt('./Result/RotationMatrix.txt')
TranslationMatrix = np.loadtxt('./Result/TranslationMatrix.txt')
project_points = np.loadtxt('./Result/projected_points.txt')
mean_error = np.loadtxt('./Result/mean_error.txt')

mean_error_EPnP = np.mean(mean_error[:,0])
mean_error_P3P = np.mean(mean_error[:,1])
times = np.size(mean_error,0)

fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(111)
ax.scatter(range(1, times + 1), mean_error[:,0], c='cyan', s=10, 
           edgecolor='k', linewidth=1, marker='o')
ax.scatter(range(1, times + 1), mean_error[:,1], c='cyan', s=10, 
           edgecolor='r', linewidth=1, marker='^', alpha=0.8)
ax.axhline(mean_error_EPnP, c='k')
ax.axhline(mean_error_P3P, c='r')
ax.set_title('Reprojection error', fontsize=14)
ax.set_xlabel('times', fontsize=10)
ax.set_ylabel('Reprojection error/pixel', fontsize=10)
ax.set_ylim((-11, 200))
ax.set_xlim((-10, 1010))
ax.text(-50, mean_error_EPnP, f'EPnP:{mean_error_EPnP:.2f}',
        verticalalignment="bottom", horizontalalignment="right", c='k', fontsize=10)
ax.text(-50, mean_error_P3P, f'P3P:{mean_error_P3P:.2f}',
        verticalalignment="bottom", horizontalalignment="right", c='r', fontsize=10)
plt.tight_layout()
plt.show()
# ======================================
# 本算法用于检验测试得到的数据是否准确
# ======================================

import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
from scipy.spatial.transform import Rotation
from typing import Tuple
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# ================== 可视化的处理 ==================
def visualization(original_points: np.ndarray, transformed_points: np.ndarray) -> None:
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(121, projection="3d")

    # 绘制顶点
    ax.scatter(original_points[:,0], original_points[:,1], original_points[:,2], 
                c='r', s=50)
    # 绘制四面体的面
    ax.add_collection3d(Poly3DCollection([  
        [original_points[0], original_points[1], original_points[2]],  # 底面
        [original_points[0], original_points[1], original_points[3]],  # 侧面1
        [original_points[0], original_points[2], original_points[3]],  # 侧面2
        [original_points[1], original_points[2], original_points[3]]   # 顶面
    ], alpha=0.5, edgecolor='r'))
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_zlabel('Z', fontsize=12)
    ax.set_title('original_points', fontsize=14)

    ax2 = fig.add_subplot(122, projection="3d")
    # 绘制顶点
    ax2.scatter(transformed_points[:,0], transformed_points[:,1], transformed_points[:,2], 
                c='b', s=50)
    # 绘制四面体的面
    ax2.add_collection3d(Poly3DCollection([  
        [transformed_points[0], transformed_points[1], transformed_points[2]],  # 底面
        [transformed_points[0], transformed_points[1], transformed_points[3]],  # 侧面1
        [transformed_points[0], transformed_points[2], transformed_points[3]],  # 侧面2
        [transformed_points[1], transformed_points[2], transformed_points[3]]   # 顶面
    ], alpha=0.5, edgecolor='b'))
    ax2.set_xlabel('X', fontsize=12)
    ax2.set_ylabel('Y', fontsize=12)
    ax2.set_zlabel('Z', fontsize=12)
    ax2.set_title('original_points', fontsize=14)

    plt.show()

    

mean_error = np.loadtxt('./Result/mean_error.txt')
original_points = np.loadtxt('./Result/original_points.txt')
projected_points = np.loadtxt('./Result/projected_points.txt')
rotation_matrix = np.loadtxt('./Result/rotation_matrix.txt')
rvec_AP3P = np.loadtxt('./Result/rvec_AP3P.txt')
transformed_points = np.loadtxt('./Result/transformed_points.txt')
translation_matrix = np.loadtxt('./Result/translation_matrix.txt')
tvec_AP3P = np.loadtxt('./Result/tvec_AP3P.txt')

# 需要验证的次数
times = 18
times = times - 1

# 读取数据
mean_error = mean_error[times]

original_points = np.array([
    [original_points[times, 0], original_points[times, 1], original_points[times, 2]],
    [original_points[times, 3], original_points[times, 4], original_points[times, 5]],
    [original_points[times, 6], original_points[times, 7], original_points[times, 8]],
    [original_points[times, 9], original_points[times, 10], original_points[times, 11]]
])

rotation_matrix = np.array([
    [rotation_matrix[times, 0], rotation_matrix[times, 1], rotation_matrix[times, 2]],
    [rotation_matrix[times, 3], rotation_matrix[times, 4], rotation_matrix[times, 5]],
    [rotation_matrix[times, 6], rotation_matrix[times, 7], rotation_matrix[times, 8]]
])

translation_matrix = np.array([
    [translation_matrix[times, 0], translation_matrix[times, 1], translation_matrix[times, 2]]
])

transformed_points = np.array([
    [transformed_points[times, 0], transformed_points[times, 1], transformed_points[times, 2]],
    [transformed_points[times, 3], transformed_points[times, 4], transformed_points[times, 5]],
    [transformed_points[times, 6], transformed_points[times, 7], transformed_points[times, 8]],
    [transformed_points[times, 9], transformed_points[times, 10], transformed_points[times, 11]]
])

projected_points = np.array([
    [projected_points[times, 0], projected_points[times, 1]],
    [projected_points[times, 2], projected_points[times, 3]],
    [projected_points[times, 4], projected_points[times, 5]],
    [projected_points[times, 6], projected_points[times, 7]]
])

rvec_AP3P = np.array([
    rvec_AP3P[times, 0], rvec_AP3P[times, 1], rvec_AP3P[times, 2]
])

tvec_AP3P = np.array([
    tvec_AP3P[times, 0], tvec_AP3P[times, 1], tvec_AP3P[times, 2]
])

# 刚性变换可视化
# visualization(original_points, transformed_points)

# PnP逆变换
R, _ = cv2.Rodrigues(rvec_AP3P)

retransformed_points = (R @ original_points.T).T + tvec_AP3P

print('mean_error: ', mean_error)
print('original_points: ', original_points)
print('rotation_matrix: ', rotation_matrix)
print('R: ', R)
print('translation_matrix: ', translation_matrix)
print('tvec_AP3P: ', tvec_AP3P)
print('transformed_points: ', transformed_points)
print('retransformed_points: ', retransformed_points)
print('projected_points: ', projected_points)
print('rvec_AP3P: ', rvec_AP3P)


# 变换检验
print(
    '原始点对距离: ',
    np.linalg.norm(original_points[0]-original_points[1]),
    np.linalg.norm(original_points[0]-original_points[2]),
    np.linalg.norm(original_points[0]-original_points[3]),
    np.linalg.norm(original_points[1]-original_points[2]),
    np.linalg.norm(original_points[1]-original_points[3]),
    np.linalg.norm(original_points[2]-original_points[3])
)

print(
    '变换点对距离: ',
    np.linalg.norm(transformed_points[0]-transformed_points[1]),
    np.linalg.norm(transformed_points[0]-transformed_points[2]),
    np.linalg.norm(transformed_points[0]-transformed_points[3]),
    np.linalg.norm(transformed_points[1]-transformed_points[2]),
    np.linalg.norm(transformed_points[1]-transformed_points[3]),
    np.linalg.norm(transformed_points[2]-transformed_points[3])
)

print(
    '重变换点对距离: ',
    np.linalg.norm(retransformed_points[0]-retransformed_points[1]),
    np.linalg.norm(retransformed_points[0]-retransformed_points[2]),
    np.linalg.norm(retransformed_points[0]-retransformed_points[3]),
    np.linalg.norm(retransformed_points[1]-retransformed_points[2]),
    np.linalg.norm(retransformed_points[1]-retransformed_points[3]),
    np.linalg.norm(retransformed_points[2]-retransformed_points[3])
)
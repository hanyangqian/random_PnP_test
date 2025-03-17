import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
from scipy.spatial.transform import Rotation
from typing import Tuple
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# ================== 第三部分：可视化的处理 ==================
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
times = 2
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

print(mean_error)
print(original_points)
print(rotation_matrix)
print(translation_matrix)
print(transformed_points)
print(projected_points)
print(rvec_AP3P)
print(tvec_AP3P)

# 刚性变换可视化
visualization(original_points, transformed_points)

# PnP逆变换
R, _ = cv2.Rodrigues(rvec_AP3P)
print(R)
retransformed_points = (R @ original_points.T).T + tvec_AP3P
print(retransformed_points)
projected_points, _ = cv2.projectPoints(transformed_points, R, t, K, dist_coeffs)


'''
orginal_points
-40.982999 -74.780183 -26.625237 -28.844122 58.429718 -5.860308 27.349563 87.206130 46.402504 17.430229 4.038518 12.594936 
transformed_points
-77.252358 -514.783941 345.456301 -30.063490 -429.767292 251.282034 -28.501086 -452.093516 172.438441 -69.226433 -491.674289 242.673796 
R
-0.714641 0.322619 0.620649 -0.202284 0.754057 -0.624884 -0.669604 -0.572115 -0.473619 
t
-65.890023 -483.323331 262.620846

projected
467.639633 -1525.645494 347.075143 -1471.671695 252.879792 -1215.430072 358.890048 -1269.078540
'''
'''
world_points = np.array([
    [-40.982999, -74.780183, -26.625237],
    [-28.844122, 58.429718, -5.860308],
    [27.349563, 87.206130, 46.402504],
    [17.430229, 4.038518, 12.594936]
])
image_points = np.array([
    [467.639633, -1525.645494],
    [347.075143, -1471.671695],
    [252.879792, -1215.430072],
    [358.890048, -1269.078540]
])

# 相机内参
fu, fv = 800, 800  # 焦距
img_width, img_height = 640, 480  # 图像分辨率
u0, v0 = fu/2 , fv/2  # 像主点坐标
K = np.array([
    [fu, 0, u0],
    [0, fv, v0],
    [0, 0, 1]
])

# 相机外参（假定世界坐标系与相机坐标系重合）
camera_pose = np.eye(4)
camera_pose[:3, 3] = [0, 0, -2000]
M = camera_pose

camera_matrix = K  # 相机内参
    
# 畸变参数（默认无畸变）
dist_coeffs = np.zeros((5,1), dtype=np.float32)

retval_EPnP, rvec_EPnP, tvec_EPnP = cv2.solvePnP(world_points, image_points, camera_matrix, None, flags=cv2.SOLVEPNP_AP3P)
reprojected_points, _ = cv2.projectPoints(world_points, rvec_EPnP, tvec_EPnP, camera_matrix, dist_coeffs)

print(reprojected_points)

R_mat = cv2.Rodrigues(rvec_EPnP)[0]

print(retval_EPnP)
print(rvec_EPnP)
print(tvec_EPnP)
print(R_mat)
'''

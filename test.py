import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
from scipy.spatial.transform import Rotation
from typing import Tuple

# ================== 第一部分：点的处理 ==================
# 生成虚拟点，模拟CAD模型，为世界坐标系点坐标
# 生成随机旋转矩阵
# 生成随机平移矩阵
# 为刚性变换后的坐标点添加高斯噪声

# 生成三维虚拟点
def points_generation(n, x_min=-50, x_max=50, y_min=-50, y_max=50, z_min=-50, z_max=50):
    x = np.random.uniform(x_min, x_max, n)
    y = np.random.uniform(y_min, y_max, n)
    z = np.random.uniform(z_min, z_max, n)
    points = np.column_stack((x, y, z))
    return points

# 随机生成旋转矩阵函数
def generate_rotation_matrix() -> np.ndarray:
    # 生成随机旋转实例
    random_rot = Rotation.random()
    # 转换为旋转矩阵
    rot_matrix_scipy = random_rot.as_matrix()
    return rot_matrix_scipy

# 随机生成平移矩阵
def generate_translation(trans_range=(-500, 500)):
    """
    生成随机平移向量
    参数：
        trans_range: 平移范围 (默认[-500, 500])
    返回：
        3维平移向量
    """
    t = np.random.uniform(*trans_range, 3)
    return t

# 添加高斯噪声
def add_noise(points, mu=0, sigma=0.1):
    """
    为点云添加高斯噪声
    参数：
        points: 输入点云
        mu: 均值 (默认0)
        sigma: 标准差 (默认0.1)
    返回：
        添加噪声后的点云
    """
    noise = np.random.normal(mu, sigma, points.shape)
    return points + noise

# ================== 第二部分：投影矩阵计算 ==================
def project(transformed_points, K, camera_pose):
    """投影函数（与原始代码保持一致）"""
    R = camera_pose[:3, :3]
    t = camera_pose[:3, 3]
    points_cam = (R @ transformed_points.T).T + t
    points_2d_hom = K @ (points_cam / points_cam[:, 2].reshape(-1, 1)).T
    projected_points = (points_2d_hom[:2, :] / points_2d_hom[2, :]).T
    return projected_points

# ================== 第三部分：可视化的处理 ==================
def visualization(original_points: np.ndarray, transformed_points: np.ndarray, 
                  projected_points: np.ndarray) -> None:
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(131, projection="3d")
    ax.scatter(original_points[:,0], original_points[:,1], original_points[:,2], 
               c='r', s=50, edgecolor='k', alpha=0.8)
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_zlabel('Z', fontsize=12)
    ax.set_title('original_points', fontsize=14)
    ax.view_init(elev=20, azim=45)  # 设置视角
    
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.scatter(transformed_points[:,0], transformed_points[:,1], transformed_points[:,2], 
            c='r', s=50, edgecolor='k', alpha=0.8)
    ax2.set_xlabel('X', fontsize=12)
    ax2.set_ylabel('Y', fontsize=12)
    ax2.set_zlabel('Z', fontsize=12)
    ax2.set_title('transformed_points', fontsize=14)
    ax2.view_init(elev=20, azim=45)  # 设置视角
    
    ax3 = fig.add_subplot(133)
    ax3.scatter(projected_points[:,0], projected_points[:,1], 
                c='cyan', s=100, edgecolor='k', linewidth=1, 
                marker='o', label='Projected Points')
    ax3.set_title('Image Coordinate System', fontsize=14)
    ax3.axis('off')
    
    plt.tight_layout()
    plt.show()
    
# ================== 第四部分：重投影误差的计算 ==================
def average_reprojection_error(original_points: np.ndarray, rvec: np.ndarray, tvec: np.ndarray, 
                               camera_matrix: np.ndarray, image_points: np.ndarray) -> None:
    # 重投影计算
    dist_coeffs = np.zeros((5,1), dtype=np.float32)
    reprojected_points, _ = cv2.projectPoints(original_points, rvec, tvec, camera_matrix, dist_coeffs)

    # 计算误差
    errors = [np.linalg.norm(p - image_points[i]) for i, p in enumerate(reprojected_points.squeeze())]
    mean_error = np.mean(errors)
    return mean_error

# ================== 第五部分：姿态估计计算 ==================
def EPnP(world_points: np.ndarray, image_points: np.ndarray, camera_matrix: np.ndarray) -> Tuple[bool, np.ndarray, np.ndarray, float]:
    time_start = time.time()  # 计时开始
    # EPnP计算
    retval, rvec, tvec = cv2.solvePnP(world_points, image_points, camera_matrix, None, flags=cv2.SOLVEPNP_P3P)
    time_end = time.time()  # 计时结束
    time_sum = time_end - time_start
    return retval, rvec, tvec, time_sum
    
def P3P(world_points: np.ndarray, image_points: np.ndarray, camera_matrix: np.ndarray) -> Tuple[bool, np.ndarray, np.ndarray, float]:
    time_start = time.time()  # 计时开始
    # P3P计算
    retval, rvec, tvec = cv2.solvePnP(world_points, image_points, camera_matrix, None, flags=cv2.SOLVEPNP_EPNP)
    time_end = time.time()  # 计时结束
    time_sum = time_end - time_start
    return retval, rvec, tvec, time_sum

def AP3P(world_points: np.ndarray, image_points: np.ndarray, camera_matrix: np.ndarray) -> Tuple[bool, np.ndarray, np.ndarray, float]:
    time_start = time.time()  # 计时开始
    # AP3P计算
    retval, rvec, tvec = cv2.solvePnP(world_points, image_points, camera_matrix, None, flags=cv2.SOLVEPNP_AP3P)
    time_end = time.time()  # 计时结束
    time_sum = time_end - time_start
    return retval, rvec, tvec, time_sum

if __name__ == '__main__':
    
    jishuqi = 1
    
    while(jishuqi <= 100):
        # 点个数
        points_num = 4  # 随机点个数
        
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
        
        # 生成原始点云
        original_points = points_generation(points_num)
        np.savetxt("./Result/original_points.txt", original_points, fmt='%.6f')
        
        # 生成变换矩阵
        R = generate_rotation_matrix()
        t = generate_translation()
        np.savetxt("./Result/RotationMatrix.txt", R, fmt="%.6f")
        np.savetxt("./Result/TranslationMatrix.txt", t, fmt="%.6f")
        
        
        # 应用刚体变换
        transformed_points = (R @ original_points.T).T + t
        np.savetxt("./Result/transformed_points.txt", transformed_points, fmt='%.6f')
        
        ''' 
        # 添加高斯噪声
        noisy_points = add_noise(transformed_points)
        '''
        
        # 投影矩阵计算
        projected_points = project(transformed_points, K, np.linalg.inv(camera_pose))
        np.savetxt("./Result/projected_points.txt", projected_points, fmt='%.6f')

        # 三维坐标可视化
        visualization(original_points, transformed_points, projected_points)
        
        # ================== Opencv姿态估计算法 ==================
        world_points = original_points  # 3D点（世界坐标系）
        image_points = projected_points  # 对应的2D像素坐标
        camera_matrix = K  # 相机内参

        # ================== EPnP ==================
        retval_EPnP, rvec_EPnP, tvec_EPnP, time_sum_EPnP = EPnP(world_points, image_points, camera_matrix)
        
        if retval_EPnP == 1:
            # 重投影误差计算
            mean_error_EPnP = average_reprojection_error(world_points, rvec_EPnP, tvec_EPnP, camera_matrix, image_points)

        # ================== P3P ==================
        retval_P3P, rvec_P3P, tvec_P3P, time_sum_P3P = P3P(world_points, image_points, camera_matrix)

        # 重投影误差计算
        mean_error_P3P = average_reprojection_error(world_points, rvec_P3P, tvec_P3P, camera_matrix, image_points)


        print(jishuqi)
        jishuqi = jishuqi + 1
        
        
        
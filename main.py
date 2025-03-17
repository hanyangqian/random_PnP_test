# ======================================
# 主程序
# ======================================

import numpy as np
import cv2
from scipy.spatial.transform import Rotation

# ================== 第一部分：点的处理 ==================
# 生成虚拟点，模拟CAD模型，为世界坐标系点坐标
# 生成随机旋转矩阵
# 生成随机平移矩阵
# 为刚性变换后的坐标点添加高斯噪声

# 生成三维虚拟点
def points_generation(n, x_min=-50, x_max=50, y_min=-100, y_max=100, z_min=-50, z_max=50) -> np.ndarray:
    x = np.random.uniform(x_min, x_max, n)
    y = np.random.uniform(y_min, y_max, n)
    z = np.random.uniform(z_min, z_max, n)
    points = np.column_stack((x, y, z))
    return points

# 生成旋转矩阵函数
def generate_rotation_matrix() -> np.ndarray:
    # 生成随机旋转实例
    random_rot = Rotation.random()
    # 转换为旋转矩阵
    rot_matrix_scipy = random_rot.as_matrix()
    return rot_matrix_scipy

# 生成平移矩阵
def generate_translation_matrix() -> np.ndarray:
    x = np.random.uniform(-500, 500)
    y = np.random.uniform(-500, 500)
    z = np.random.uniform(2000, 3000)
    t = np.array([x, y, z])
    return t

# 添加高斯噪声
def add_noise(points: np.ndarray, mu=0, sigma=0.1) -> np.ndarray:
    """
        points: 输入点云
        mu: 均值 (默认0)
        sigma: 标准差 (默认0.1)
    """
    noise = np.random.normal(mu, sigma, points.shape)
    return points + noise
    
# ================== 第三部分：重投影误差的计算 ==================
def average_reprojection_error(original_points: np.ndarray, rvec: np.ndarray, tvec: np.ndarray, 
                               camera_matrix: np.ndarray, image_points: np.ndarray):
    # 重投影计算
    dist_coeffs = np.zeros((5,1), dtype=np.float32)
    reprojected_points, _ = cv2.projectPoints(original_points, rvec, tvec, camera_matrix, dist_coeffs)

    # 计算误差
    errors = [np.linalg.norm(p - image_points[i]) for i, p in enumerate(reprojected_points.squeeze())]
    mean_error = np.mean(errors)
    return mean_error

if __name__ == '__main__':
    # ========================================== 参数设置 ============================================
    
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
    camera_pose[:3, 3] = [0, 0, 0]
    M = camera_pose
    
    # 畸变参数（默认无畸变）
    dist_coeffs = np.zeros((5,1), dtype=np.float32)
    
    # txt文裆重置
    with open('./Result/mean_error.txt', 'w') as f:
        pass
    with open('./Result/original_points.txt', 'w') as f:
        pass
    with open('./Result/projected_points.txt', 'w') as f:
        pass
    with open('./Result/rotation_matrix.txt', 'w') as f:
        pass
    with open('./Result/transformed_points.txt', 'w') as f:
        pass
    with open('./Result/translation_matrix.txt', 'w') as f:
        pass
    with open('./Result/rvec_AP3P.txt', 'w') as f:
        pass
    with open('./Result/tvec_AP3P.txt', 'w') as f:
        pass
    
    # 文档说明
    # mean_error.txt 保存数据为2列，第一列为AP3P算法重投影误差，(第二列为EPnP算法重投影误差)
    # original_points.txt 保存数据为12列，每行依次为四个点的x, y, z坐标值
    # projected_points.txt 保存数据为8列，每行依次为四个点投影到图像坐标系的x, y坐标值
    # rotation_matrix.txt 保存数据为9列，每行依次为旋转矩阵9个值
    # rvec_AP3P 保存数据为3列，每行依次为cv2.AP3P计算出的旋转向量
    # transformed_points.txt 保存数据为12列，每行依次为四个点变换后的x, y, z坐标值
    # translation_matrix.txt 保存数据为3列，每行依次为平移矩阵
    # tvec_AP3P 保存数据为3列，每行依次为cv2.AP3P计算出的平移向量
        
    # ========================================== 循环计算 ============================================
    # 循环计数
    jishuqi = 1
    
    while(jishuqi <= 1000):
        # 生成原始点云
        original_points = points_generation(points_num)
        
        with open('./Result/original_points.txt', 'a') as f:
            np.savetxt(f, original_points, fmt='%.6f', newline=' ')
            f.write('\n')
        
        # 生成变换矩阵
        R = generate_rotation_matrix()
        t = generate_translation_matrix()
        
        with open('./Result/rotation_matrix.txt', 'a') as f:
            np.savetxt(f, R, fmt='%.6f', newline=' ')
            f.write('\n')
        with open('./Result/translation_matrix.txt', 'a') as f:
            np.savetxt(f, t, fmt='%.6f', newline=' ')
            f.write('\n')
        
        # 应用刚体变换
        transformed_points = (R @ original_points.T).T + t
        
        with open('./Result/transformed_points.txt', 'a') as f:
            np.savetxt(f, transformed_points, fmt='%.6f', newline=' ')
            f.write('\n')
        
        ''' 
        # 添加高斯噪声
        noisy_points = add_noise(transformed_points)
        '''
        
        # ================== Opencv姿态估计算法 ==================
        camera_pose_R = camera_pose[:3, :3]
        camera_pose_t = camera_pose[:3, 3]
        
        # 投影矩阵计算
        projected_points, _ = cv2.projectPoints(transformed_points, camera_pose_R, camera_pose_t, K, dist_coeffs)
        
        for i in range(0, points_num):
            with open('./Result/projected_points.txt', 'a') as f:
                np.savetxt(f, projected_points[i], fmt='%.6f', newline=' ')
        with open('./Result/projected_points.txt', 'a') as f:
            f.write('\n')
        
        world_points = original_points  # 3D点（世界坐标系）
        image_points = projected_points  # 对应的2D像素坐标
        camera_matrix = K  # 相机内参
        
        # ================== AP3P ==================
        retval_AP3P, rvec_AP3P, tvec_AP3P = cv2.solvePnP(world_points, image_points, camera_matrix, None, flags=cv2.SOLVEPNP_AP3P)
        
        if retval_AP3P == True:
            # 重投影误差计算
            mean_error_AP3P = average_reprojection_error(world_points, rvec_AP3P, tvec_AP3P, camera_matrix, image_points)
        else:
            mean_error_AP3P = -10
            rvec_AP3P = np.array([0, 0, 0])
            tvec_AP3P = np.array([0, 0, 0])
            
        
        # ================== P3P ==================
        retval_P3P, rvec_P3P, tvec_P3P = cv2.solvePnP(world_points, image_points, camera_matrix, None, flags=cv2.SOLVEPNP_P3P)
        
        if retval_P3P == True:
            # 重投影误差计算
            mean_error_P3P = average_reprojection_error(world_points, rvec_P3P, tvec_P3P, camera_matrix, image_points)
        else:
            mean_error_P3P = -10
            
        
        # ================== EPnP ==================
        retval_EPnP, rvec_EPnP, tvec_EPnP = cv2.solvePnP(world_points, image_points, camera_matrix, None, flags=cv2.SOLVEPNP_EPNP)
        
        if retval_EPnP == True:
            # 重投影误差计算
            mean_error_EPnP = average_reprojection_error(world_points, rvec_EPnP, tvec_EPnP, camera_matrix, image_points)
        else:
            mean_error_EPnP = -10
        
        
        
        # 保存数据  
        with open('./Result/rvec_AP3P.txt', 'a') as f:
            np.savetxt(f, rvec_AP3P, fmt='%.4f', newline=' ')
            f.write('\n')
        
        with open('./Result/tvec_AP3P.txt', 'a') as f:
            np.savetxt(f, tvec_AP3P, fmt='%.4f', newline=' ')
            f.write('\n')
        
        mean_error = np.array([mean_error_AP3P, mean_error_P3P, mean_error_EPnP])
        with open('./Result/mean_error.txt', 'a') as f:
            np.savetxt(f, mean_error, fmt='%.4f', newline=' ')
            f.write('\n')
        
        print(jishuqi)
        jishuqi = jishuqi + 1
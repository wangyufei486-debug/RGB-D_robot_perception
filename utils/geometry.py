import numpy as np


def get_depth_median(depth, u, v, window_size=7, depth_scale=5000.0):
    """读取中心像素邻域内的有效深度中位数，返回单位为米的深度。"""
    # 截取中心像素附近的窗口，并限制窗口不超出图像边界
    half = window_size // 2
    x1 = max(0, u - half)
    x2 = min(depth.shape[1], u + half + 1)
    y1 = max(0, v - half)
    y2 = min(depth.shape[0], v + half + 1)

    # 深度值为 0 表示该像素没有有效测量结果
    depth_region = depth[y1:y2, x1:x2]
    valid_depths = depth_region[depth_region > 0]
    if len(valid_depths) == 0:
        return None

    # 中位数比单个中心像素更不容易受到深度噪声影响
    return np.median(valid_depths) / depth_scale


def pixel_to_camera(u, v, depth, fx, fy, cx, cy):
    """将像素坐标和深度转换为相机坐标系下的三维坐标。"""
    # 根据针孔相机模型进行反投影，depth 就是相机坐标系的 Z
    x = (u - cx) * depth / fx
    y = (v - cy) * depth / fy
    return x, y, depth

def pose_to_transform(tx, ty, tz, qx, qy, qz, qw):
    """将 TUM 位姿转换为相机坐标系到世界坐标系的 4×4 变换矩阵。"""
    quaternion = np.array([qx, qy, qz, qw], dtype=np.float64)
    quaternion /= np.linalg.norm(quaternion)
    qx, qy, qz, qw = quaternion

    rotation = np.array([
        [
            1 - 2 * (qy**2 + qz**2),
            2 * (qx * qy - qz * qw),
            2 * (qx * qz + qy * qw),
        ],
        [
            2 * (qx * qy + qz * qw),
            1 - 2 * (qx**2 + qz**2),
            2 * (qy * qz - qx * qw),
        ],
        [
            2 * (qx * qz - qy * qw),
            2 * (qy * qz + qx * qw),
            1 - 2 * (qx**2 + qy**2),
        ],
    ])

    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = [tx, ty, tz]
    return transform

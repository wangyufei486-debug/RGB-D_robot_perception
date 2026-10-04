import cv2
import numpy as np
import open3d as o3d


def create_scene_pointcloud(rgb, depth, intrinsic, depth_scale=5000.0, depth_trunc=5.0):
    """由一组 RGB-D 图像生成完整场景点云。"""
    # OpenCV 使用 BGR 排列，Open3D 使用 RGB 排列
    rgb_for_o3d = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    color_o3d = o3d.geometry.Image(np.ascontiguousarray(rgb_for_o3d))
    depth_o3d = o3d.geometry.Image(np.ascontiguousarray(depth))

    # 将彩色图和深度图组合成 Open3D 的 RGB-D 图像
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d, depth_o3d, depth_scale=depth_scale, depth_trunc=depth_trunc,
        convert_rgb_to_intensity=False,
    )
    return o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)


def mask_to_pointcloud(rgb, depth, mask, fx, fy, cx, cy, depth_scale=5000.0):
    """将实例分割 Mask 内的有效 RGB-D 像素转换为目标点云。"""
    # 将 TUM 原始深度转换为米，并只保留 Mask 内的有效深度像素
    z = depth.astype(np.float32) / depth_scale
    valid = mask & (depth > 0)
    if np.count_nonzero(valid) == 0:
        return None

    # 为整幅深度图生成像素坐标，再利用相机内参反投影到三维空间
    height, width = depth.shape
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    points = np.stack([x[valid], y[valid], z[valid]], axis=1)

    # 为每个三维点保存对应像素的 RGB 颜色
    rgb_for_o3d = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    colors = rgb_for_o3d[valid].astype(np.float64) / 255.0
    object_pcd = o3d.geometry.PointCloud()
    object_pcd.points = o3d.utility.Vector3dVector(points)
    object_pcd.colors = o3d.utility.Vector3dVector(colors)
    return object_pcd


def clean_object_pointcloud(object_pcd, voxel_size, nb_neighbors, std_ratio, dbscan_eps,
                            dbscan_min_points):
    """依次执行体素降采样、统计滤波，并保留 DBSCAN 最大点云簇。"""
    # 体素降采样减少重复点，降低后续滤波和聚类的计算量
    object_pcd_down = object_pcd.voxel_down_sample(voxel_size=voxel_size)

    # 删除与邻域点平均距离明显偏大的离群点
    clean_pcd, _ = object_pcd_down.remove_statistical_outlier(
        nb_neighbors=nb_neighbors, std_ratio=std_ratio
    )
    print("去噪后点数：", len(clean_pcd.points))

    # DBSCAN 将空间上相邻的点分组，标签 -1 表示聚类噪声
    labels = np.array(clean_pcd.cluster_dbscan(
        eps=dbscan_eps, min_points=dbscan_min_points, print_progress=False
    ))
    valid_labels = labels[labels >= 0]
    if len(valid_labels) > 0:
        # 最大点云簇最有可能是完整目标，其余小簇作为背景或噪声丢弃
        largest_label = np.bincount(valid_labels).argmax()
        largest_cluster_index = np.where(labels == largest_label)[0]
        clean_pcd = clean_pcd.select_by_index(largest_cluster_index)

    print(f"最大聚类保留后点数：{len(clean_pcd.points)}")
    return clean_pcd


def compute_object_geometry(object_pcd):
    """计算目标三维中心、轴对齐包围盒和旋转包围盒。"""
    center = object_pcd.get_center()

    # AABB 始终与相机坐标轴平行，显示为绿色
    aabb = object_pcd.get_axis_aligned_bounding_box()
    aabb.color = (0, 1, 0)

    # OBB 可以随目标方向旋转，显示为蓝色
    obb = object_pcd.get_oriented_bounding_box(robust=True)
    obb.color = (0, 0, 1)
    return center, aabb, obb

from pathlib import Path

import cv2
import open3d as o3d
from ultralytics import YOLO

from utils.dataset import load_rgbd_frame
from utils.geometry import get_depth_median, pixel_to_camera
from utils.pointcloud import (
    clean_object_pointcloud,
    compute_object_geometry,
    create_scene_pointcloud,
    mask_to_pointcloud,
)
from utils.visualization import save_segmentation_result, show_3d_result


# 项目路径：以 main.py 所在位置为根目录，避免从其他目录运行时找不到文件
project_root = Path(__file__).resolve().parent
dataset_dir = project_root / "data" / "rgbd_dataset_freiburg1_xyz"
model_path = project_root / "yolo26n-seg.pt"
output_dir = project_root / "outputs"
output_2d_dir = output_dir / "2D"
output_3d_dir = output_dir / "3D"
output_point_dir = output_dir / "point"

# 选择需要处理的 RGB 帧和目标类别
frame_index = 0
target_class = "keyboard"

# Freiburg1 相机内参：图像尺寸、焦距和主点坐标
image_width = 640
image_height = 480
fx = 517.3
fy = 516.5
cx = 318.6
cy = 255.3

# TUM 深度值除以 5000 转为米，只保留 5 米以内的场景点
depth_scale = 5000.0
depth_trunc = 5.0

# 点云处理参数：体素越大点数越少，计算速度越快
scene_voxel_size = 0.01
object_voxel_size = 0.005

# 统计滤波参数：根据邻域内 20 个点的距离分布去除离群点
outlier_nb_neighbors = 20
outlier_std_ratio = 2.0

# DBSCAN 参数：在 0.03 米邻域内至少有 10 个点才形成聚类
dbscan_eps = 0.03
dbscan_min_points = 10


def main():
    # 创建二维图像、三维截图和目标点云输出目录
    for directory in (output_2d_dir, output_3d_dir, output_point_dir):
        directory.mkdir(parents=True, exist_ok=True)

    # 加载 YOLO26 实例分割模型并创建 Open3D 相机内参
    model = YOLO(str(model_path))
    intrinsic = o3d.camera.PinholeCameraIntrinsic(image_width, image_height, fx, fy, cx, cy)

    # 读取并匹配RGB-D图像
    rgb, depth = load_rgbd_frame(dataset_dir, frame_index)

    # 生成完整场景点云
    scene_pcd = create_scene_pointcloud(rgb, depth, intrinsic, depth_scale, depth_trunc)
    scene_pcd = scene_pcd.voxel_down_sample(voxel_size=scene_voxel_size)

    # YOLO实例分割
    result = model(rgb, conf=0.5, retina_masks=True, verbose=False)[0]
    annotated_frame = result.plot()
    save_segmentation_result(annotated_frame, output_2d_dir, frame_index)
    cv2.imshow("YOLO Segmentation Result", annotated_frame)

    # 遍历所有检测目标，查找指定类别
    for index, box in enumerate(result.boxes):
        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        confidence = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        # 使用二维检测框中心作为目标中心像素
        u = int((x1 + x2) / 2)
        v = int((y1 + y2) / 2)

        # 计算中心像素的深度与三维坐标
        center_depth = get_depth_median(depth, u, v, window_size=7, depth_scale=depth_scale)
        if center_depth is None:
            print(f"{class_name:<12}pixel=({u},{v})  没有有效深度")
            continue
        _center_xyz = pixel_to_camera(u, v, center_depth, fx, fy, cx, cy)

        # 只对指定类别继续执行三维点云处理
        if class_name != target_class:
            continue

        print("\n========== RGB-D 三维目标感知结果 ==========")
        print(f"\n目标：{class_name}")
        print(f"置信度：{confidence:.2f}")

        # 提取目标Mask
        mask = result.masks.data[index].cpu().numpy() > 0.5
        print("Mask shape：", mask.shape)
        print("Depth shape：", depth.shape)

        # 生成目标三维点云
        object_pcd = mask_to_pointcloud(rgb, depth, mask, fx, fy, cx, cy, depth_scale)
        if object_pcd is None:
            print(f"{target_class}局部点云提取失败")
            continue

        # 点云降采样与去噪
        clean_pcd = clean_object_pointcloud(
            object_pcd, object_voxel_size, outlier_nb_neighbors, outlier_std_ratio,
            dbscan_eps, dbscan_min_points,
        )

        # 计算目标三维中心、AABB和OBB
        center, aabb, obb = compute_object_geometry(clean_pcd)
        print(f"\n目标点云三维中心：({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}) m")
        print(f"OBB估计尺寸：{obb.extent[0]:.3f} × {obb.extent[1]:.3f} × {obb.extent[2]:.3f} m")

        # 只保存分割并处理后的目标点云
        point_path = output_point_dir / f"frame_{frame_index:04d}_{target_class}.ply"
        if o3d.io.write_point_cloud(str(point_path), clean_pcd):
            print(f"目标点云已保存：{point_path}")
        else:
            print("目标点云保存失败")

        # 显示三维结果并保存当前视角截图
        screenshot_path = output_3d_dir / f"frame_{frame_index:04d}_3d.png"
        show_3d_result(scene_pcd, clean_pcd, aabb, obb, screenshot_path)


if __name__ == "__main__":
    main()

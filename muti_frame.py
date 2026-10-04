from pathlib import Path

import cv2
import open3d as o3d
from ultralytics import YOLO

from utils.dataset import read_file_list, read_groundtruth
from utils.geometry import pose_to_transform
from utils.pointcloud import (
    clean_object_pointcloud,
    compute_object_geometry,
    create_scene_pointcloud,
    mask_to_pointcloud,
)
from utils.visualization import show_3d_result


# 项目和 TUM 数据集路径
project_root = Path(__file__).resolve().parent
dataset_dir = project_root / "data" / "rgbd_dataset_freiburg1_xyz"
model_path = project_root / "yolo26n-seg.pt"
output_3d_dir = project_root / "outputs" / "3D"
output_point_dir = project_root / "outputs" / "point"

# 从第 0 帧开始，每隔 2 帧取一帧，共处理 50 帧
start_frame = 0
num_frames = 50
frame_step = 2
target_class = "keyboard"

# Freiburg1 相机内参
image_width = 640
image_height = 480
fx = 517.3
fy = 516.5
cx = 318.6
cy = 255.3

# 保持与 main.py 相同的深度和点云参数
depth_scale = 5000.0
depth_trunc = 5.0
scene_voxel_size = 0.01
fusion_voxel_size = 0.01
object_voxel_size = 0.005
outlier_nb_neighbors = 20
outlier_std_ratio = 2.0
dbscan_eps = 0.03
dbscan_min_points = 10

# RGB 与最近位姿的时间差超过该阈值时跳过当前帧
pose_time_threshold = 0.02


def main():
    # 索引文件只读取一次，避免在多帧循环中重复打开文件
    rgb_list = read_file_list(dataset_dir / "rgb.txt")
    depth_list = read_file_list(dataset_dir / "depth.txt")
    groundtruth_list = read_groundtruth(dataset_dir / "groundtruth.txt")
    output_3d_dir.mkdir(parents=True, exist_ok=True)
    output_point_dir.mkdir(parents=True, exist_ok=True)

    if start_frame < 0 or start_frame >= len(rgb_list):
        raise ValueError(f"start_frame={start_frame} 超出范围，RGB图片总数为 {len(rgb_list)}")
    if num_frames <= 0 or frame_step <= 0:
        raise ValueError("num_frames 和 frame_step 必须大于 0")

    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        image_width, image_height, fx, fy, cx, cy
    )
    model = YOLO(str(model_path))
    fused_scene_pcd = o3d.geometry.PointCloud()
    fused_object_pcd = o3d.geometry.PointCloud()
    end_frame = min(start_frame + num_frames * frame_step, len(rgb_list))
    frame_indices = range(start_frame, end_frame, frame_step)
    fused_count = 0
    target_frame_count = 0
    target_instance_count = 0

    for frame_index in frame_indices:
        # 为当前 RGB 帧匹配时间戳最近的 Depth 和相机位姿
        rgb_time, rgb_path = rgb_list[frame_index]
        depth_time, depth_path = min(depth_list, key=lambda item: abs(item[0] - rgb_time))
        pose = min(groundtruth_list, key=lambda item: abs(item[0] - rgb_time))
        pose_time, tx, ty, tz, qx, qy, qz, qw = pose

        pose_time_diff = abs(pose_time - rgb_time)
        if pose_time_diff > pose_time_threshold:
            print(f"第 {frame_index} 帧位姿时间差过大：{pose_time_diff:.6f} s，已跳过")
            continue

        # 读取当前帧 RGB-D 图像
        rgb = cv2.imread(str(dataset_dir / rgb_path))
        depth = cv2.imread(str(dataset_dir / depth_path), cv2.IMREAD_UNCHANGED)
        if rgb is None or depth is None:
            raise FileNotFoundError(f"第 {frame_index} 帧 RGB 或 Depth 图像读取失败")

        # 生成当前相机坐标系下的场景点云并先进行单帧降采样
        frame_pcd = create_scene_pointcloud(rgb, depth, intrinsic, depth_scale, depth_trunc)
        frame_pcd = frame_pcd.voxel_down_sample(voxel_size=scene_voxel_size)

        # 根据 groundtruth 位姿将当前点云变换到统一世界坐标系
        transform = pose_to_transform(tx, ty, tz, qx, qy, qz, qw)
        frame_pcd.transform(transform)
        fused_scene_pcd += frame_pcd

        # YOLO实例分割并提取当前帧中的目标点云
        result = model(rgb, conf=0.5, retina_masks=True, verbose=False)[0]
        current_target_points = 0
        frame_has_target = False

        if result.masks is not None:
            for object_index, box in enumerate(result.boxes):
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                if class_name != target_class:
                    continue

                mask = result.masks.data[object_index].cpu().numpy() > 0.5
                object_pcd = mask_to_pointcloud(rgb, depth, mask, fx, fy, cx, cy, depth_scale)
                if object_pcd is None:
                    continue

                clean_pcd = clean_object_pointcloud(
                    object_pcd, object_voxel_size, outlier_nb_neighbors,
                    outlier_std_ratio, dbscan_eps, dbscan_min_points
                )
                if len(clean_pcd.points) == 0:
                    continue

                # 将目标点云变换到与融合场景相同的世界坐标系
                clean_pcd.transform(transform)
                fused_object_pcd += clean_pcd
                current_target_points += len(clean_pcd.points)
                target_instance_count += 1
                frame_has_target = True

        if frame_has_target:
            target_frame_count += 1
        fused_count += 1

        depth_time_diff = abs(depth_time - rgb_time)
        print(
            f"已融合第 {frame_index:04d} 帧 | 点数：{len(frame_pcd.points)} | "
            f"目标点数：{current_target_points} | "
            f"Depth时间差：{depth_time_diff:.6f} s | 位姿时间差：{pose_time_diff:.6f} s"
        )

    if fused_count == 0:
        raise RuntimeError("没有可用于融合的 RGB-D 帧，请检查帧范围和时间戳阈值")

    # 所有帧进入世界坐标系后统一降采样，减少重叠区域的重复点
    print(f"\n融合帧数：{fused_count}")
    print(f"最终降采样前点数：{len(fused_scene_pcd.points)}")
    fused_scene_pcd = fused_scene_pcd.voxel_down_sample(voxel_size=fusion_voxel_size)
    print(f"最终降采样后点数：{len(fused_scene_pcd.points)}")

    if len(fused_object_pcd.points) == 0:
        raise RuntimeError(f"处理的帧中没有检测到可用的 {target_class} 点云")

    # 对融合后的目标再次降采样，减少多帧重叠产生的重复点
    print(f"检测到目标的帧数：{target_frame_count}")
    print(f"融合目标实例数：{target_instance_count}")
    print(f"目标最终降采样前点数：{len(fused_object_pcd.points)}")
    fused_object_pcd = fused_object_pcd.voxel_down_sample(voxel_size=object_voxel_size)
    print(f"目标最终降采样后点数：{len(fused_object_pcd.points)}")

    # 在世界坐标系中计算融合目标的三维中心、AABB和OBB
    center, aabb, obb = compute_object_geometry(fused_object_pcd)
    print(f"融合目标三维中心：{center}")
    print(f"融合目标OBB尺寸：{obb.extent}")

    # point文件夹只保存融合后的目标点云
    file_stem = f"fused_{target_class}_{fused_count}_frames"
    point_path = output_point_dir / f"{file_stem}.ply"
    o3d.io.write_point_cloud(str(point_path), fused_object_pcd)
    print(f"目标点云已保存：{point_path}")

    # 显示融合场景、红色目标点云和三维包围盒，并保存截图
    screenshot_path = output_3d_dir / f"{file_stem}.png"
    show_3d_result(fused_scene_pcd, fused_object_pcd, aabb, obb, screenshot_path)


if __name__ == "__main__":
    main()

from pathlib import Path
from time import perf_counter

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


# 实验路径和固定参数与 main.py 保持一致
project_root = Path(__file__).resolve().parent
dataset_dir = project_root / "data" / "rgbd_dataset_freiburg1_xyz"
model_path = project_root / "yolo26n-seg.pt"
target_class = "keyboard"
start_frame = 0
pose_time_threshold = 0.02

# Freiburg1 相机内参和深度参数
image_width = 640
image_height = 480
fx = 517.3
fy = 516.5
cx = 318.6
cy = 255.3
depth_scale = 5000.0
depth_trunc = 5.0

# 统计滤波与 DBSCAN 参数保持不变
outlier_nb_neighbors = 20
outlier_std_ratio = 2.0
dbscan_eps = 0.03
dbscan_min_points = 10

# 只比较帧数和体素大小，不增加过多实验组合
experiment_configs = [
    (10, 2, 0.01),
    (20, 2, 0.01),
    (50, 2, 0.01),
    (50, 2, 0.005),
    (50, 2, 0.02),
]


def run_experiment(model, rgb_list, depth_list, groundtruth_list, num_frames,
                   frame_step, voxel_size):
    """运行一组多帧融合实验并返回点数和核心处理时间。"""
    intrinsic = o3d.camera.PinholeCameraIntrinsic(image_width, image_height, fx, fy, cx, cy)
    fused_scene_pcd = o3d.geometry.PointCloud()
    fused_object_pcd = o3d.geometry.PointCloud()
    object_voxel_size = voxel_size / 2
    end_frame = min(start_frame + num_frames * frame_step, len(rgb_list))
    fused_count = 0
    start_time = perf_counter()

    for frame_index in range(start_frame, end_frame, frame_step):
        rgb_time, rgb_path = rgb_list[frame_index]
        _, depth_path = min(depth_list, key=lambda item: abs(item[0] - rgb_time))
        pose = min(groundtruth_list, key=lambda item: abs(item[0] - rgb_time))
        pose_time, tx, ty, tz, qx, qy, qz, qw = pose
        if abs(pose_time - rgb_time) > pose_time_threshold:
            continue

        rgb = cv2.imread(str(dataset_dir / rgb_path))
        depth = cv2.imread(str(dataset_dir / depth_path), cv2.IMREAD_UNCHANGED)
        if rgb is None or depth is None:
            raise FileNotFoundError(f"第 {frame_index} 帧 RGB 或 Depth 图像读取失败")

        # 生成场景点云并变换到世界坐标系
        frame_pcd = create_scene_pointcloud(rgb, depth, intrinsic, depth_scale, depth_trunc)
        frame_pcd = frame_pcd.voxel_down_sample(voxel_size=voxel_size)
        transform = pose_to_transform(tx, ty, tz, qx, qy, qz, qw)
        frame_pcd.transform(transform)
        fused_scene_pcd += frame_pcd

        # 分割目标，并使用与正式流程相同的滤波和聚类参数
        result = model(rgb, conf=0.5, retina_masks=True, verbose=False)[0]
        if result.masks is not None:
            for object_index, box in enumerate(result.boxes):
                class_name = model.names[int(box.cls[0])]
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

                clean_pcd.transform(transform)
                fused_object_pcd += clean_pcd

        fused_count += 1
        if fused_count % 10 == 0:
            print(f"  已处理 {fused_count}/{num_frames} 帧")

    if fused_count == 0 or len(fused_object_pcd.points) == 0:
        raise RuntimeError("当前实验没有获得可用的场景或目标点云")

    # 融合后再次降采样，并完成正式流程中的目标几何计算
    fused_scene_pcd = fused_scene_pcd.voxel_down_sample(voxel_size=voxel_size)
    fused_object_pcd = fused_object_pcd.voxel_down_sample(voxel_size=object_voxel_size)
    compute_object_geometry(fused_object_pcd)
    elapsed_time = perf_counter() - start_time

    return {
        "frames": fused_count,
        "frame_step": frame_step,
        "voxel_size": voxel_size,
        "scene_points": len(fused_scene_pcd.points),
        "target_points": len(fused_object_pcd.points),
        "time": elapsed_time,
    }


def main():
    rgb_list = read_file_list(dataset_dir / "rgb.txt")
    depth_list = read_file_list(dataset_dir / "depth.txt")
    groundtruth_list = read_groundtruth(dataset_dir / "groundtruth.txt")
    if not rgb_list or not depth_list or not groundtruth_list:
        raise RuntimeError("RGB、Depth 或 groundtruth 数据为空")

    model = YOLO(str(model_path))

    # 预热一次模型，避免首次推理初始化时间影响第一组结果
    warm_rgb = cv2.imread(str(dataset_dir / rgb_list[start_frame][1]))
    if warm_rgb is None:
        raise FileNotFoundError("模型预热使用的 RGB 图像读取失败")
    model(warm_rgb, conf=0.5, retina_masks=True, verbose=False)

    results = []
    for num_frames, frame_step, voxel_size in experiment_configs:
        print(f"\n开始实验：帧数={num_frames}，Frame Step={frame_step}，Voxel Size={voxel_size}")
        result = run_experiment(
            model, rgb_list, depth_list, groundtruth_list,
            num_frames, frame_step, voxel_size
        )
        results.append(result)
        print(
            f"完成：场景点数={result['scene_points']}，目标点数={result['target_points']}，"
            f"运行时间={result['time']:.2f} s"
        )

    print("\n| 帧数 | Frame Step | Voxel Size | 融合后场景点数 | 目标点数 | 运行时间 |")
    print("| ---: | ---: | ---: | ---: | ---: | ---: |")
    for result in results:
        print(
            f"| {result['frames']} | {result['frame_step']} | {result['voxel_size']:.3f} | "
            f"{result['scene_points']} | {result['target_points']} | {result['time']:.2f} s |"
        )


if __name__ == "__main__":
    main()

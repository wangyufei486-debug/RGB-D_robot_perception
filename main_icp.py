"""RGB-D 多帧三维目标感知：使用相邻帧 ICP 估计位姿并融合。

与原 main.py 独立运行；复用相同的 utils，不读取 Groundtruth。
默认处理第 0、2、...、38 帧（20 帧）。
"""
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
import open3d as o3d

from utils.dataset import read_file_list
from utils.geometry import get_depth_median, pixel_to_camera
from utils.pointcloud import (
    clean_object_pointcloud,
    compute_object_geometry,
    create_scene_pointcloud,
    mask_to_pointcloud,
)
from utils.visualization import save_segmentation_result, show_3d_result

# 和原 main.py 保持一致的路径、相机内参、点云参数
project_root = Path(__file__).resolve().parent
dataset_dir = project_root / "data" / "rgbd_dataset_freiburg1_xyz"
model_path = project_root / "yolo26n-seg.pt"

# ICP 结果单独存放，不覆盖 Groundtruth 主程序的结果
output_dir = project_root / "outputs" / "icp"
output_2d_dir = output_dir / "2D"
output_3d_dir = output_dir / "3D"
output_point_dir = output_dir / "point"

start_frame = 0
num_frames = 50
frame_step = 2
target_class = "keyboard"

image_width = 640
image_height = 480
fx, fy = 517.3, 516.5
cx, cy = 318.6, 255.3

depth_scale = 5000.0
depth_trunc = 5.0
scene_voxel_size = 0.01
fusion_voxel_size = 0.01
object_voxel_size = 0.005
outlier_nb_neighbors = 20
outlier_std_ratio = 2.0
dbscan_eps = 0.03
dbscan_min_points = 10

# ICP 单独使用较粗点云以提高速度，融合点云仍采用原分辨率
icp_voxel_size = 0.02
icp_threshold = 0.05
icp_max_iteration = 50



def estimate_poses():
    """在独立进程中计算全部 ICP 位姿，不加载 YOLO/PyTorch。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    rgb_list = read_file_list(dataset_dir / "rgb.txt")
    depth_list = read_file_list(dataset_dir / "depth.txt")
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        image_width, image_height, fx, fy, cx, cy
    )

    frame_indices = list(range(
        start_frame,
        min(start_frame + num_frames * frame_step, len(rgb_list)),
        frame_step,
    ))
    if not frame_indices or not depth_list:
        raise RuntimeError("RGB-D 数据为空或帧数设置不正确")

    transforms = []
    T_world = np.eye(4)
    previous_icp_pcd = None

    for frame_index in frame_indices:
        rgb_time, rgb_path = rgb_list[frame_index]
        _, depth_path = min(depth_list, key=lambda item: abs(item[0] - rgb_time))
        rgb = cv2.imread(str(dataset_dir / rgb_path))
        depth = cv2.imread(str(dataset_dir / depth_path), cv2.IMREAD_UNCHANGED)
        if rgb is None or depth is None:
            raise FileNotFoundError(f"第 {frame_index} 帧 RGB-D 图像读取失败")

        frame_pcd = create_scene_pointcloud(
            rgb, depth, intrinsic, depth_scale, depth_trunc
        ).voxel_down_sample(voxel_size=scene_voxel_size)
        current_icp_pcd = frame_pcd.voxel_down_sample(voxel_size=icp_voxel_size)
        if len(current_icp_pcd.points) == 0:
            raise RuntimeError(f"第 {frame_index} 帧 ICP 点云为空")

        if previous_icp_pcd is None:
            print(f"[ICP] 第 {frame_index:04d} 帧：初始化参考坐标系", flush=True)
        else:
            result = o3d.pipelines.registration.registration_icp(
                current_icp_pcd,
                previous_icp_pcd,
                icp_threshold,
                np.eye(4),
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                o3d.pipelines.registration.ICPConvergenceCriteria(
                    max_iteration=icp_max_iteration
                ),
            )
            T_world = T_world @ result.transformation
            print(
                f"[ICP] 第 {frame_index:04d} 帧 | "
                f"Fitness: {result.fitness:.4f} | "
                f"RMSE: {result.inlier_rmse:.4f} m",
                flush=True,
            )
            if result.fitness < 0.7:
                print("[ICP] 警告：匹配质量偏低，可能存在位姿漂移", flush=True)

        transforms.append(T_world.copy())
        previous_icp_pcd = current_icp_pcd

    # 原子写入避免上次中断遗留半个有效文件
    tmp_path = output_dir / "icp_poses_tmp.npz"
    np.savez(
        tmp_path,
        frame_indices=np.array(frame_indices, dtype=int),
        transforms=np.stack(transforms, axis=0),
    )
    tmp_path.replace(output_dir / "icp_poses.npz")
    print("[ICP] 位姿估计完成，开始主程序融合", flush=True)

def main():
    # 子进程只计算 ICP；主进程只做 YOLO 和点云融合，隔离 OpenMP 运行库。
    subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--estimate-poses"],
        check=True,
    )
    pose_file = np.load(output_dir / "icp_poses.npz")
    icp_frame_indices = pose_file["frame_indices"]
    icp_transforms = pose_file["transforms"]

    # YOLO 仅在主进程里导入，保证 ICP 子进程不加载 PyTorch。
    from ultralytics import YOLO

    for directory in (output_2d_dir, output_3d_dir, output_point_dir):
        directory.mkdir(parents=True, exist_ok=True)

    rgb_list = read_file_list(dataset_dir / "rgb.txt")
    depth_list = read_file_list(dataset_dir / "depth.txt")
    if not rgb_list or not depth_list:
        raise RuntimeError("RGB 或 Depth 索引为空")
    if not (0 <= start_frame < len(rgb_list)) or num_frames <= 0 or frame_step <= 0:
        raise ValueError("start_frame、num_frames 或 frame_step 不合法")

    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        image_width, image_height, fx, fy, cx, cy
    )
    model = YOLO(str(model_path))

    fused_scene_pcd = o3d.geometry.PointCloud()
    fused_object_pcd = o3d.geometry.PointCloud()

    # ICP 累计位姿已由独立子进程算好，不在当前进程里再次运行 ICP。
    fused_count = 0
    target_frame_count = 0
    target_instance_count = 0

    end_frame = min(start_frame + num_frames * frame_step, len(rgb_list))

    expected_indices = np.array(list(range(start_frame, end_frame, frame_step)))
    if not np.array_equal(icp_frame_indices, expected_indices):
        raise RuntimeError("ICP 位姿帧序号与主程序不一致")

    for pose_idx, frame_index in enumerate(expected_indices):
        T_world = icp_transforms[pose_idx]
        rgb_time, rgb_path = rgb_list[frame_index]
        depth_time, depth_path = min(depth_list, key=lambda item: abs(item[0] - rgb_time))

        rgb = cv2.imread(str(dataset_dir / rgb_path))
        depth = cv2.imread(str(dataset_dir / depth_path), cv2.IMREAD_UNCHANGED)
        if rgb is None or depth is None:
            raise FileNotFoundError(f"第 {frame_index} 帧 RGB 或 Depth 读取失败")

        # 仍复用原项目的场景点云生成和降采样
        frame_pcd = create_scene_pointcloud(rgb, depth, intrinsic, depth_scale, depth_trunc)
        frame_pcd = frame_pcd.voxel_down_sample(voxel_size=scene_voxel_size)
        if len(frame_pcd.points) == 0:
            raise RuntimeError(f"第 {frame_index} 帧场景点云为空")

        # 使用独立进程计算的 ICP 累计位姿，而非 Groundtruth。
        # T_world 表示：当前相机坐标系 -> 第一个处理帧坐标系。

        # 将场景点云变换到第一个处理帧的坐标系，并融合
        frame_pcd.transform(T_world)
        fused_scene_pcd += frame_pcd

        # 原来 main.py 的 YOLO + Mask -> 目标点云流程保持不变
        result = model(rgb, conf=0.5, retina_masks=True, verbose=False)[0]
        save_segmentation_result(result.plot(), output_2d_dir, frame_index)
        current_target_points = 0
        frame_has_target = False

        if result.masks is not None:
            for object_index, box in enumerate(result.boxes):
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                if class_name != target_class:
                    continue

                confidence = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                u, v = int((x1 + x2) / 2), int((y1 + y2) / 2)
                center_depth = get_depth_median(
                    depth, u, v, window_size=7, depth_scale=depth_scale
                )
                if center_depth is not None:
                    center_camera = pixel_to_camera(u, v, center_depth, fx, fy, cx, cy)
                    print(
                        f"第 {frame_index:04d} 帧目标：{class_name} | "
                        f"置信度：{confidence:.2f} | "
                        f"相机坐标：({center_camera[0]:.3f}, "
                        f"{center_camera[1]:.3f}, {center_camera[2]:.3f}) m"
                    )

                mask = result.masks.data[object_index].cpu().numpy() > 0.5
                object_pcd = mask_to_pointcloud(
                    rgb, depth, mask, fx, fy, cx, cy, depth_scale
                )
                if object_pcd is None:
                    continue
                clean_pcd = clean_object_pointcloud(
                    object_pcd, object_voxel_size, outlier_nb_neighbors,
                    outlier_std_ratio, dbscan_eps, dbscan_min_points
                )
                if len(clean_pcd.points) == 0:
                    continue

                # 与场景点云使用同一个 ICP 累计变换
                clean_pcd.transform(T_world)
                fused_object_pcd += clean_pcd
                current_target_points += len(clean_pcd.points)
                target_instance_count += 1
                frame_has_target = True

        if frame_has_target:
            target_frame_count += 1
        fused_count += 1
        print(
            f"已融合第 {frame_index:04d} 帧 | 场景点数：{len(frame_pcd.points)} | "
            f"目标点数：{current_target_points} | "
            f"Depth时间差：{abs(depth_time - rgb_time):.6f} s"
        )

    if fused_count == 0:
        raise RuntimeError("没有成功融合的帧")
    if len(fused_object_pcd.points) == 0:
        raise RuntimeError(f"没有检测到可用的 {target_class} 目标点云")

    fused_scene_pcd = fused_scene_pcd.voxel_down_sample(voxel_size=fusion_voxel_size)
    fused_object_pcd = fused_object_pcd.voxel_down_sample(voxel_size=object_voxel_size)
    print(f"\nICP 融合帧数：{fused_count} | 场景点数：{len(fused_scene_pcd.points)}")
    print(f"检测到目标的帧数：{target_frame_count} | 目标实例数：{target_instance_count}")
    print(f"融合目标点数：{len(fused_object_pcd.points)}")

    center, aabb, obb = compute_object_geometry(fused_object_pcd)
    print(f"融合目标三维中心：({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}) m")
    print(f"融合目标 OBB 尺寸：{obb.extent}")

    file_stem = f"fused_icp_{target_class}_{fused_count}_frames"
    point_path = output_point_dir / f"{file_stem}.ply"
    if not o3d.io.write_point_cloud(str(point_path), fused_object_pcd):
        raise RuntimeError(f"目标点云保存失败：{point_path}")
    print(f"目标点云已保存：{point_path}")

    screenshot_path = output_3d_dir / f"{file_stem}.png"
    show_3d_result(fused_scene_pcd, fused_object_pcd, aabb, obb, screenshot_path)


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--estimate-poses":
        estimate_poses()
    else:
        main()

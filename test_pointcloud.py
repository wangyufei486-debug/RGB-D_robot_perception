import cv2
import numpy as np
import open3d as o3d
from pathlib import Path

# Freiburg1 相机内参
fx = 517.3
fy = 516.5
cx = 318.6
cy = 255.3
# 相机内参
intrinsic = o3d.camera.PinholeCameraIntrinsic(
    640,
    480,
    fx,
    fy,
    cx,
    cy
)

#1. 读取 RGB / Depth 和自动匹配时间戳
dataset_dir = Path("data/rgbd_dataset_freiburg1_xyz")

rgb_txt = dataset_dir / "rgb.txt"
depth_txt = dataset_dir / "depth.txt"

def read_file_list(txt_path):
    data = []

    with open(txt_path, "r") as f:
        for line in f:

            # 去掉首尾空格
            line = line.strip()

            # 忽略空行和注释
            if not line or line.startswith("#"):
                continue

            timestamp, file_path = line.split()

            data.append(
                (float(timestamp), file_path)
            )

    return data

rgb_list = read_file_list(rgb_txt)
depth_list = read_file_list(depth_txt)

rgb_time, rgb_path = rgb_list[0]

depth_time, depth_path = min(
    depth_list,
    key=lambda x: abs(x[0] - rgb_time)
)

rgb = cv2.imread(
    str(dataset_dir / rgb_path)
)

depth = cv2.imread(
    str(dataset_dir / depth_path),
    cv2.IMREAD_UNCHANGED
)

#2.把 OpenCV 图片转成 Open3D 图片
# OpenCV：BGR → RGB
rgb_for_o3d = cv2.cvtColor(
    rgb,
    cv2.COLOR_BGR2RGB
)
# numpy → Open3D Image
color_o3d = o3d.geometry.Image(
    np.ascontiguousarray(rgb_for_o3d)
)

depth_o3d = o3d.geometry.Image(
    np.ascontiguousarray(depth)
)

#3.合成 RGB-D 图像
rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
    color_o3d,
    depth_o3d,

    depth_scale=5000.0,

    depth_trunc=5.0,

    convert_rgb_to_intensity=False
)

#5. 生成点云 RGB-D → Point Cloud
pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
    rgbd_image,
    intrinsic
)

print("降采样前点数：", len(pcd.points))

# 体素降采样
pcd_down = pcd.voxel_down_sample(
    voxel_size=0.01
)

print("降采样后点数：", len(pcd_down.points))

#6. 显示点云
#加入坐标轴
coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=0.5,
    origin=[0, 0, 0]
)

o3d.visualization.draw_geometries(
    [pcd_down, coordinate_frame],
    window_name="Downsampled Point Cloud"
)
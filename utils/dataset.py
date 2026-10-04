from pathlib import Path

import cv2


def read_file_list(txt_path):
    """读取 TUM 数据集的时间戳与文件路径列表。"""
    data = []
    with open(txt_path, "r", encoding="utf-8") as file:
        for line in file:
            # TUM 索引文件中的空行和井号开头内容不是图像记录
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            timestamp, file_path = line.split()
            data.append((float(timestamp), file_path))
    return data


def load_rgbd_frame(dataset_dir, frame_index):
    """读取指定序号的 RGB 图像及时间戳最接近的深度图。"""
    dataset_dir = Path(dataset_dir)

    # 分别读取 RGB 和 Depth 的时间戳索引
    rgb_list = read_file_list(dataset_dir / "rgb.txt")
    depth_list = read_file_list(dataset_dir / "depth.txt")

    if frame_index >= len(rgb_list):
        raise ValueError(f"frame_index={frame_index} 超出范围，当前RGB图片总数为 {len(rgb_list)}")

    # 以指定 RGB 帧的时间戳匹配时间上最接近的 Depth 帧
    rgb_time, rgb_path = rgb_list[frame_index]
    _, depth_path = min(depth_list, key=lambda item: abs(item[0] - rgb_time))

    # 彩色图按 BGR 读取，深度图保留原始 uint16 数值
    rgb = cv2.imread(str(dataset_dir / rgb_path))
    depth = cv2.imread(str(dataset_dir / depth_path), cv2.IMREAD_UNCHANGED)

    if rgb is None or depth is None:
        raise FileNotFoundError("RGB 或 Depth 图像读取失败，请检查数据集路径")
    return rgb, depth

def read_groundtruth(txt_path):
    """读取 TUM 相机位姿：timestamp tx ty tz qx qy qz qw。"""
    poses = []

    with open(txt_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            values = [float(value) for value in line.split()]
            timestamp = values[0]
            tx, ty, tz = values[1:4]
            qx, qy, qz, qw = values[4:8]
            poses.append((timestamp, tx, ty, tz, qx, qy, qz, qw))

    return poses
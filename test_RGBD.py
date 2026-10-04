import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path

model = YOLO("yolo26n.pt")

# Freiburg1 相机内参
fx = 517.3
fy = 516.5
cx = 318.6
cy = 255.3

def pixel_to_camera(u, v, Z, fx, fy, cx, cy):

    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy

    return X, Y, Z

def get_depth_median(depth, u, v, window_size=7):
    """
    读取目标中心附近区域的有效深度，并取中位数

    depth       : 深度图
    u, v        : 目标中心像素坐标
    window_size : 取多大的邻域，例如7×7
    """

    half = window_size // 2

    # 防止区域超出图像边界
    x1 = max(0, u - half)
    x2 = min(depth.shape[1], u + half + 1)

    y1 = max(0, v - half)
    y2 = min(depth.shape[0], v + half + 1)

    # 取中心附近的小区域
    depth_region = depth[y1:y2, x1:x2]

    # 去掉无效深度0
    valid_depths = depth_region[depth_region > 0]

    # 如果整个区域都没有有效深度
    if len(valid_depths) == 0:
        return None

    # 取中位数
    depth_value = np.median(valid_depths)

    # TUM：depth / 5000 = 米
    Z = depth_value / 5000.0

    return Z

# ==========================================
# 1. 数据集根目录
# ==========================================
dataset_dir = Path("data/rgbd_dataset_freiburg1_xyz")

rgb_txt = dataset_dir / "rgb.txt"
depth_txt = dataset_dir / "depth.txt"


# ==========================================
# 2. 读取 txt 文件
# ==========================================
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


print("RGB图像数量：", len(rgb_list))
print("Depth图像数量：", len(depth_list))


# ==========================================
# 3. 先取第一张RGB图
# ==========================================
rgb_time, rgb_path = rgb_list[0]


# ==========================================
# 4. 找到时间戳最接近的Depth图
# ==========================================
depth_time, depth_path = min(
    depth_list,
    key=lambda x: abs(x[0] - rgb_time)
)


print("\nRGB时间戳：", rgb_time)
print("RGB路径：", rgb_path)

print("\nDepth时间戳：", depth_time)
print("Depth路径：", depth_path)

print(
    "\n时间差：",
    abs(rgb_time - depth_time),
    "秒"
)


# ==========================================
# 5. 读取RGB和Depth
# ==========================================
rgb = cv2.imread(
    str(dataset_dir / rgb_path)
)

depth = cv2.imread(
    str(dataset_dir / depth_path),
    cv2.IMREAD_UNCHANGED
)

# ==========================================
# 6. YOLO检测RGB图片
# ==========================================
results = model(
    rgb,
    conf=0.5,
    verbose=False
)

result = results[0]

# YOLO画框
annotated_frame = result.plot()

for box in result.boxes:

    cls_id = int(box.cls[0])
    class_name = model.names[cls_id]

    confidence = float(box.conf[0])

    x1, y1, x2, y2 = box.xyxy[0].tolist()

    # 1. 目标中心
    u = int((x1 + x2) / 2)
    v = int((y1 + y2) / 2)
    # 2. 取中位数
    Z = get_depth_median(
        depth,
        u,
        v,
        window_size=7
    )

    if Z is None:
        print(
            f"{class_name:<12}"
            f"pixel=({u},{v})  "
            f"没有有效深度"
        )
        continue
    # 3. 像素 → 三维坐标
    X, Y, Z = pixel_to_camera(
        u,
        v,
        Z,
        fx,
        fy,
        cx,
        cy
    )
    # 4. 画中心点
    cv2.circle(
        annotated_frame,
        (u, v),
        6,
        (0, 0, 255),
        -1
    )
    # 5. 显示XYZ
    text = f"XYZ=({X:.2f},{Y:.2f},{Z:.2f})m"

    cv2.putText(
        annotated_frame,
        text,
        (u + 10, v - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 255),
        2
    )

    print(
        f"{class_name:<12}"
        f"置信度:{confidence:.2f}  "
        f"pixel=({u}, {v})  "
        f"XYZ=({X:.3f}, {Y:.3f}, {Z:.3f}) m"
    )

# ==========================================
# 7. 为了方便人眼观察，对Depth做可视化
# ==========================================
depth_visual = cv2.normalize(
    depth,
    None,
    0,
    255,
    cv2.NORM_MINMAX
)

depth_visual = depth_visual.astype("uint8")

cv2.imshow("YOLO RGB-D", annotated_frame)
cv2.imshow("Depth", depth_visual)

cv2.waitKey(0)
cv2.destroyAllWindows()

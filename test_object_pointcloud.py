import cv2
import numpy as np
import open3d as o3d
from pathlib import Path
from ultralytics import YOLO

model = YOLO("yolo26n-seg.pt")

# ==================================================
# 选择测试第几帧,测试类别
# ==================================================

FRAME_INDEX = 50
TARGET_CLASS = "keyboard"

# ==================================================
# 输出结果保存路径
# ==================================================
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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

#“目标框 → 局部点云”函数
def bbox_to_pointcloud(
        rgb,
        depth,
        x1, y1, x2, y2,
        fx, fy, cx, cy,
        depth_scale=5000.0,
        center_depth=None,
        depth_threshold=0.03):
    """
    根据YOLO二维检测框，提取目标对应的局部三维点云

    rgb              : OpenCV读取的BGR图像
    depth            : uint16深度图
    x1,y1,x2,y2      : YOLO边界框
    fx,fy,cx,cy      : 相机内参
    center_depth     : 目标中心附近估计出的深度，单位m
    depth_threshold  : 允许与中心深度相差多少米
    """

    h, w = depth.shape

    # 1. 转整数，并防止框超出图像边界
    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(w, int(x2))
    y2 = min(h, int(y2))

    # 2. 截取YOLO框内部的RGB和Depth
    depth_roi = depth[y1:y2, x1:x2]
    rgb_roi = rgb[y1:y2, x1:x2]

    # 3. 原始深度 → 米
    z = depth_roi.astype(np.float32) / depth_scale

    # 4. 有效深度
    valid = depth_roi > 0

    # 5. 根据目标中心深度，尽量排除框里的背景
    if center_depth is not None:
        valid = valid & (
            np.abs(z - center_depth) < depth_threshold
        )

    # 如果一个有效点都没有
    if np.count_nonzero(valid) == 0:
        return None

    # 6. 生成每个像素对应的原始图像坐标 u、v
    u, v = np.meshgrid(
        np.arange(x1, x2),
        np.arange(y1, y2)
    )

    # 7. 像素反投影 → 三维坐标
    X = (u - cx) * z / fx
    Y = (v - cy) * z / fy
    Z = z

    # 只保留有效点
    points = np.stack(
        [X[valid], Y[valid], Z[valid]],
        axis=1
    )

    # 8. 取得对应颜色
    # OpenCV：BGR → Open3D需要RGB
    rgb_roi = rgb_roi[:, :, ::-1].copy()

    colors = rgb_roi[valid].astype(
        np.float64
    ) / 255.0

    # 9. 创建Open3D点云
    object_pcd = o3d.geometry.PointCloud()

    object_pcd.points = o3d.utility.Vector3dVector(
        points
    )

    object_pcd.colors = o3d.utility.Vector3dVector(
        colors
    )

    return object_pcd

def mask_to_pointcloud(
        rgb,
        depth,
        mask,
        fx,
        fy,
        cx,
        cy,
        depth_scale=5000.0
):

    # 深度转换成米
    Z = depth.astype(np.float32) / depth_scale

    # 有效条件：
    # 1. 属于目标Mask
    # 2. 深度有效
    valid = mask & (depth > 0)

    if np.count_nonzero(valid) == 0:
        return None

    # 整幅图片生成像素坐标
    h, w = depth.shape

    u, v = np.meshgrid(
        np.arange(w),
        np.arange(h)
    )

    # 像素 → XYZ
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy

    points = np.stack(
        [
            X[valid],
            Y[valid],
            Z[valid]
        ],
        axis=1
    )

    # OpenCV BGR → RGB
    rgb_for_o3d = cv2.cvtColor(
        rgb,
        cv2.COLOR_BGR2RGB
    )

    colors = (
        rgb_for_o3d[valid]
        .astype(np.float64)
        / 255.0
    )

    object_pcd = o3d.geometry.PointCloud()

    object_pcd.points = (
        o3d.utility.Vector3dVector(points)
    )

    object_pcd.colors = (
        o3d.utility.Vector3dVector(colors)
    )

    return object_pcd

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

if FRAME_INDEX >= len(rgb_list):
    raise ValueError(
        f"FRAME_INDEX={FRAME_INDEX} 超出范围，"
        f"当前RGB图片总数为 {len(rgb_list)}"
    )

rgb_time, rgb_path = rgb_list[FRAME_INDEX]

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

# 整体点云降采样
# print("降采样前点数：", len(pcd.points))
pcd_down = pcd.voxel_down_sample(voxel_size=0.01)
# print("降采样后点数：", len(pcd_down.points))

results = model(
    rgb,
    conf=0.5,
    retina_masks=True,
    verbose=False
)

result = results[0]

# print("\n========== Segmentation 检查 ==========")
#
# print("检测到的目标数量：", len(result.boxes))
#
# if result.masks is None:
#     print("没有生成任何 Mask")
# else:
#     print("成功生成 Mask")
#     print("Mask 数量：", len(result.masks.data))
#     print("Mask shape：", result.masks.data.shape)

# YOLO画框
annotated_frame = result.plot()

# ==================================================
# 保存二维分割结果
# ==================================================
save_path = (
    OUTPUT_DIR
    / f"frame_{FRAME_INDEX:04d}_seg.png"
)

success = cv2.imwrite(
    str(save_path),
    annotated_frame
)

if success:
    print(f"二维分割结果已保存：{save_path}")
else:
    print("二维分割结果保存失败")

cv2.imshow(
    "YOLO Segmentation Result",
    annotated_frame
)

for i, box in enumerate(result.boxes):
    # 获取类别、置信度、Bounding Box
    cls_id = int(box.cls[0])
    class_name = model.names[cls_id]
    confidence = float(box.conf[0])
    x1, y1, x2, y2 = box.xyxy[0].tolist()

    # 1. Bounding Box中心像素
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
    # 3. 像素 → 三维坐标，计算中心像素对应的XYZ
    X, Y, Z = pixel_to_camera(
        u,
        v,
        Z,
        fx,
        fy,
        cx,
        cy
    )
    # 4.
    if class_name == TARGET_CLASS:

        print("\n""========== RGB-D 三维目标感知结果 ==========")
        print(f"\n目标：{class_name}")
        print(f"置信度：{confidence:.2f}")
        # print(f"中心像素：({u}, {v})")
        # print(f"中心XYZ："f"({X:.3f}, {Y:.3f}, {Z:.3f}) m")

        # object_pcd = bbox_to_pointcloud(
        #     rgb,
        #     depth,
        #     x1, y1, x2, y2,
        #     fx, fy, cx, cy,
        #     center_depth=Z,
        #     depth_threshold=0.15
        # )

        mask = (result.masks.data[i].cpu().numpy() > 0.5)
        print("Mask shape：",mask.shape)
        print("Depth shape：",depth.shape)

        # 5. Mask → keyboard三维点云
        object_pcd = mask_to_pointcloud(
            rgb,
            depth,
            mask,
            fx,
            fy,
            cx,
            cy
        )

        if object_pcd is None:
            print("keyboard局部点云提取失败")
            continue

        # 局部点云降采样
        # print(f"\n提取目标： {class_name}")
        # print("目标点云降采样前：",len(object_pcd.points))
        object_pcd_down = object_pcd.voxel_down_sample(voxel_size=0.005)
        # print("目标点云降采样后：",len(object_pcd_down.points))

        # ==================================================
        # 目标点云统计滤波去除离群点
        # ==================================================

        clean_pcd, ind = object_pcd_down.remove_statistical_outlier(
            nb_neighbors=20,
            std_ratio=2.0
        )

        print("去噪后点数：",len(clean_pcd.points))

        # ==================================================
        # DBSCAN聚类，只保留最大的点云簇
        # ==================================================

        labels = np.array(
            clean_pcd.cluster_dbscan(
                eps=0.03,
                min_points=10,
                print_progress=False
            )
        )

        # -1代表噪声
        valid_labels = labels[labels >= 0]

        if len(valid_labels) > 0:
            # 找出现次数最多的类别
            largest_label = np.bincount(
                valid_labels
            ).argmax()

            # 找到最大簇对应的点
            largest_cluster_index = np.where(
                labels == largest_label
            )[0]

            clean_pcd = clean_pcd.select_by_index(
                largest_cluster_index
            )

        print(f"最大聚类保留后点数："
              f"{len(clean_pcd.points)}")

        # ==================================================
        # 计算目标三维中心
        # ==================================================

        center = clean_pcd.get_center()

        print(
            f"\n目标点云三维中心："
            f"({center[0]:.3f}, "
            f"{center[1]:.3f}, "
            f"{center[2]:.3f}) m"
        )

        # ==================================================
        # 创建轴对齐3D包围盒 AABB
        # ==================================================

        aabb = clean_pcd.get_axis_aligned_bounding_box()

        # 设置包围盒颜色
        aabb.color = (0, 1, 0)
        extent = aabb.get_extent()

        width_x = extent[0]
        height_y = extent[1]
        depth_z = extent[2]

        # print(
        #     f"目标三维尺寸：\n"
        #     f"X方向：{width_x:.3f} m\n"
        #     f"Y方向：{height_y:.3f} m\n"
        #     f"Z方向：{depth_z:.3f} m"
        # )

        # ==================================================
        # OBB：旋转包围盒
        # ==================================================

        obb = clean_pcd.get_oriented_bounding_box(robust=True)
        obb.color = (0, 0, 1)
        obb_extent = obb.extent

        print(
            f"OBB估计尺寸："
            f"{obb_extent[0]:.3f} × "
            f"{obb_extent[1]:.3f} × "
            f"{obb_extent[2]:.3f} m"
        )

        # 加入坐标轴
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.5,
            origin=[0, 0, 0]
        )
        # 把keyboard点云染成红色
        clean_pcd.paint_uniform_color(
            [1.0, 0.0, 0.0]
        )

        o3d.visualization.draw_geometries(
            [
                pcd_down,
                clean_pcd,
                aabb,
                obb,
                coordinate_frame
            ],
            window_name=" RGB-D 3D Perception"
        )


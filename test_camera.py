import cv2
import time
from ultralytics import YOLO

def pixel_to_camera(u, v, depth, fx, fy, cx, cy):
    """
    将像素坐标转换成相机坐标系下的三维坐标

    u, v   : 像素坐标
    depth  : 深度，也就是Z
    fx, fy : 焦距参数
    cx, cy : 主点坐标
    """

    X = (u - cx) * depth / fx
    Y = (v - cy) * depth / fy
    Z = depth

    return X, Y, Z

fx = 600
fy = 600
cx = 320
cy = 240
depth = 0.8

# 1. 加载预训练 YOLO 模型
model = YOLO("yolo26n.pt")

# 2. 打开电脑默认摄像头
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("无法打开摄像头")
    exit()

# 上一次打印的时间
last_print_time = 0

# 每隔多少秒打印一次
print_interval = 1.0

while True:
    # 3. 从摄像头读取一帧图片
    ret, frame = cap.read()

    if not ret:
        print("无法读取画面")
        break

    # 4. 把当前图片送入 YOLO
    results = model(frame, conf=0.5,verbose=False)

    result = results[0]
    # 5. YOLO先画检测框
    annotated_frame = results[0].plot()

    # 当前时间
    current_time = time.time()

    detections = []

    # ==================================================
    # 每一帧都执行：画中心点 + 显示坐标
    # ==================================================
    for box in result.boxes:
        cls_id = int(box.cls[0])
        class_name = model.names[cls_id]
        confidence = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        # 目标中心坐标
        u = int((x1 + x2) / 2)
        v = int((y1 + y2) / 2)

        # 保存这一帧的检测结果
        detections.append(
            (class_name, confidence, u, v)
        )

        # ---------- 画红色中心点 ----------
        cv2.circle(
            annotated_frame,
            (u, v),
            7,
            (0, 0, 255),
            -1
        )

        # ---------- 在中心点旁边显示类别 + 坐标 ----------
        text = f"{class_name} ({u}, {v})"

        cv2.putText(
            annotated_frame,
            text,
            (u + 10, v - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2
        )
    # ==================================================
    # 只有距离上一次输出超过 1 秒，才打印
    # ==================================================
    if current_time - last_print_time >= print_interval:
        print("\n---------- 当前检测结果 ----------")
        for class_name, confidence, u, v in detections:

            X, Y, Z = pixel_to_camera(
                u,
                v,
                depth,
                fx,
                fy,
                cx,
                cy
            )

            print(
                f"{class_name:<12}"
                f"置信度:{confidence:.2f}  "
                f"pixel=({u}, {v})  "
                f"XYZ=({X:.3f}, {Y:.3f}, {Z:.3f}) m"
            )
        # 注意：一定放到for循环结束以后
        last_print_time = current_time

    # 6. 显示结果
    cv2.imshow("Robot Perception", annotated_frame)

    # 按 q 退出
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
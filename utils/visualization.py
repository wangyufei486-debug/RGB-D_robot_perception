from pathlib import Path

import cv2
import open3d as o3d


def save_segmentation_result(annotated_frame, output_dir, frame_index):
    """将二维实例分割结果保存到输出目录。"""
    # 输出目录不存在时自动创建，文件名中保留四位帧序号
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / f"frame_{frame_index:04d}_seg.png"
    if cv2.imwrite(str(save_path), annotated_frame):
        print(f"二维分割结果已保存：{save_path}")
    else:
        print("二维分割结果保存失败")


def show_3d_result(scene_pcd, object_pcd, aabb, obb, screenshot_path):
    """显示三维结果，自动保存默认视角，按 S 可保存调整后的视角。"""
    screenshot_path = Path(screenshot_path)
    screenshot_path.parent.mkdir(parents=True, exist_ok=True)

    # 坐标轴用于观察点云相对于相机坐标系的位置和方向
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.5, origin=[0, 0, 0]
    )

    # 将目标统一染成红色，使其与原始场景点云明显区分
    object_pcd.paint_uniform_color([1.0, 0.0, 0.0])

    # 使用可注册按键事件的窗口，以便保存用户调整后的观察视角
    visualizer = o3d.visualization.VisualizerWithKeyCallback()
    visualizer.create_window(window_name="RGB-D 3D Perception", width=1280, height=720)
    for geometry in (scene_pcd, object_pcd, aabb, obb, coordinate_frame):
        visualizer.add_geometry(geometry)

    def save_screenshot(vis):
        vis.capture_screen_image(str(screenshot_path), do_render=True)
        print(f"三维结果截图已保存：{screenshot_path}")
        return False

    visualizer.register_key_callback(ord("S"), save_screenshot)

    # 先保存自动适配场景后的默认视角，按 S 可以用调整后的视角覆盖
    visualizer.poll_events()
    visualizer.update_renderer()
    visualizer.capture_screen_image(str(screenshot_path), do_render=True)
    print(f"默认三维截图已保存：{screenshot_path}")
    print("可在 Open3D 窗口中调整视角，按 S 重新保存截图，关闭窗口结束显示")

    visualizer.run()
    visualizer.destroy_window()

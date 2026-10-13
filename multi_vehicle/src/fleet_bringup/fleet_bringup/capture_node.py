import sys
import threading
import time
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import Point, PoseStamped
from rclpy.node import Node
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray


def _register_dir():
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / 'aerial_register' / 'demo.py').is_file():
            return parent / 'aerial_register'
    return Path('/home/jsetc/multi_vehicle/aerial_register')


def _cell_seed(x, y, attempt):
    return int(abs(round(x * 10) * 100 + round(y * 10))) + int(attempt) * 17 + 11


def facade_wall(local, labels):
    """用高出地面的立面点拟合一堵墙：中心、朝向、长度、厚度和高度。"""
    facade = np.asarray(labels) == 'facade'
    if not np.any(facade):
        return None
    tall = np.asarray(local)[facade]
    tall = tall[tall[:, 2] > 0.35]
    if len(tall) < 8:
        return None
    center = np.median(tall[:, :2], axis=0)
    centered = tall[:, :2] - center
    _u, _s, axes = np.linalg.svd(centered, full_matrices=False)
    axis = axes[0]
    yaw = float(np.arctan2(axis[1], axis[0]))
    along = centered @ axis
    across = centered @ np.array([-axis[1], axis[0]])
    length = float(np.percentile(along, 95) - np.percentile(along, 5))
    length = float(np.clip(length + 0.08, 0.28, 1.2))
    thickness = float(np.percentile(np.abs(across), 95) * 2.0 + 0.16)
    thickness = float(np.clip(thickness, 0.22, 0.45))
    height = float(np.clip(np.percentile(tall[:, 2], 90), 0.45, 1.8))
    return (
        float(center[0]), float(center[1]), yaw, length, thickness, height,
    )


class CaptureNode(Node):
    def __init__(self):
        super().__init__('capture')
        register_dir = _register_dir()
        if str(register_dir) not in sys.path:
            sys.path.insert(0, str(register_dir))
        self.pending = []
        self.surveys = []
        self.reports = []
        self.lock = threading.Lock()
        self.pub = self.create_publisher(MarkerArray, '/fleet/survey', 10)
        self.report_pub = self.create_publisher(Marker, '/fleet/survey_report', 10)
        self.create_subscription(PoseStamped, '/fleet/capture', self.on_capture, 10)
        self.create_timer(0.5, self.publish_surveys)
        self.worker = threading.Thread(target=self.work, daemon=True)
        self.worker.start()
        self.get_logger().info('航拍节点已就绪，平地和立面分开配准')

    def on_capture(self, msg):
        item = (
            float(msg.pose.position.x),
            float(msg.pose.position.y),
            float(msg.pose.position.z) > 0.5,
            int(round(msg.pose.orientation.w)),
        )
        with self.lock:
            self.pending.append(item)
        kind = '立面' if item[2] else '平地'
        self.get_logger().info(
            f'收到{kind}格 ({item[0]:.1f}, {item[1]:.1f})，第 {item[3] + 1} 次配准')

    def _queue_report(self, cell_x, cell_y, code, wall=None):
        with self.lock:
            self.reports.append((cell_x, cell_y, float(code), wall))

    def work(self):
        from demo import registration_ok, survey_cloud
        while rclpy.ok():
            with self.lock:
                item = self.pending.pop(0) if self.pending else None
            if item is None:
                time.sleep(0.2)
                continue
            cell_x, cell_y, aerial, attempt = item
            seeds = [_cell_seed(cell_x, cell_y, attempt)]
            seeds.extend(seeds[0] + offset for offset in (31, 62, 93))
            result = None
            for seed in seeds:
                result = survey_cloud(
                    seed=seed,
                    with_facade=aerial,
                    noise=0.4,
                    vary_scene=True,
                )
                if registration_ok(result, aerial):
                    break
            if not registration_ok(result, aerial):
                self.get_logger().error(
                    f'({cell_x:.1f}, {cell_y:.1f}) 配准失败：控制点或平面误差过大')
                self._queue_report(cell_x, cell_y, -1.0)
                continue
            local = self.to_map(result['points'], result['labels'], cell_x, cell_y)
            wall = facade_wall(local, result['labels']) if aerial else None
            with self.lock:
                self.surveys.append((cell_x, cell_y, local, result['labels'], wall is not None))
            self._queue_report(cell_x, cell_y, 1.0 if wall else 0.0, wall)
            if result['posed']:
                detail = (
                    f'地面偏差 {result["ground_error"]:.2f}，'
                    f'立面偏差 {result["facade_error"]:.2f}，'
                    f'旋转 {result["rotation_error"]:.2f}°，'
                    f'平移 {result["translation_error"]:.2f}°'
                    + ('，用了第二张角度' if result.get('second_view') else '')
                )
            else:
                detail = f'地面控制点 {result["assigned"]["ground"][1]:.2f} px'
            self.get_logger().info(
                f'({cell_x:.1f}, {cell_y:.1f}) 配准完成：'
                f'{len(result["points"])} 个点，{detail}'
                + (
                    f'，墙心 ({wall[0]:.2f}, {wall[1]:.2f})，长 {wall[3]:.2f} m'
                    if wall else '，平地'
                )
            )

    def to_map(self, points, labels, cell_x, cell_y):
        ground = labels == 'ground'
        origin_x = float(np.median(points[:, 0]))
        origin_y = float(np.median(points[:, 1]))
        origin_z = float(np.median(points[ground, 2])) if ground.any() else float(np.median(points[:, 2]))
        local = np.column_stack([
            points[:, 0] - origin_x,
            points[:, 1] - origin_y,
            points[:, 2] - origin_z,
        ])
        flat = np.linalg.norm(local[:, :2], axis=1)
        span = float(np.percentile(flat, 90)) if len(flat) else 1.0
        local *= 0.75 / max(span, 1e-3)
        local[:, 2] -= np.min(local[:, 2])
        local[:, 2] += 0.05
        local[:, 0] += cell_x
        local[:, 1] += cell_y
        return local

    def publish_surveys(self):
        with self.lock:
            surveys = list(self.surveys)
            reports = self.reports
            self.reports = []
        for cell_x, cell_y, code, wall in reports:
            report = Marker()
            report.header.stamp = self.get_clock().now().to_msg()
            report.header.frame_id = 'map'
            report.ns = 'report'
            report.type = Marker.CUBE
            report.action = Marker.ADD
            report.pose.position.x = cell_x
            report.pose.position.y = cell_y
            report.pose.position.z = code
            report.pose.orientation.w = 1.0
            if wall is not None:
                center_x, center_y, yaw, length, thickness, height = wall
                report.scale.x = length
                report.scale.y = thickness
                report.scale.z = height
                report.text = f'{center_x:.4f},{center_y:.4f},{yaw:.4f}'
            self.report_pub.publish(report)
        now = self.get_clock().now().to_msg()
        markers = []
        paints = {
            'ground': ColorRGBA(r=0.25, g=0.9, b=0.3, a=1.0),
            'facade': ColorRGBA(r=1.0, g=0.5, b=0.1, a=1.0),
            'outlier': ColorRGBA(r=0.9, g=0.2, b=0.2, a=1.0),
        }
        for index, (cell_x, cell_y, points, labels, obstacle) in enumerate(surveys):
            cloud = Marker()
            cloud.header.stamp = now
            cloud.header.frame_id = 'map'
            cloud.ns = 'survey'
            cloud.id = index
            cloud.type = Marker.POINTS
            cloud.action = Marker.ADD
            cloud.scale.x = 0.07
            cloud.scale.y = 0.07
            cloud.points = [
                Point(x=float(x), y=float(y), z=float(z))
                for x, y, z in points
            ]
            cloud.colors = [paints.get(label, paints['outlier']) for label in labels]
            markers.append(cloud)
            text = Marker()
            text.header.stamp = now
            text.header.frame_id = 'map'
            text.ns = 'survey_label'
            text.id = index
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.pose.position.x = cell_x
            text.pose.position.y = cell_y
            text.pose.position.z = 1.15
            text.pose.orientation.w = 1.0
            text.scale.z = 0.18
            text.color = ColorRGBA(r=1.0, g=0.85, b=0.4, a=1.0)
            text.text = '发现立面' if obstacle else '已配准'
            markers.append(text)
        array = MarkerArray()
        array.markers = markers
        if markers:
            self.pub.publish(array)


def main():
    rclpy.init()
    node = CaptureNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

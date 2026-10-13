import math
import threading
import time

import rclpy
from rclpy.action import ActionServer
from rclpy.action.server import CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from visualization_msgs.msg import Marker, MarkerArray

from fleet_bringup.world import KNOWN_OBSTACLES
from fleet_interfaces.action import Mission
from fleet_interfaces.msg import VehicleState


def wrap_angle(angle):
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


class VehicleNode(Node):
    def __init__(self):
        super().__init__('vehicle')
        self.declare_parameter('vehicle_id', 'tb1')
        self.declare_parameter('kind', 'robot')
        self.declare_parameter('start_x', 0.0)
        self.declare_parameter('start_y', 0.0)
        self.declare_parameter('start_yaw', 0.0)
        self.declare_parameter('home_x', 0.0)
        self.declare_parameter('home_y', 0.0)
        self.declare_parameter('peer_topics', [''])

        self.vehicle_id = str(self.get_parameter('vehicle_id').value)
        self.kind = str(self.get_parameter('kind').value)
        self.x = float(self.get_parameter('start_x').value)
        self.y = float(self.get_parameter('start_y').value)
        self.yaw = float(self.get_parameter('start_yaw').value)
        self.home_x = float(self.get_parameter('home_x').value)
        self.home_y = float(self.get_parameter('home_y').value)
        self.idle = True
        self.battery = 55.0
        self.progress = 0.0
        self.phase = 'idle'
        self.peers = {}
        self.obstacles = list(KNOWN_OBSTACLES)
        self.walls = []
        self.lock = threading.Lock()
        self.max_speed = 0.85 if self.kind == 'uav' else 0.35

        group = ReentrantCallbackGroup()
        self.state_pub = self.create_publisher(VehicleState, 'state', 10)
        self.create_timer(0.1, self.publish_state, callback_group=group)
        for topic in self.get_parameter('peer_topics').value:
            if not topic:
                continue
            self.create_subscription(
                VehicleState, topic, self.on_peer, 10, callback_group=group)
        if self.kind == 'robot':
            self.create_subscription(
                MarkerArray, '/fleet/obstacles', self.on_obstacles, 10,
                callback_group=group)
        self._action_server = ActionServer(
            self,
            Mission,
            'mission',
            execute_callback=self.execute_mission,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
            callback_group=group,
        )
        self.get_logger().info(
            f'{self.vehicle_id} ({self.kind}) 已就绪，起点 ({self.x:.1f}, {self.y:.1f})')

    def goal_callback(self, goal_request):
        charging = goal_request.mode == 'charge'
        with self.lock:
            if not self.idle or (self.battery < 15.0 and not charging):
                return GoalResponse.REJECT
            self.idle = False
            self.progress = 0.0
            self.phase = 'charge' if charging else 'survey'
        action = '回基站' if charging else '勘查'
        self.get_logger().info(
            f'接受{action}：({goal_request.target_x:.1f}, {goal_request.target_y:.1f})')
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        return CancelResponse.ACCEPT

    def on_obstacles(self, msg):
        walls = []
        for marker in msg.markers:
            if marker.action != Marker.ADD or marker.type != Marker.CUBE:
                continue
            walls.append((
                float(marker.pose.position.x),
                float(marker.pose.position.y),
                _yaw_from_quaternion(marker.pose.orientation),
                float(marker.scale.x),
                float(marker.scale.y),
            ))
        with self.lock:
            self.walls = walls

    def on_peer(self, msg):
        if msg.vehicle_id == self.vehicle_id:
            return
        with self.lock:
            self.peers[msg.vehicle_id] = (msg.x, msg.y, time.monotonic())

    def execute_mission(self, goal_handle):
        target_x = float(goal_handle.request.target_x)
        target_y = float(goal_handle.request.target_y)
        charging = goal_handle.request.mode == 'charge'
        with self.lock:
            initial = max(math.hypot(target_x - self.x, target_y - self.y), 0.2)
            best_dist = initial
        last_improve = time.monotonic()
        result = Mission.Result()
        feedback = Mission.Feedback()

        while rclpy.ok():
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                result.success = False
                result.message = '已取消'
                self._finish()
                return result

            with self.lock:
                linear, angular, dist = self._command(target_x, target_y)
                self._integrate(linear, angular, 0.1)
                self.progress = max(0.0, min(1.0, 1.0 - dist / initial))
                feedback.progress = float(self.progress)
                feedback.x = self.x
                feedback.y = self.y
                battery = self.battery
                if linear > 0.01:
                    self.battery = max(0.0, self.battery - 0.10)
                arrived = dist < 0.18

            goal_handle.publish_feedback(feedback)
            if arrived:
                if charging:
                    charged = self._charge_until_full(goal_handle, feedback)
                    if not charged:
                        result.success = False
                        result.message = '充电中断'
                        self._finish()
                        return result
                    goal_handle.succeed()
                    result.success = True
                    result.message = '充电完成'
                    self._finish()
                    return result
                with self.lock:
                    self.progress = 1.0
                goal_handle.succeed()
                result.success = True
                result.message = '到达目标'
                self.get_logger().info(
                    f'到达 ({target_x:.1f}, {target_y:.1f})，剩余电量 {battery:.0f}%')
                self._finish()
                return result
            if battery < 8.0 and not charging:
                goal_handle.abort()
                result.success = False
                result.message = '电量不足'
                self._finish()
                return result
            if dist < best_dist - 0.05:
                best_dist = dist
                last_improve = time.monotonic()
            elif time.monotonic() - last_improve > 25.0:
                goal_handle.abort()
                result.success = False
                result.message = '路径受阻'
                self.get_logger().warn('长时间没有靠近目标，放弃本次任务')
                self._finish()
                return result
            time.sleep(0.1)

        result.success = False
        result.message = '节点退出'
        self._finish()
        return result

    def _finish(self):
        with self.lock:
            self.idle = True
            self.phase = 'idle'

    def _charge_until_full(self, goal_handle, feedback):
        while rclpy.ok():
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                return False
            with self.lock:
                self.phase = 'charge'
                self.battery = min(100.0, self.battery + 0.6)
                self.progress = self.battery / 100.0
                feedback.progress = float(self.progress)
                feedback.x = self.x
                feedback.y = self.y
                full = self.battery >= 99.5
            goal_handle.publish_feedback(feedback)
            if full:
                return True
            time.sleep(0.1)
        return False

    def _command(self, target_x, target_y):
        dx = target_x - self.x
        dy = target_y - self.y
        dist = math.hypot(dx, dy)
        error = wrap_angle(math.atan2(dy, dx) - self.yaw)
        angular = max(-1.2, min(1.2, error * 2.2))
        linear = 0.0 if abs(error) > 0.7 else min(self.max_speed, max(dist, 0.0))
        if self.kind == 'robot':
            linear, angular = self._avoid(linear, angular)
        return linear, angular, dist

    def _avoid(self, linear, angular):
        ahead = self._clearance(self.x + math.cos(self.yaw) * 0.55,
                                self.y + math.sin(self.yaw) * 0.55)
        if ahead >= 0.18:
            return linear, angular
        left = self._clearance(self.x + math.cos(self.yaw + 1.0) * 0.55,
                               self.y + math.sin(self.yaw + 1.0) * 0.55)
        right = self._clearance(self.x + math.cos(self.yaw - 1.0) * 0.55,
                                self.y + math.sin(self.yaw - 1.0) * 0.55)
        return 0.0, 0.9 if left >= right else -0.9

    def _clearance(self, x, y):
        clearance = 10.0
        for ox, oy, radius in self.obstacles:
            clearance = min(clearance, math.hypot(x - ox, y - oy) - radius)
        for wall in self.walls:
            clearance = min(clearance, _wall_clearance(x, y, wall))
        now = time.monotonic()
        stale = [name for name, (_, _, stamp) in self.peers.items() if now - stamp > 1.0]
        for name in stale:
            del self.peers[name]
        for px, py, _stamp in self.peers.values():
            clearance = min(clearance, math.hypot(x - px, y - py) - 0.35)
        return clearance

    def _integrate(self, linear, angular, dt):
        self.yaw = wrap_angle(self.yaw + angular * dt)
        self.x += linear * math.cos(self.yaw) * dt
        self.y += linear * math.sin(self.yaw) * dt

    def publish_state(self):
        with self.lock:
            msg = VehicleState()
            msg.vehicle_id = self.vehicle_id
            msg.kind = self.kind
            msg.x = self.x
            msg.y = self.y
            msg.yaw = self.yaw
            msg.idle = self.idle
            msg.battery = float(self.battery)
            msg.progress = float(self.progress)
            msg.phase = self.phase
        self.state_pub.publish(msg)


def _yaw_from_quaternion(rotation):
    return math.atan2(
        2.0 * rotation.w * rotation.z,
        1.0 - 2.0 * rotation.z * rotation.z,
    )


def _wall_clearance(x, y, wall):
    """点到立面墙矩形的有符号距离，墙内为负。"""
    center_x, center_y, yaw, length, thickness = wall
    dx = x - center_x
    dy = y - center_y
    along = dx * math.cos(yaw) + dy * math.sin(yaw)
    across = -dx * math.sin(yaw) + dy * math.cos(yaw)
    outside_x = abs(along) - length * 0.5
    outside_y = abs(across) - thickness * 0.5
    outside = math.hypot(max(outside_x, 0.0), max(outside_y, 0.0))
    inside = min(max(outside_x, outside_y), 0.0)
    return outside + inside


def main():
    rclpy.init()
    node = VehicleNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

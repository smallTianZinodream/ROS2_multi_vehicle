import math
import time

import rclpy
from geometry_msgs.msg import Point, PoseStamped, Quaternion
from rclpy.action import ActionClient
from rclpy.node import Node
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

from fleet_bringup.world import COLORS, HOMES, KNOWN_OBSTACLES, survey_cells
from fleet_interfaces.action import Mission
from fleet_interfaces.msg import VehicleState


class DispatcherNode(Node):
    def __init__(self):
        super().__init__('dispatcher')
        self.declare_parameter('vehicles', ['tb1', 'tb2', 'uav1'])
        self.declare_parameter('min_battery', 40.0)
        names = list(self.get_parameter('vehicles').value)
        self.min_battery = float(self.get_parameter('min_battery').value)

        self.board = []
        for cell in survey_cells():
            cell.update({'attempts': 0, 'status': 'wait', 'owner': ''})
            self.board.append(cell)
        self.total = len(self.board)
        self.aerial_total = sum(1 for cell in self.board if cell['aerial'])
        self.task_pub = self.create_publisher(MarkerArray, '/fleet/tasks', 10)
        self.capture_pub = self.create_publisher(PoseStamped, '/fleet/capture', 10)
        self.obstacle_pub = self.create_publisher(MarkerArray, '/fleet/obstacles', 10)
        self.create_subscription(Marker, '/fleet/survey_report', self.on_report, 10)
        self.finished = 0
        self.dropped = 0
        self.announced = False
        self.states = {}
        self.busy = {}
        self.pending_checks = {}
        self.discovered = []
        self._action_clients = {}
        self.last_feedback = {}

        for name in names:
            self._action_clients[name] = ActionClient(self, Mission, f'/{name}/mission')
            self.create_subscription(
                VehicleState, f'/{name}/state', self._bind_state(name), 10)
        self.create_timer(0.5, self.assign_tasks)
        self.create_timer(1.0, self._publish_obstacles)
        self.get_logger().info(
            f'调度器已启动：地面格 {self.total - self.aerial_total}，'
            f'障碍上空 {self.aerial_total}。按距离、电量和机型分配')

    def _bind_state(self, name):
        def _on_state(msg):
            self.states[name] = msg
        return _on_state

    def assign_tasks(self):
        self._publish_tasks()
        self._expire_checks()
        if (
            self.finished + self.dropped >= self.total
            and not self.busy
            and not self.pending_checks
        ):
            if not self.announced:
                self.announced = True
                self.get_logger().info(
                    f'覆盖完成：勘查 {self.finished}，放弃 {self.dropped}')
            return

        free = []
        for name, state in self.states.items():
            if name in self.busy or not state.idle:
                continue
            if not self._action_clients[name].server_is_ready():
                continue
            free.append(name)

        for name in list(free):
            state = self.states[name]
            if state.battery >= self.min_battery:
                continue
            free.remove(name)
            self._send(name, self._home_task(name), 'charge')
            self.get_logger().info(
                f'{name} 电量 {state.battery:.0f}%，先回基站充电')

        waiting = [
            cell for cell in self.board
            if cell['status'] == 'wait' and cell['attempts'] < 3
        ]
        while free and waiting:
            best = None
            best_cost = None
            for name in free:
                for cell in waiting:
                    value = self._cost(self.states[name], cell)
                    if value is None:
                        continue
                    if best_cost is None or value < best_cost:
                        best_cost = value
                        best = (name, cell)
            if best is None:
                for name in list(free):
                    state = self.states[name]
                    if state.battery >= 80.0:
                        continue
                    if not any(self._can_fly(state, cell) for cell in waiting):
                        continue
                    free.remove(name)
                    self._send(name, self._home_task(name), 'charge')
                    self.get_logger().info(
                        f'{name} 电量不够去更远的格子，回基站充电')
                break
            name, cell = best
            free.remove(name)
            waiting.remove(cell)
            where = '障碍上空' if cell['aerial'] else '地面格'
            self.get_logger().info(
                f'{where} ({cell["x"]:.1f}, {cell["y"]:.1f}) 派给 {name}，'
                f'代价 {best_cost:.1f}（{self.states[name].kind}，'
                f'电量 {self.states[name].battery:.0f}%）')
            self._send(name, cell, 'survey')

    def _can_fly(self, state, cell):
        return not (cell['aerial'] and state.kind != 'uav')

    def _cost(self, state, cell):
        if not self._can_fly(state, cell):
            return None
        dist = math.hypot(state.x - cell['x'], state.y - cell['y'])
        speed = 0.85 if state.kind == 'uav' else 0.35
        if state.battery < dist / speed + 18.0:
            return None
        cost = dist / speed
        if state.kind == 'uav' and not cell['aerial']:
            cost += 8.0
        cost *= 1.0 + (100.0 - state.battery) / 120.0
        return cost

    def _home_task(self, name):
        x, y = HOMES[name]
        return {
            'x': x, 'y': y, 'aerial': False, 'home': True,
            'attempts': 0, 'status': 'run', 'owner': name,
        }

    def _send(self, name, task, mode):
        task['status'] = 'run'
        task['owner'] = name
        task['mode'] = mode
        self.busy[name] = task
        goal = Mission.Goal()
        goal.target_x = task['x']
        goal.target_y = task['y']
        goal.mode = mode
        future = self._action_clients[name].send_goal_async(
            goal, feedback_callback=self._bind_feedback(name))
        future.add_done_callback(self._bind_response(name))

    def _bind_feedback(self, name):
        def _on_feedback(feedback):
            now = self.get_clock().now().nanoseconds
            last = self.last_feedback.get(name, 0)
            if now - last < 2_000_000_000:
                return
            self.last_feedback[name] = now
            task = self.busy.get(name)
            if task is not None and task.get('mode') == 'charge':
                self.get_logger().info(
                    f'{name} 回充中，电量 {self.states[name].battery:.0f}%')
                return
            self.get_logger().info(
                f'{name} 勘查进度 {feedback.feedback.progress * 100:.0f}% ，'
                f'位置 ({feedback.feedback.x:.1f}, {feedback.feedback.y:.1f})')
        return _on_feedback

    def _bind_response(self, name):
        def _on_response(future):
            goal_handle = future.result()
            if goal_handle is None or not goal_handle.accepted:
                self.get_logger().warn(f'{name} 拒绝了任务，稍后改派')
                self._requeue(name)
                return
            result_future = goal_handle.get_result_async()
            result_future.add_done_callback(self._bind_result(name))
        return _on_response

    def _bind_result(self, name):
        def _on_result(future):
            task = self.busy.get(name)
            result = future.result().result
            if result.success:
                self.busy.pop(name, None)
                if task is not None and task.get('home'):
                    self.get_logger().info(f'{name} {result.message}')
                    return
                state = self.states.get(name)
                if task is not None and state is not None and state.kind == 'uav':
                    self._hold_for_registration(name, task)
                    return
                if task is not None:
                    task['status'] = 'done'
                    task['owner'] = ''
                self.finished += 1
                where = ''
                if task is not None:
                    where = f' ({task["x"]:.1f}, {task["y"]:.1f})'
                self.get_logger().info(f'{name} 勘查完成{where}')
                return
            self.get_logger().warn(f'{name} 未完成：{result.message}')
            self._requeue(name)
        return _on_result

    def _cell_key(self, x, y):
        return (round(float(x), 1), round(float(y), 1))

    def _hold_for_registration(self, name, task):
        task['status'] = 'check'
        task['owner'] = ''
        key = self._cell_key(task['x'], task['y'])
        self.pending_checks[key] = (task, time.monotonic())
        stamped = PoseStamped()
        stamped.header.stamp = self.get_clock().now().to_msg()
        stamped.header.frame_id = 'map'
        stamped.pose.position.x = float(task['x'])
        stamped.pose.position.y = float(task['y'])
        stamped.pose.position.z = 1.0 if task['aerial'] else 0.0
        stamped.pose.orientation.w = float(task['attempts'])
        self.capture_pub.publish(stamped)
        self.get_logger().info(
            f'{name} 到达 ({task["x"]:.1f}, {task["y"]:.1f})，开始航拍配准')

    def on_report(self, msg):
        key = self._cell_key(msg.pose.position.x, msg.pose.position.y)
        pending = self.pending_checks.get(key)
        if pending is None:
            return
        task, _stamp = pending
        code = float(msg.pose.position.z)
        if code < 0.0:
            self._fail_check(task, '配准未通过')
            return
        self.pending_checks.pop(key, None)
        task['status'] = 'done'
        task['owner'] = ''
        self.finished += 1
        wall = self._wall_from_report(msg)
        if wall is not None and self._discover(wall):
            self.get_logger().info(
                f'({task["x"]:.1f}, {task["y"]:.1f}) 配准通过，立面已写入地图')
        elif wall is not None:
            self.get_logger().info(
                f'({task["x"]:.1f}, {task["y"]:.1f}) 配准通过，立面与已知障碍一致')
        else:
            self.get_logger().info(
                f'({task["x"]:.1f}, {task["y"]:.1f}) 配准通过，该格是平地')

    def _fail_check(self, task, reason):
        key = self._cell_key(task['x'], task['y'])
        self.pending_checks.pop(key, None)
        task['status'] = 'wait'
        task['owner'] = ''
        task['attempts'] += 1
        if task['attempts'] >= 3:
            task['status'] = 'drop'
            self.dropped += 1
            self.get_logger().error(
                f'({task["x"]:.1f}, {task["y"]:.1f}) {reason}，'
                f'已失败 {task["attempts"]} 次，放弃')
            return
        self.get_logger().warn(
            f'({task["x"]:.1f}, {task["y"]:.1f}) {reason}，重新派发')

    def _expire_checks(self):
        now = time.monotonic()
        expired = [
            task for task, stamp in self.pending_checks.values()
            if now - stamp > 60.0
        ]
        for task in expired:
            self._fail_check(task, '配准超时')

    def _wall_from_report(self, msg):
        if not msg.text or float(msg.scale.x) < 0.05:
            return None
        center_x, center_y, yaw = (float(part) for part in msg.text.split(','))
        return (
            center_x, center_y, yaw,
            float(msg.scale.x), float(msg.scale.y), float(msg.scale.z),
        )

    def _discover(self, wall):
        x, y, _yaw, _length, _thickness, _height = wall
        for ox, oy, radius in KNOWN_OBSTACLES:
            if math.hypot(x - ox, y - oy) < radius + 0.2:
                return False
        for prev in self.discovered:
            if math.hypot(x - prev[0], y - prev[1]) < 0.4:
                return False
        self.discovered.append(tuple(float(value) for value in wall))
        self.get_logger().info(
            f'发现立面墙 ({x:.2f}, {y:.2f})，长 {_length:.2f} m，地面车改为绕行')
        self._publish_obstacles()
        return True

    def _publish_obstacles(self):
        now = self.get_clock().now().to_msg()
        markers = []
        for index, (x, y, yaw, length, thickness, height) in enumerate(self.discovered):
            marker = Marker()
            marker.header.stamp = now
            marker.header.frame_id = 'map'
            marker.ns = 'discovered'
            marker.id = index
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            marker.pose.position.x = x
            marker.pose.position.y = y
            marker.pose.position.z = height * 0.5
            marker.pose.orientation = _yaw_quaternion(yaw)
            marker.scale.x = length
            marker.scale.y = thickness
            marker.scale.z = height
            marker.color = ColorRGBA(r=0.95, g=0.75, b=0.15, a=0.85)
            markers.append(marker)
        array = MarkerArray()
        array.markers = markers
        self.obstacle_pub.publish(array)

    def _requeue(self, name):
        task = self.busy.pop(name, None)
        if task is None or task.get('home'):
            return
        task['status'] = 'wait'
        task['owner'] = ''
        task['attempts'] += 1
        if task['attempts'] >= 3:
            task['status'] = 'drop'
            self.dropped += 1
            self.get_logger().error(
                f'({task["x"]:.1f}, {task["y"]:.1f}) 已失败 {task["attempts"]} 次，放弃')

    def _publish_tasks(self):
        now = self.get_clock().now().to_msg()
        markers = []
        for index, task in enumerate(self.board):
            markers.append(self._goal_marker(now, index, task))
            markers.append(self._link_marker(now, index, 'links', task))
        for offset, task in enumerate(self.busy.values()):
            if task.get('home'):
                markers.append(self._link_marker(now, 200 + offset, 'home_links', task))
        array = MarkerArray()
        array.markers = markers
        self.task_pub.publish(array)

    def _goal_marker(self, now, index, task):
        marker = Marker()
        marker.header.stamp = now
        marker.header.frame_id = 'map'
        marker.ns = 'goals'
        marker.id = index
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.pose.position.x = task['x']
        marker.pose.position.y = task['y']
        if task['aerial']:
            marker.type = Marker.SPHERE
            marker.pose.position.z = 1.2
            marker.scale.x = marker.scale.y = marker.scale.z = 0.34
        else:
            marker.type = Marker.CUBE
            marker.pose.position.z = 0.04
            marker.scale.x = marker.scale.y = 0.62
            marker.scale.z = 0.04
        if task['status'] == 'done':
            marker.color = ColorRGBA(r=0.3, g=0.95, b=0.4, a=0.9)
        elif task['status'] == 'drop':
            marker.color = ColorRGBA(r=0.9, g=0.25, b=0.2, a=0.9)
        elif task['status'] == 'check':
            marker.color = ColorRGBA(r=0.35, g=0.75, b=1.0, a=0.95)
        elif task['status'] == 'run':
            paint = COLORS.get(task['owner'], (1.0, 0.8, 0.2))
            marker.color = ColorRGBA(r=paint[0], g=paint[1], b=paint[2], a=0.95)
        else:
            marker.color = ColorRGBA(r=0.95, g=0.85, b=0.2, a=0.75)
        return marker

    def _link_marker(self, now, index, namespace, task):
        link = Marker()
        link.header.stamp = now
        link.header.frame_id = 'map'
        link.ns = namespace
        link.id = index
        link.type = Marker.LINE_STRIP
        link.scale.x = 0.04
        owner = task.get('owner', '')
        state = self.states.get(owner)
        if task.get('status') == 'run' and state is not None:
            link.action = Marker.ADD
            paint = COLORS.get(owner, (1.0, 1.0, 1.0))
            link.color = ColorRGBA(r=paint[0], g=paint[1], b=paint[2], a=0.9)
            height = 1.25 if state.kind == 'uav' else 0.2
            goal_z = 1.2 if task.get('aerial') else 0.08
            link.points = [
                Point(x=state.x, y=state.y, z=height),
                Point(x=task['x'], y=task['y'], z=goal_z),
            ]
        else:
            link.action = Marker.DELETE
        return link


def _yaw_quaternion(yaw):
    rotation = Quaternion()
    rotation.z = math.sin(yaw * 0.5)
    rotation.w = math.cos(yaw * 0.5)
    return rotation


def main():
    rclpy.init()
    node = DispatcherNode()
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

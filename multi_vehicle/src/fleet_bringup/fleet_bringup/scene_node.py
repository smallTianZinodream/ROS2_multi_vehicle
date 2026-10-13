import math

import rclpy
from geometry_msgs.msg import Point, Quaternion, TransformStamped
from rclpy.node import Node
from tf2_ros import StaticTransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray

from fleet_bringup.world import COLORS, HOMES, KNOWN_OBSTACLES
from fleet_interfaces.msg import VehicleState


def vehicle_marker_id(name):
    order = {'tb1': 1, 'tb2': 2, 'uav1': 3}
    return order.get(name, 9)


def yaw_quaternion(yaw):
    q = Quaternion()
    q.z = math.sin(yaw * 0.5)
    q.w = math.cos(yaw * 0.5)
    return q


def color(r, g, b, a=1.0):
    from std_msgs.msg import ColorRGBA
    paint = ColorRGBA()
    paint.r = float(r)
    paint.g = float(g)
    paint.b = float(b)
    paint.a = float(a)
    return paint


class SceneNode(Node):
    def __init__(self):
        super().__init__('scene')
        self.declare_parameter('vehicles', ['tb1', 'tb2', 'uav1'])
        self.states = {}
        self.trails = {}
        self.discovered = []
        for name in self.get_parameter('vehicles').value:
            self.create_subscription(
                VehicleState, f'/{name}/state', self._bind(name), 10)
            self.trails[name] = []
        self.pub = self.create_publisher(MarkerArray, '/fleet/scene', 10)
        self.create_subscription(MarkerArray, '/fleet/obstacles', self.on_obstacles, 10)
        self._publish_map_frame()
        self.create_timer(0.1, self.publish_scene)
        self.get_logger().info('场景已发布到 /fleet/scene，用 RViz 的 map 坐标系查看')

    def _publish_map_frame(self):
        broadcaster = StaticTransformBroadcaster(self)
        transform = TransformStamped()
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = 'map'
        transform.child_frame_id = 'fleet_origin'
        transform.transform.rotation.w = 1.0
        broadcaster.sendTransform(transform)
        self._broadcaster = broadcaster

    def on_obstacles(self, msg):
        self.discovered = [
            marker for marker in msg.markers if marker.action == Marker.ADD
        ]

    def _bind(self, name):
        def _on_state(msg):
            self.states[name] = msg
            trail = self.trails[name]
            if not trail or math.hypot(msg.x - trail[-1][0], msg.y - trail[-1][1]) > 0.08:
                trail.append((msg.x, msg.y))
                if len(trail) > 500:
                    del trail[: len(trail) - 500]
        return _on_state

    def publish_scene(self):
        now = self.get_clock().now().to_msg()
        markers = []
        markers.append(self._ground(now))
        for index, (x, y, radius) in enumerate(KNOWN_OBSTACLES):
            markers.append(self._obstacle(now, index, x, y, radius))
        for index, source in enumerate(self.discovered):
            markers.append(self._discovered_wall(now, 20 + index, source))
        for index, (name, (x, y)) in enumerate(HOMES.items()):
            markers.append(self._home(now, index, name, x, y))
        for name, state in self.states.items():
            paint = COLORS.get(name, (0.8, 0.8, 0.8))
            height = 1.25 if state.kind == 'uav' else 0.16
            markers.append(self._body(now, name, state, paint, height))
            markers.append(self._label(now, name, state, height))
            trail = self._trail(now, name, paint, 0.08 if state.kind == 'uav' else 0.05)
            if trail is not None:
                markers.append(trail)
            if state.kind == 'uav':
                markers.append(self._altitude(now, name, state, paint))
        array = MarkerArray()
        array.markers = markers
        self.pub.publish(array)

    def _base(self, now, marker_id, namespace):
        marker = Marker()
        marker.header.stamp = now
        marker.header.frame_id = 'map'
        marker.ns = namespace
        marker.id = marker_id
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        return marker

    def _ground(self, now):
        marker = self._base(now, 0, 'ground')
        marker.type = Marker.CUBE
        marker.pose.position.z = -0.02
        marker.scale.x = 12.0
        marker.scale.y = 10.0
        marker.scale.z = 0.02
        marker.color = color(0.12, 0.16, 0.14)
        return marker

    def _obstacle(self, now, index, x, y, radius):
        marker = self._base(now, index, 'obstacles')
        marker.type = Marker.CYLINDER
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = 0.22
        marker.scale.x = radius * 2.0
        marker.scale.y = radius * 2.0
        marker.scale.z = 0.45
        marker.color = color(0.65, 0.65, 0.68)
        return marker

    def _discovered_wall(self, now, index, source):
        marker = self._base(now, index, 'obstacles')
        marker.type = Marker.CUBE
        marker.pose = source.pose
        marker.scale = source.scale
        marker.color = color(0.95, 0.75, 0.15, 0.85)
        return marker

    def _home(self, now, index, name, x, y):
        marker = self._base(now, index, 'homes')
        marker.type = Marker.CYLINDER
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = 0.05
        marker.scale.x = 0.42
        marker.scale.y = 0.42
        marker.scale.z = 0.08
        paint = COLORS.get(name, (0.8, 0.8, 0.8))
        marker.color = color(*paint, 0.55)
        return marker

    def _body(self, now, name, state, paint, height):
        marker = self._base(now, vehicle_marker_id(name), 'vehicles')
        marker.pose.position.x = state.x
        marker.pose.position.y = state.y
        marker.pose.position.z = height
        if state.kind == 'uav':
            marker.type = Marker.SPHERE
            marker.scale.x = marker.scale.y = marker.scale.z = 0.32
        else:
            marker.type = Marker.ARROW
            marker.pose.orientation = yaw_quaternion(state.yaw)
            marker.scale.x = 0.55
            marker.scale.y = 0.14
            marker.scale.z = 0.14
        marker.color = color(*paint)
        return marker

    def _label(self, now, name, state, height):
        marker = self._base(now, vehicle_marker_id(name), 'labels')
        marker.type = Marker.TEXT_VIEW_FACING
        marker.pose.position.x = state.x
        marker.pose.position.y = state.y
        marker.pose.position.z = height + 0.35
        marker.scale.z = 0.22
        marker.color = color(1.0, 1.0, 1.0)
        if state.phase == 'charge':
            marker.text = f'{name} 充电 {state.battery:.0f}%'
        elif state.phase == 'survey':
            marker.text = f'{name} {state.battery:.0f}%'
        else:
            marker.text = f'{name} 待命 {state.battery:.0f}%'
        return marker

    def _trail(self, now, name, paint, width):
        points = self.trails[name]
        if len(points) < 2:
            return None
        marker = self._base(now, vehicle_marker_id(name), 'trails')
        marker.type = Marker.LINE_STRIP
        marker.scale.x = width
        marker.color = color(*paint, 0.85)
        marker.points = [Point(x=x, y=y, z=0.05) for x, y in points]
        return marker

    def _altitude(self, now, name, state, paint):
        marker = self._base(now, vehicle_marker_id(name), 'altitude')
        marker.type = Marker.LINE_LIST
        marker.scale.x = 0.03
        marker.color = color(*paint, 0.7)
        marker.points = [
            Point(x=state.x, y=state.y, z=0.02),
            Point(x=state.x, y=state.y, z=1.25),
        ]
        return marker


def main():
    rclpy.init()
    node = SceneNode()
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

# 多载具协同勘查

ROS 2 仿真里，两台地面车（`tb1`、`tb2`）和一架无人机（`uav1`）一起覆盖一块场地。调度器按距离、电量和机型派任务；地面车绕开已知障碍、同伴和后来发现的立面墙；无人机飞到障碍上空做航拍配准，配准通过后把立面写回地图，地面车随即改道。

## 环境

- ROS 2（`rclpy`、`launch`、`rviz2`）
- Python 3
- NumPy、OpenCV（`opencv-python`）

接口包用 `ament_cmake`，执行包用 `ament_python`。

## 目录

```
multi_vehicle/
├── src/
│   ├── fleet_interfaces/     状态消息与任务动作
│   └── fleet_bringup/        载具、调度、场景、航拍节点和启动文件
└── aerial_register/          低空航拍的多平面配准与三角化
```

## 编译与运行

在 ROS 2 工作空间中，把本仓库的 `multi_vehicle` 当作工作空间根目录（或把它的 `src` 链到已有工作空间）：

```bash
cd multi_vehicle
colcon build
source install/setup.bash
ros2 launch fleet_bringup fleet.launch.py
```

启动后会拉起三台载具、调度器、场景、航拍节点和 RViz。RViz 使用 `map` 坐标系，配置在 `src/fleet_bringup/rviz/fleet.rviz`。

只跑航拍配准演示、不启动 ROS：

```bash
cd multi_vehicle
python aerial_register/demo.py
```

脚本会在 `multi_vehicle/` 下写出 `aerial_registration.png`（两视图与配准叠加）和 `aerial_points.png`（三角化点云的侧视与俯视）。

## 节点

| 节点 | 作用 |
| --- | --- |
| `vehicle_node` | 单机执行。接收 `Mission`，差速运动，地面车做局部避障，电量随行驶下降，回基站后充电。 |
| `dispatcher_node` | 把勘查格按代价分给空闲载具。障碍上空只派给无人机。电量不够或去不了更远的格子时先回充。同一格失败三次后放弃。 |
| `scene_node` | 把地面、已知圆柱障碍、基站、车体、航迹和已发现立面发到 `/fleet/scene`。 |
| `capture_node` | 无人机到达后做多平面单应和基础矩阵估计，三角化点云，拟合立面墙，并把结果回报调度器。 |

默认编队在 `launch/fleet.launch.py`：

| 载具 | 类型 | 基站 |
| --- | --- | --- |
| `tb1` | 地面车，0.35 m/s | (-3.5, -2.2) |
| `tb2` | 地面车，0.35 m/s | (-3.5, 0.2) |
| `uav1` | 无人机，0.85 m/s | (-3.5, 2.2) |

地面勘查格为 `x ∈ {-2.2, 0.2, 2.6}`、`y ∈ {-2.2, 0.2, 2.2}` 中未落在障碍膨胀区内的点。两处真实障碍中，地面车启动时只知道 `(1.6, 0.6)`；`(-0.4, 1.7)` 要等无人机配准成功后才写入 `/fleet/obstacles`。

## 接口

`fleet_interfaces/msg/VehicleState.msg`：载具编号、机型、位姿、是否空闲、电量、进度和阶段（`idle` / `survey` / `charge`）。

`fleet_interfaces/action/Mission.action`：

- 目标：`target_x`、`target_y`、`mode`（`survey` 或 `charge`）
- 结果：是否成功、说明
- 反馈：进度和当前位置

调度器在电量低于 `min_battery`（默认 40%）时强制回充。代价是预计行驶时间，再按剩余电量加权；无人机去普通地面格会额外加代价，把平地尽量留给地面车。

## 话题

| 话题 | 类型 | 说明 |
| --- | --- | --- |
| `/{id}/state` | `VehicleState` | 各机状态，同伴之间互订以保持间距 |
| `/{id}/mission` | `Mission` 动作 | 调度器下发的勘查或回充 |
| `/fleet/tasks` | `MarkerArray` | 待查、执行中、配准中、完成、放弃的格子，以及派工连线 |
| `/fleet/scene` | `MarkerArray` | 场景与车体 |
| `/fleet/capture` | `PoseStamped` | 无人机到达后请求航拍 |
| `/fleet/survey` | `MarkerArray` | 配准得到的点云 |
| `/fleet/survey_report` | `Marker` | 配准结论；`z < 0` 为失败，否则可带立面墙参数 |
| `/fleet/obstacles` | `MarkerArray` | 新发现的立面，供地面车绕行 |

RViz 里格子颜色：黄为待分配，车身颜色为执行中，蓝为配准中，绿为完成，红为放弃。点云里绿色是地面，橙色是立面。

## 航拍配准

`aerial_register/` 实现 Cai 等在 *Remote Sensing* 2018, 10(2):291 中的残差引导采样，用来估计单应矩阵。固定窗口的 Multi-GS 在低空、倾斜、重复纹理上容易失配，这里提供文中的三种相似度，默认使用逆序残差（`rro`）：

- `rw`：松弛窗口，在一段窗口长度里取最高重叠比例
- `sw`：滑动窗口，重叠结构数固定，窗口长度可变
- `rro`：逆序残差，重叠之外再惩罚残差名次颠倒

同一相机运动下，地面和立面各有一个单应矩阵，但共用一个基础矩阵。`estimate_multi_homography` 抽出一个平面的内点后再估下一个平面；`estimate_fundamental` 用八点法估计基础矩阵；`triangulate` 恢复第二台相机姿态并把内点三角化。立面点再拟合成一堵墙（中心、朝向、长、厚、高）。

无人机格会用两张不同角度的影像各估一次，留下更能解释控制点的那次。控制点转移误差、地面深度和立面距离都要过关，配准才算成功。

## 许可

Apache-2.0。

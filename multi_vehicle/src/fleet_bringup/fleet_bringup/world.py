import math

# 仿真里真实存在的障碍。机器人一开始只知道第一处，另一处等无人机配准后再公布。
OBSTACLES = (
    (1.6, 0.6, 0.40),
    (-0.4, 1.7, 0.35),
)
KNOWN_OBSTACLES = (
    (1.6, 0.6, 0.40),
)

HOMES = {
    'tb1': (-3.5, -2.2),
    'tb2': (-3.5, 0.2),
    'uav1': (-3.5, 2.2),
}

COLORS = {
    'tb1': (0.25, 0.55, 1.0),
    'tb2': (0.20, 0.85, 0.35),
    'uav1': (1.0, 0.50, 0.10),
}


def _blocked(x, y):
    for ox, oy, radius in OBSTACLES:
        if math.hypot(x - ox, y - oy) < radius + 0.45:
            return True
    return False


def survey_cells():
    cells = []
    for x in (-2.2, 0.2, 2.6):
        for y in (-2.2, 0.2, 2.2):
            if _blocked(x, y):
                continue
            cells.append({'x': x, 'y': y, 'aerial': False})
    for ox, oy, _radius in OBSTACLES:
        cells.append({'x': ox, 'y': oy, 'aerial': True})
    return cells

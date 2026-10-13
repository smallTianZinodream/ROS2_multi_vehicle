import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'fleet_bringup'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob(os.path.join('launch', '*.py'))),
        (os.path.join('share', package_name, 'rviz'),
            glob(os.path.join('rviz', '*.rviz'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jsetc',
    maintainer_email='jsetc@todo.todo',
    description='多载具协同：单机执行节点、任务调度和启动文件',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'vehicle_node = fleet_bringup.vehicle_node:main',
            'dispatcher_node = fleet_bringup.dispatcher_node:main',
            'scene_node = fleet_bringup.scene_node:main',
            'capture_node = fleet_bringup.capture_node:main',
        ],
    },
)

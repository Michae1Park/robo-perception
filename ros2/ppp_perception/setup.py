from setuptools import find_packages, setup

package_name = 'ppp_perception'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Michael Park',
    maintainer_email='parkmjb@gmail.com',
    description='Visual-prompt detection (YOLOE), FoundationPose bridge and table/free-space perception.',
    license='MIT',
    entry_points={
        'console_scripts': [
            'detector_node = ppp_perception.detector_node:main',
            'pose_node = ppp_perception.pose_node:main',
            'scene_node = ppp_perception.scene_node:main',
        ],
    },
)

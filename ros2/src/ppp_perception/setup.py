from setuptools import setup

setup(
    name='ppp_perception',
    version='0.1.0',
    packages=['ppp_perception'],
    data_files=[('share/ament_index/resource_index/packages', ['resource/ppp_perception']),
                ('share/ppp_perception', ['package.xml'])],
    install_requires=['setuptools'],
    zip_safe=True,
    entry_points={'console_scripts': [
        'detect_node = ppp_perception.detect_node:main',
        'pose_node = ppp_perception.pose_node:main',
        'grasp_node = ppp_perception.grasp_node:main',
        'place_node = ppp_perception.place_node:main',
        'eval_node = ppp_perception.eval_node:main',
    ]},
)

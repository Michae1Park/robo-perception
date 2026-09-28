from setuptools import find_packages, setup

package_name = 'ppp_manipulation'

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
    description='Grasp and placement planning, MoveIt 2 task execution and a sim trajectory executor.',
    license='MIT',
    entry_points={
        'console_scripts': [
            'grasp_node = ppp_manipulation.grasp_node:main',
            'place_node = ppp_manipulation.place_node:main',
            'executor_node = ppp_manipulation.executor_node:main',
            'task_manager = ppp_manipulation.task_manager:main',
        ],
    },
)

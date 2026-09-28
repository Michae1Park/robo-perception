from setuptools import setup

package_name = 'ppp_common'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Michael Park',
    maintainer_email='parkmjb@gmail.com',
    description='Shared geometry, config and ROS helpers for prompt-pick-place.',
    license='MIT',
)

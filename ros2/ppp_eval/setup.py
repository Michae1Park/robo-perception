from setuptools import find_packages, setup

package_name = 'ppp_eval'

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
    description='Automated ground-truth evaluation of the pick-and-place pipeline.',
    license='MIT',
    entry_points={
        'console_scripts': [
            'eval_node = ppp_eval.eval_node:main',
            'summarize = ppp_eval.summarize:main',
        ],
    },
)

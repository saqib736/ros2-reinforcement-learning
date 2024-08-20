import os
from glob import glob 
from setuptools import find_packages, setup

package_name = 'rl_racing'

cur_directory_path = os.path.abspath(os.path.dirname(__file__))

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        
        (os.path.join('share', package_name,'launch'), glob('launch/*.launch.py')),
        
        
        
        
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='saqib',
    maintainer_email='saqibmehmood736@gmail.com',
    description='Reinforcement learning for Robot racing based on stabel_baseline and OpenAI gym',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        'start_learning = rl_racing.start_learning:main',
        ],
    },
)

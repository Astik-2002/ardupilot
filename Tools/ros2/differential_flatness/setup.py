from setuptools import find_packages, setup

package_name = 'differential_flatness'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    entry_points={
        'console_scripts': [
                'commander = differential_flatness.differential_flatness:main',
                'commander_plotter = differential_flatness.df_plotting:main'
        ],
    },
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='astik',
    maintainer_email='41.astiksrivastava@gmail.com',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
)

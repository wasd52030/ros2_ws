from setuptools import find_packages, setup

package_name = 'cv_pubsub'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='sobel',
    maintainer_email='sobel@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'webcam_pub = cv_pubsub.webcam_pub:main',
            'webcam_sub = cv_pubsub.webcam_sub:main',
        ],
    },
)

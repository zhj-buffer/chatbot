from setuptools import setup

package_name = 'py_nvasrnlptts'

submodules = "py_nvasrnlptts/nvaudio"

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name, submodules],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='nx',
    maintainer_email='nx@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        'talker = py_nvasrnlptts.publisher_member_function:main',
         'listener = py_nvasrnlptts.subscriber_member_function:main',
        ],
    },
)

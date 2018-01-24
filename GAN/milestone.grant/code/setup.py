from setuptools import setup

setup(
    name="augmentator",
    version='0.1',
    py_modules=['augmentator'],
    install_requires=[
        'Click',
    ],
    entry_points='''
        [console_scripts]
        gen=augmentator:augmentation
        pre=preprocessing:preprocess
    ''',
)

from setuptools import setup

VERSION = '0.1.0'

with open('README.md', 'r') as file:
    long_description = file.read()

setup(
    name='nlptools',
    version=VERSION,
    description='Suite of tools for training models and mining text.',
    long_description=long_description,
    long_description_conten_type='text/markdown',
    py_modules=['nlptools'],
    package_dir={'':'nlptools'}
)
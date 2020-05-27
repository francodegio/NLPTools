from setuptools import setup

with open('README.md', 'r') as file:
    long_description = file.head()
setup(
    name='nlptools',
    version='0.1.0',
    description='Suite of tools for training models and mining text.',
    long_description=long_description,
    long_description_conten_type='text/markdown',
    py_modules=['nlptools'],
    package_dir={'':'nlptools'}
)
from setuptools import setup, find_packages

VERSION = '0.1.0'

with open('README.md', 'r') as file:
    long_description = file.read()

setup(
    name='nlptools',
    version=VERSION,
    description='Suite of tools for training models and mining text.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages(where='src'),
    # py_modules=['nlptools', 'tools', 'models'],
    package_dir={'':'src'},
    install_requires=[
        'numpy', 
        'pandas', 
        'rapidfuzz',
        'spacy',
        'num2words']
)
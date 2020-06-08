from setuptools import setup, find_packages

VERSION = '0.2.3'

with open('README.md', 'r') as file:
    long_description = file.read()

setup(
    name='nlptools',
    version=VERSION,
    description='Suite of tools for training models and mining text.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages(where='src'),
    package_dir={'':'src'},
    package_data={
        'nlptools': ['data/*.csv.zip'],
    },
    include_package_data=True,
    install_requires=[
        'numpy', 
        'pandas', 
        'rapidfuzz',
        'spacy',
        'num2words', 
        'wheel']
)
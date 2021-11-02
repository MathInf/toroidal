import setuptools


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='toroidal',
    version='0.0.1.post2',
    description="Toroidal - Lightweight Transformers for PyTorch",
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='MathInf GmbH',
    url='https://github.com/mathinf/toroidal/',
    install_requires=['torch'],
    packages=setuptools.find_packages(),
)

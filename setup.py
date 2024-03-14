from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='fastfiz-env',
    description='Gymnasium environments for FastFiz',
    version='0.0.1',
    license='MIT',
    install_requires=requirements,
    packages=find_packages(where='src', include=['fastfiz_env']),
    package_dir={"": "src"},
)

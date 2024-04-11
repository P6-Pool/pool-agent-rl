from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="fastfiz-env",
    description="Gymnasium environment for FastFiz pool simulator",
    version="0.0.1",
    license="MIT",
    install_requires=requirements,
    test_requires=["pytest"],
    packages=find_packages(where="src"),
    package_dir={"": "src"},
)

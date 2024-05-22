import re
from setuptools import setup, find_packages

def get_version():
    with open("fastfiz_env/__init__.py", "r") as f:
        for line in f:
            match = re.match(r"__version__\s*=\s*['\"]([^'\"]+)['\"]", line)
            if match:
                return match.group(1)
    raise RuntimeError("Version not found in __init__.py")


setup(version=get_version(), packages=find_packages())

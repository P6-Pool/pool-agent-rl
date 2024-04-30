import re
from setuptools import setup

# with open("requirements.txt") as f:
#     requirements = f.read().splitlines()


def get_version():
    with open("src/fastfiz_env/__init__.py", "r") as f:
        for line in f:
            match = re.match(r"__version__\s*=\s*['\"]([^'\"]+)['\"]", line)
            if match:
                return match.group(1)
    raise RuntimeError("Version not found in __init__.py")


setup(
    version=get_version(),
    # install_requires=requirements,
    # test_requires=["pytest"],
    # packages=find_packages(where="src"),
    # package_dir={"": "src"},
)

"""Bayesian online changepoint detection.

Based on template from
  https://github.com/dtolpin/python-project-skeleton
"""

from setuptools import setup, find_packages
from os import path
import bocd

# Get the long description from the README file
here = path.abspath(path.dirname(__file__))
with open(path.join(here, "README.md")) as f:
    long_description = f.read()


setup(
    name="changepoint",
    version=bocd.__version__,

    description="Online implementation of online changepoint detection",
    long_description=long_description,

    packages=find_packages(exclude=["doc", "data"]),

    # Generating the command-line tool
    entry_points={
        "console_scripts": [
        ]
    },

    # author and license
    author="David Tolpin",
    author_email="david.tolpin@gmail.com",
    license="MIT",

    # dependencies, a list of rules
    install_requires=["numpy", "scipy"],
    # add links to repositories if modules are not on pypi
    dependency_links=[
    ],

    #  PyTest integration
    setup_requires=["pytest-runner"],
    tests_require=["pytest"]
)

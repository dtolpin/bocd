"""Bayesian online changepoint detection.

Based on template from
  https://github.com/dtolpin/python-project-skeleton
"""

from setuptools import setup, find_packages
from os import path
import clew.changepoint

# Get the long description from the README file
here = path.abspath(path.dirname(__file__))
with open(path.join(here, "README.md")) as f:
    long_description = f.read()


setup(
    name="changepoint",
    version=clew.changepoint.__version__,

    description="Online implementation of online changepoint detection",
    long_description=long_description,
    url="https://intensix.atlassian.net/browse/AR-259",

    packages=find_packages(exclude=["doc", "data"]),

    # source code layout
    namespace_packages=["clew"],

    # Generating the command-line tool
    entry_points={
        "console_scripts": [
        ]
    },

    # author and license
    author="David Tolpin",
    author_email="david.t@clewmed.com",
    license="Proprietary",

    # dependencies, a list of rules
    install_requires=["numpy", "scipy"],
    # add links to repositories if modules are not on pypi
    dependency_links=[
    ],

    #  PyTest integration
    setup_requires=["pytest-runner"],
    tests_require=["pytest"]
)

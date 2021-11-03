"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import setuptools
from setuptools import setup,find_packages
import sys, os

setup(name="real_data_transformations",
      packages=find_packages(),
      long_description=open('README.md').read(),
)
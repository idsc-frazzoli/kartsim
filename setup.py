#!/usr/bin/env python3

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
        name='kartsim',
        version='0.0',
        author='Michael von Bueren',
        author_email='vbueremi@ethz.ch',
        package_dir={'': 'src'},
        packages=setuptools.find_packages('src'),
        test_suite='tests',
        url='https://github.com/idsc-frazzoli/kartsim',
        description='Gokart simulator',
        long_description=long_description,
        long_description_content_type="text/markdown",
        python_requires='>=3',
        install_requires=['pycontracts']
)

#!/usr/bin/env python
__author__ = 'Zoheyr Doctor'
__description__ = 'hi'
from distutils.core import setup

setup(
        name='GP ROM',
        version='0.1',
        url='https://github.com/zodoctor/GaussianProcessROM',
        author = __author__,
        author_email = 'zoheyr@gmail.com',
        description = __description__,
        scripts = [
            'bin/runalloverlaps1D_qchi.py',
            'bin/create_WF_ROM_dataset.py',
            ],
        packages = [
            'gp_rom',
            ],
        data_files = [],
        requires = [])


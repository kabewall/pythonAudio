#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name = "pythonAudio",
    version = '1.0.0'
    url = 'https://github.com/kabewall/pythonAudio.git',
    long_description = 'readme'
    license = license,
    author = "kabewall",
    author_email = "kouhei.kabe@gmail.com",
    keywords = '',
    packages = find_packages(exclude=('test', 'docs'))
)

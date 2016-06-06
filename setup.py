# -*- coding: utf-8 -*-

from setuptools import setup
from setuptools import find_packages


setup(name='hyper',
      version='0.0.1',
      description='Knowledge Propagation in Knowledge Graphs',
      author='Pasquale Minervini',
      author_email='p.minervini@gmail.com',
      url='https://github.com/pminervini/knowledge-propagation',
      test_suite='tests',
      license='MIT',
      install_requires=[
          'theano>=0.8.0',
          'terminaltables>=2.1.0',
          'colorclass>=2.2.0',
          'pylearn2>=0.1dev'
      ],
      packages=find_packages())

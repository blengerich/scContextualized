"""
Setup and requirements for scContextualized.ML
"""

from setuptools import find_packages, setup

DESCRIPTION = "Helper tools for contextualized analysis of single-cell data."
VERSION = '0.0.9'

setup(
    name='scContextualized',
    author="scContextualized Team",
    url="https://github.com/blengerich/scContextualized",
    version=VERSION,
    description=DESCRIPTION,
    packages=find_packages(),
  install_requires=[
      'pytorch-lightning',
      'torch',
      'numpy',
      'tqdm',
      'scikit-learn',
      'matplotlib',
      'pandas',
      'umap-learn',
      'numpy>=1.19.2',
      'ipywidgets',
      'torchvision',
      'scanpy',
      'contextualized-ml',
  ],
)

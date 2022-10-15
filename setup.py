import setuptools

setuptools.setup(name='scContextualized',
      packages=['scContextualized'],
      version='0.0.8',
      install_requires=[
          'pytorch-lightning',
          'torch',
          'numpy',
          'tqdm',
          'scikit-learn',
          'python-igraph',
          'matplotlib',
          'pandas',
          'umap-learn',
          'numpy>=1.19.2',
          'ipywidgets',
          'torchvision',
          'scanpy',
      ],
)

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
          'interpret',
          'tensorflow>=2.4.0',
          'tensorflow-addons',
          'numpy>=1.19.2',
          'ipywidgets',
          'torchvision',
          'scanpy',
      ],
)

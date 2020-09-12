from distutils.core import setup
setup(
  name = 'MLevaluator',
  packages = ['MLevaluator', 'MLevaluator.auxfunc'],
  version = '0.0.1',
  license= '',
  description = 'Package for evaluate Computer Vision models that return MLgeometry objects',
  author = 'Juan Carlos Arbelaez',
  author_email = 'juanarbelaez@vaico.com.co',
  url = 'https://jarbest@bitbucket.org/jarbest/mlevaluator.git',
  download_url = 'https://bitbucket.org/jarbest/mlevaluator/get/master.tar.gz',
  keywords = ['vaico', 'common', 'ml', 'computer vision', 'machine learning', 'evaluation'],
  install_requires=['MLgeometry', 'tqdm', 'opencv-python', 'numpy', 'matplotlib'],
  classifiers=[
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Developers',
    'Topic :: Software Development :: Build Tools',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
  ]
)
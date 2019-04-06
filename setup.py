from distutils.core import setup


try:
    import lifelines
except ImportError:
    raise Exception('lifelines is not installed.')

try:
    import sksurv
except ImportError:
    raise Exception('scikit-survival is not installed.')

setup(name='ttea',
      version='0.1',
      description='Time to event analysis',
      author='Lei Huang',
      author_email='lh389@cornell.edu',
      url='https://github.com/leihuang/ttea',
      packages=['distutils', 'distutils.command'],
     )

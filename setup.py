from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

#ext_modules = [ Extension( 'c2v2_util', ['c2v2_util.pyx'], 
#                           language = 'c++',
#                           extra_compile_args = ['-std=c++11', '-O3', '-lpthread'],
#                           include_dirs = [numpy.get_include()] ) ]

ext_modules = [ Extension( 'cu', ['cu.pyx'], 
                           language = 'c++',
                           extra_compile_args = ['-std=c++11', '-O3', '-lpthread'],
                           include_dirs = [numpy.get_include()] ) ]
setup(
	name = 'cu',
	cmdclass = { 'build_ext': build_ext },
 	ext_modules = ext_modules,
)
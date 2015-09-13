from distutils.core import setup, Extension
import sys
if sys.platform == 'darwin':
    module1 = Extension('ale_python_interface.ale_c_wrapper',
                        libraries = ['ale_c'],
                        include_dirs = ['src','/usr/local/include/SDL'],
                        library_dirs = ['ale_python_interface'],
                        extra_compile_args=['-D__STDC_CONSTANT_MACROS','-D__USE_SDL'],
                        extra_link_args=['-framework','Cocoa'],
                        sources=['ale_python_interface/ale_c_wrapper.cpp'])
else:
    module1 = Extension('ale_python_interface.ale_c_wrapper',
                        libraries = ['ale_c'],
                        include_dirs = ['src'],
                        library_dirs = ['ale_python_interface'],
                        extra_compile_args=['-D__STDC_CONSTANT_MACROS'],
                        sources=['ale_python_interface/ale_c_wrapper.cpp'])
setup(name = 'ale_python_interface',
      version='0.0.1',
      description = 'Arcade Learning Environment Python Interface',
      url='https://github.com/bbitmaster/ale_python_interface',
      author='Ben Goodrich',
      license='GPL',
      ext_modules = [module1],
      packages=['ale_python_interface'],
      package_dir={'ale_python_interface': 'ale_python_interface'},
      package_data={'ale_python_interface': ['libale_c.so']})

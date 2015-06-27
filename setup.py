from distutils.core import setup, Extension
import shutil

shutil.copy('./libale.so', 'ale_python_interface/')

module1 = Extension('ale_python_interface.ale_c_wrapper',
                    libraries = ['ale'],
                    include_dirs = ['src'],
                    library_dirs = ['ale_python_interface/..'],
                    runtime_library_dirs = ['$ORIGIN'],
                    extra_compile_args=['-D__STDC_CONSTANT_MACROS'],
                    sources=['ale_python_interface/ale_c_wrapper.cpp'])
setup(name = 'ale_python_interface',
      description = 'Arcade Learning Environment Python Interface',
      url='https://github.com/bbitmaster/ale_python_interface',
      author='Ben Goodrich',
      license='GPL',
      ext_modules = [module1],
      packages=['ale_python_interface'],
      package_dir={'ale_python_interface': 'ale_python_interface'},
      package_data={'ale_python_interface':
                    ['libale_c_wrapper.so', 'libale.so']})

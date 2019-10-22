import numpy as np

try:
    from setuptools import setup, find_packages, Extension
except ImportError:
    from distutils.core import setup, find_packages, Extension

include_dirs = [np.get_include()]
compile_args = ['/openmp']

euphonic_extension = Extension(
    'euphonic._euphonic',
    extra_compile_args=compile_args,
    sources=['c/_euphonic.c'],
    include_dirs=include_dirs
)

with open('README.rst', 'r') as f:
    long_description = f.read()

packages = ['euphonic',
            'euphonic.data',
            'euphonic.plot',
            'euphonic._readers']

scripts = ['scripts/dispersion.py',
           'scripts/dos.py',
           'scripts/optimise_eta.py']

setup(
    name='euphonic',
    version='0.1dev3',
    author='Rebecca Fair',
    author_email='rebecca.fair@stfc.ac.uk',
    description=(
        'Euphonic calculates phonon bandstructures and inelastic neutron '
        'scattering intensities from modelling code outputs (e.g. CASTEP)'),
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/pace-neutrons/Euphonic',
    packages=packages,
    install_requires=[
        'numpy>=1.9.1',
	'scipy>=1.0.0',
        'seekpath>=1.1.0',
        'pint>=0.8.0'
    ],
    extras_require={
        'matplotlib': ['matplotlib>=1.4.2'],
    },
    scripts=scripts,
    ext_modules=[euphonic_extension]
)

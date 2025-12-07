# Setuptools for ASPEigCIM package
import io

from setuptools import setup, find_packages


__version__ = '0.1'

# Readme file as long_description:
long_description = ('==========================================================================================\n' +
                    'Adiabatic state preparation from the eigendecomposition of the combined interaction matrix\n' +
                    '==========================================================================================\n')

# Read in package requirements.txt
requirements = open('requirements.txt').readlines()
requirements = [r.strip() for r in requirements]

setup(
    name='aspeigcim',
    version=__version__,
    author='Dyon van Vreumingen',
    author_email='d.vanvreumingen@uva.nl',
    url='https://github.com/DyonvVr-UvA/ASPEigCIM',
    description=('Adiabatic state preparation from combined interaction matrix eigendecomposition'),
    long_description=long_description,
    install_requires=requirements,
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    include_package_data=True,
)


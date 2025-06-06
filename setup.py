#
# metavirommodel setuptools script
#
# This file is part of metavirommodel
# (https://github.com/SABS-R3-Epidemiology/metavirommodel.git) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md
# for copyright notice and full license details.
#
from setuptools import setup, find_packages


def get_version():
    """
    Get version number from the metavirommodel module.
    The easiest way would be to just ``import metavirommodel ``, but note that this may  # noqa
    fail if the dependencies have not been installed yet. Instead, we've put
    the version number in a simple version_info module, that we'll import here
    by temporarily adding the oxrse directory to the pythonpath using sys.path.
    """
    import os
    import sys

    sys.path.append(os.path.abspath('metavirommodel'))
    from version_info import VERSION as version
    sys.path.pop()

    return version


def get_readme():
    """
    Load README.md text for use as description.
    """
    with open('README.md', encoding='utf-8') as f:
        return f.read()


setup(
    # Module name (lowercase)
    name='metavirommodel',

    # Version
    version=get_version(),

    description='This is a one-week project in which we are using branching processes to estimate the time-dependent reproduction number of a disease.',  # noqa

    long_description=get_readme(),

    license='BSD 3-Clause "New" or "Revised" License',

    # author='',

    # author_email='',

    maintainer='',

    maintainer_email='',

    url='https://github.com/SABS-R3-Epidemiology/metavirommodel.git',

    # Packages to include
    packages=find_packages(include=('metavirommodel', 'metavirommodel.*')),
    include_package_data=True,

    # List of dependencies
    install_requires=[
        # Dependencies go here!
        'matplotlib',
        'numpy',
        'pandas',
        'plotly',
        'scipy',
        'pints'
    ],
    python_requires='>=3.9',
    extras_require={
        'docs': [
            # Sphinx for doc generation. Version 1.7.3 has a bug:
            'sphinx>=1.5, !=1.7.3',

            # Nice theme for docs
            'alabaster',
        ],
        'dev': [
            # Flake8 for code style checking
            'flake8>=3',
        ],
    }
)

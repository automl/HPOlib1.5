# -*- encoding: utf-8 -*-
import setuptools

with open("hpolib/__version__.py") as fh:
    version = fh.readlines()[-1].split()[-1].strip("\"'")

with open('requirements.txt') as fh:
    requirements = fh.read()
requirements = requirements.split('\n')
requirements = [requirement.strip() for requirement in requirements]


setuptools.setup(
    name='hpolib2',
    description='Automated machine learning.',
    version=version,
    packages=setuptools.find_packages(exclude=['test']),
    install_requires=requirements,
    tests_require=[
        'pynisher',
        'nose',
        'mock'],
    test_suite='nose.collector',
    scripts=[],
    #include_package_data=True,
    author_email='eggenspk@informatik.uni-freiburg.de',
    license='GPLv3',
    platforms=['Linux'],
    classifiers=[
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Development Status :: 3 - Alpha',
        'Natural Language :: English',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: POSIX :: Linux',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering',
        'Topic :: Software Development',
    ]
)

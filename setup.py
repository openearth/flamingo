from setuptools import setup, find_packages

setup(
    name='flamingo',
    version='0.0',
    author='Bas Hoonhout',
    author_email='bas.hoonhout@deltares.nl',
    packages=find_packages(),
    description='Flamingo Image Analysis Toolbox',
    long_description=open('README.txt').read(),
    install_requires=[
        'pandas',
        'scipy',
        'numpy',
        'docopt',
        'psutil'
        # also opencv but that's not available on pip
    ],
    #setup_requires=[
    #    'sphinx',
    #    'sphinx_rtd_theme'
    #],
    tests_require=[
        'nose'
    ],
    test_suite='nose.collector',
    entry_points={'console_scripts': [
        '{0} = flamingo.classify:main'.format(
            'classify-images'),
        '{0} = flamingo.rectify:main'.format(
            'rectify-images'),
        '{0} = flamingo.calibrate:main'.format(
            'calibrate-camera'),
        '{0} = flamingo.console:classify'.format(
            'flamingo-classify'),
        '{0} = flamingo.console:rectify'.format(
            'flamingo-rectify'),
        '{0} = flamingo.console:calibrate'.format(
            'flamingo-calibrate'),
    ]},
    #data_files=[
    #    ('images', [
    #        "data/argusnl/1399006805.Fri.May.02_05_00_05.UTC.2014.kijkduin.c5.snap.jpg",
    #        "data/argusnl/1399019402.Fri.May.02_08_30_02.UTC.2014.kijkduin.c2.snap.jpg"
    #    ])
    #]

)

from setuptools import setup

setup(
    name='mltoolbox',
    version='dev',    # PyPI version automatically set by CI/CD tool
    author='Amine Laghaout',
    description='Tool for training models.',
    install_requires=[
        'pandas>=0.23',
        'matplotlib>=2.0',
        'keras>=2.0'
    ],
    packages=['mltoolbox'],
    python_requires=">=3.6",
    zip_safe=False,
)

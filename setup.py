from setuptools import setup

setup(
    name='learners',
    version='dev',    # PyPI version automatically set by CI/CD tool
    author='laghaout',
    description='Object-oriented learners for machine learning.',
    install_requires=[
        'seaborn>=0.11.0',
        'matplotlib>=3.3.0',
        'numpy>=1.19.1',
        'pandas>=1.1.1'
        'scikit-learn>=0.23.2',
        'tensorflow>=2.1.0'
    ],
    packages=['learners'],
    python_requires=">=3.7",
    zip_safe=False,
)

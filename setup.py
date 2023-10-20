from setuptools import setup, find_packages

setup(
    name='sentence-models',
    version='0.0.1',
    packages=find_packages(),
    install_requires=[
        'scikit-learn', 'spacy', 'lazylines', 'embetter[sentence-tfm]'
    ],
    python_requires='>=3.6',
)

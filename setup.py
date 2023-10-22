from setuptools import setup, find_packages

setup(
    name='sentence-models',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'scikit-learn', 'spacy', 'lazylines', 'embetter[sentence-tfm]', 'skops', 'rich'
    ],
    python_requires='>=3.6',
)

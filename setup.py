from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name='DL_vs_HateSpeech',       # Replace with your package name 
    version='0.1',                  # Package version
    authors='',             # Author name
    author_email='',  # Author email
    description='Deep learning project against hate speech',  # Short description
    long_description='TBD',  # Full description
    long_description_content_type='text/markdown',  # Description content type
    url='https://github.com/MattiaBarbiere/Deep_Learning_against_hate_speech',  # URL to your package repository
    packages=find_packages(),       # Automatically find all packages in the directory
    install_requires=requirements,
)
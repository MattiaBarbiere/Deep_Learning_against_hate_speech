from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name='DL_vs_HateSpeech',
    version='0.1',
    author='Group 19',
    author_email='your_email@example.com',
    description='Deep learning project against hate speech',
    long_description="Deep learning project aimed at detecting and mitigating hate speech in online platforms. This project utilizes advanced neural network architectures to analyze text data and classify it based on the presence of hate speech.",
    long_description_content_type='text/markdown',
    url='https://github.com/MattiaBarbiere/Deep_Learning_against_hate_speech',
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
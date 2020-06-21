import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="setka",
    version="1.1",
    author="Mikhail Romanov",
    author_email="romanov.michael.v@gmail.com",
    description="A set of scripts for fast Neural Network training",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/RomanovMikeV/setka",
    packages=setuptools.find_packages(),
    install_requires=[
        'torch',
        'torchvision',
        'numpy',
        'future',
        'tensorboard>=1.14',
        'tqdm',
        'matplotlib',
        'scikit-image',
        'termcolor',
        'pandas',
        'efficientnet_pytorch',
        'imgaug',
        'pytest',
        'pytest-cov',
        'jsonlines'
    ],
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    )
)

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pytorch-scorch",
    version="0.0.9",
    author="Mikhail Romanov",
    author_email="romanov.michael.v@gmail.com",
    description="A set of scripts for fast Neural Network prototyping",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/RomanovMikeV/scorch",
    packages=setuptools.find_packages(),
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
    entry_points = {
        'console_scripts': [
            'scorch-train = scorch.bash:training',
            'scorch-test = scorch.bash:testing'
        ]
    }
)

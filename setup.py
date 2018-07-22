import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pytorch-scorch",
    version="0.0.1",
    author="Mikhail Romanov",
    author_email="romanov.michael.v@gmail.com",
    description="A set of scripts to make the Neural Network training with pytorch faster",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/RomanovMikeV/scorch",
    packages=setuptools.find_packages(),
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
    scripts = [],
    entry_points = {
        "scorch-train = scorch.train:train"
    }
)

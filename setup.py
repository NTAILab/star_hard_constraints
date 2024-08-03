from setuptools import setup, find_packages


with open("README.md", "r", encoding="utf8") as fh:
    long_description = fh.read()

with open("requirements.txt") as f:
    install_requirements = f.read().splitlines()

setup(
    name="star_hard_constraints",
    version="1.0.0",
    author="Polytech NTAILab",
    description="Neural network layers with star-shape constrained output.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=install_requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"],
    license='MIT',
    python_requires='>=3.10',
)

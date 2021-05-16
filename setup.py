import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="kmembert",
    version="0.0.0",
    author="CentraleSupelec x Gustave Roussy",
    description="Estimation of cancer patients survival time based on french medical reports using Transformers",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    python_requires=">=3.7",
)
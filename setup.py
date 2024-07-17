import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="cvxpy-benchmarks",
    version="0.0.1",
    description="Code and data related to CVXPY benchmarks.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cvxpy/benchmarks",
    project_urls={
        "Bug Tracker": "https://github.com/cvxpy/benchmarks/issues",
    },
    package_dir={"": "benchmark"},
    packages=setuptools.find_packages(where="benchmark"),
    python_requires=">=3.8",
    license='Apache License, Version 2.0',
    install_requires=[
        "cvxpy",
        "asv<0.6",
        "virtualenv",
    ],
)

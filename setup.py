from setuptools import find_packages, setup

setup(
    name="syssimpyplots",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    package_data={"syssimpyplots/data": ["*.csv", "*.txt"]}, # doesn't work though unless we include them in a MANIFEST.in file
)

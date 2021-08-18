from setuptools import setup, find_packages


PACKAGENAME = "dpbench_datagen"
VERSION = "0.0.dev"


setup(
    name=PACKAGENAME,
    version=VERSION,
    description="Data generation package for dpbench applications.",
    long_description="Data generation package for dpbench applications.",
    #install_requires=["numpy", "scikit-learn"],
    packages=find_packages()
)

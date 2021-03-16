from setuptools import find_packages, setup

setup(
    name="likelihood",
    version="0.0",
    description="all relevant likelihood code",
    zip_safe=True,
    packages=find_packages(),
    python_requires=">=3.5",
    install_requires=["cobaya>=3.0"],
    package_data={"likelihood": ["yxgxk_like.py",
                                 "ccl.py"]},
)

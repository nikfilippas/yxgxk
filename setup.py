from setuptools import find_packages, setup
setup(
    name="yxgxk_like",
    version="0.0",
    description="gyk likelihood",
    zip_safe=True,
    packages=find_packages(),
    python_requires=">=3.5",
    install_requires=["cobaya>=3.0"],
    package_data={"yxgxk_like": ["yxgxk_like.py"]},
)

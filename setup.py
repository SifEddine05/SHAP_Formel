from setuptools import setup, find_packages

# Read dependencies from requirements.txt
with open("requirements.txt", "r") as f:
    requirements = f.read().splitlines()

setup(
    name='shap_formel_xai',
    version='0.1.0',
    description='SHAP + Formal explanations for Random Forest models',
    author='SELLAMI',
    packages=find_packages(),
    install_requires=requirements,
    python_requires='>=3.7'
)

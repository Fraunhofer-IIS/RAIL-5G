from setuptools import setup, find_packages

setup(
    name='Neural Positioning',
    version='0.0.1',
    #url='https://github.com/mypackage.git',
    author='Jonas Pirkl',
    description='Package that implements different neural positioning methods',
    packages=find_packages(),
    install_requires=[
        "torch",
        "lightning",
        "hydra-core",
        "matplotlib",
        "numpy",
        "h5py",
        "tensorboard",
        "scipy",
        "scikit-learn",
        "sammon-mapping",
        "umap-learn",
        "torchvision",
        "pandas"
    ]
)
from setuptools import setup, find_packages

setup(
    name="mde",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch>=1.9.0",
        "torchvision>=0.10.0",
        "numpy>=1.20.0",
        "pillow>=8.0.0",
        "matplotlib>=3.4.0",
        "tensorboard>=2.6.0",
        "tqdm>=4.62.0",
        "pyyaml>=5.4.0"
    ],
)

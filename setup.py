from setuptools import find_packages, setup


with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()


setup(
    name="m3fusiontrack",
    version="0.1.0",
    description="All-Weather Object Tracking with Multi-modal Multi-frequency "
                "Foundation Model and Adaptive Gated Fusion (reference "
                "implementation).",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="M3FusionTrack authors",
    license="MIT",
    packages=find_packages(exclude=["tools", "tools.*", "configs", "docs",
                                    "assets"]),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0",
        "torchvision>=0.15",
        "numpy>=1.22",
        "pillow>=9.0",
        "pyyaml>=6.0",
        "tqdm>=4.65",
        "matplotlib>=3.6",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
    ],
)

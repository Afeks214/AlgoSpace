#!/usr/bin/env python3
# AlgoSpace - AI-Powered Algorithmic Trading System

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="algospace",
    version="1.0.0",
    author="AlgoSpace Development Team",
    author_email="info@algospace.ai",
    description="AI-Powered Algorithmic Trading System with Multi-Agent Reinforcement Learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Afeks214/AlgoSpace",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Office/Business :: Financial :: Investment",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.7.1",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "structlog>=23.1.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.3.0",
    ],
    extras_require={
        "gpu": ["nvidia-ml-py3>=7.352.0"],
        "dev": ["pytest>=7.4.0", "pytest-cov>=4.1.0", "black>=23.7.0", "flake8>=6.0.0"],
        "docs": ["sphinx>=7.1.0", "sphinx-rtd-theme>=1.3.0"],
    },
    entry_points={
        "console_scripts": [
            "algospace=src.main:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)

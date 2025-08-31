"""Setup script for Jeopardy Benchmarking System."""

from setuptools import setup, find_packages
import os

# Read README for long description
readme_path = os.path.join(os.path.dirname(__file__), "README.md")
if os.path.exists(readme_path):
    with open(readme_path, "r", encoding="utf-8") as fh:
        long_description = fh.read()
else:
    long_description = "Language Model Benchmarking System using Jeopardy Questions"

# Read requirements
def read_requirements(filename):
    """Read requirements from file."""
    requirements_path = os.path.join(os.path.dirname(__file__), filename)
    if os.path.exists(requirements_path):
        with open(requirements_path, "r", encoding="utf-8") as fh:
            lines = fh.readlines()
        # Filter out comments and empty lines, and references to other files
        requirements = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith("#") and not line.startswith("-r"):
                requirements.append(line)
        return requirements
    return []

install_requires = read_requirements("requirements.txt")

setup(
    name="alex-trebench",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Language Model Benchmarking System using Jeopardy Questions",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/alex-trebench",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    entry_points={
        "console_scripts": [
            "alex=main:cli",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Testing",
    ],
    python_requires=">=3.8",
    install_requires=install_requires,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0", 
            "pytest-cov>=4.1.0",
            "pytest-mock>=3.11.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
            "pre-commit>=3.4.0",
        ],
        "docs": [
            "sphinx>=7.1.0",
            "mkdocs>=1.5.0", 
            "mkdocs-material>=9.2.0",
        ]
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json", "*.md"],
    },
    project_urls={
        "Bug Reports": "https://github.com/yourusername/alex-trebench/issues",
        "Source": "https://github.com/yourusername/alex-trebench",
        "Documentation": "https://alex-trebench.readthedocs.io/",
    },
)
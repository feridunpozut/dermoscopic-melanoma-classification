from setuptools import setup, find_packages

def parse_requirements(filename):
    with open(filename, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="dermoscopic-melanoma-classification",
    version="0.1",
    packages=find_packages(include=["src*"]),
    install_requires=parse_requirements("requirements.txt"),  # ✨
    include_package_data=True,
    description="Skin Lesion Classification and Grad-CAM Visualization",
    author="Feridun Pözüt",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)

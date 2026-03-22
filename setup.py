from setuptools import setup, find_packages

setup(
    name="lung_nodule",
    version="1.0.0",
    description="Lung nodule malignancy classification pipeline (LUNA25 Group 12)",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "monai>=1.3.0",
        "itk>=5.3.0",
        "nibabel>=5.0.0",
        "SimpleITK>=2.3.0",
        "numpy>=1.24.0,<2.0.0",
        "scipy>=1.10.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "timm>=0.9.0",
        "tqdm>=4.65.0",
    ],
)

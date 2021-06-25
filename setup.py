import setuptools


setuptools.setup(
    name="dataset-loader",
    author="Artem Yushkovskiy",
    author_email="ajuszkowski@ya.ru",
    packages=["dataset_loader"],  # setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=["opencv-python==4.2.0.34", "Pillow==7.2.0", "torch>1.2.0"],
)

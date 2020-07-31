import setuptools


setuptools.setup(
    name="giant-exercise",
    author="Artem Yushkovskiy",
    author_email="ajuszkowski@ya.ru",
    packages=["giant_exercise"],  # setuptools.find_packages(),
    # package_dir={'': 'src'},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=["opencv-python==4.2.0.34", "Pillow==7.2.0", "torch>1.2.0"],
)

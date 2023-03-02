import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name='schismatic',
    version='0.0.1',
    description='Library for pre- and post-processing of SCHISM simulations',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Adrian Croucher',
    author_email='a.croucher@auckland.ac.nz',
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent"],
    python_requires='>=2.7',
    install_requires=['numpy', 'scipy', 'pyproj', 'netCDF4', 'meshio',
                      'matplotlib', 'vtk']
)

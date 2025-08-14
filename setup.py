from setuptools import setup, find_packages 
 
def read_readme(): 
    with open("README.md", "r", encoding="utf-8") as fh: 
        return fh.read() 
 
setup( 
    name="hybrid-exponential-distributions", 
    version="1.0.0", 
    author="Ayeni Taiwo Michael", 
    author_email="ayenitaiwomichael24@gmail.com", 
    description="A comprehensive package for hybrid exponential distributions", 
    long_description=read_readme(), 
    long_description_content_type="text/markdown", 
    url="https://github.com/ayeni-T/hybrid-exponential-distributions", 
    packages=find_packages(), 
    classifiers=[ 
        "Development Status :: 4 - Beta", 
        "Intended Audience :: Science/Research", 
        "License :: OSI Approved :: MIT License", 
        "Programming Language :: Python :: 3", 
        "Topic :: Scientific/Engineering :: Mathematics", 
    ], 
    python_requires=">=3.8", 
    install_requires=[ 
        "numpy>=1.20.0", 
        "scipy>=1.7.0", 
        "matplotlib>=3.5.0", 
        "pandas>=1.3.0" 
    ] 
) 

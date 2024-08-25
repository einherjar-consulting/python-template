from setuptools import setup, find_packages

setup(
    name="python_template",
    version="1.0.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    author="Jarno Ralli",
    author_email="jarno.ralli@einherjar.fi",
    description="Template for a typical Python library",
    classifiers=[
        "Intended Audience :: Developers",
        "Topic :: Computer vision",
        "Programming Language :: Python :: 3.10",
    ],
    install_requires=["numpy", "opencv-python"],
)

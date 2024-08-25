# 1 Python Template

[![pre-commit](https://github.com/einherjar-consulting/python-template/actions/workflows/pre-commit.yml/badge.svg?branch=main&event=push)](https://github.com/einherjar-consulting/python-template/actions/workflows/pre-commit.yml)

This repository is a template for a typical deep-learning Python project that can also be distributed as a Python package. Common code is placed in the `src/library/` directory (rename `library` to match the name you want to give to the library). The common idea it to make the contents of the repository easy to understand for others and as easy as possible for the others to replicate the results. Avoid using absolute path names.

# 2 Contents

* [.flake8](./.flake8)
  * Flake8 configuration file
* [conda](./conda/README.md)
  * This directory contains conda environments used in the project
* [experiments](./experiments/README.md)
  * This directory contains experiments, development code etc. related to different deep-learning features
* [jupyter](./jupyter/README.md)
  * This directory contains example Jupyter notebooks
* [.pre-commit-config.yaml](./.pre-commit-config.yaml)
  * Pre-commit configuration file
* [src](./src/)
  * This directory contains the Python package that can be distributed
  * Rename `library` to match the name you want to give to the library and modify `setup.py` accordingly
* [setup.py](./setup.py)
  * Information of the Python package in the [src](./src/) directory

# 3 Usage

## 3.1 Committing Code to the Repository

Before committing any code to the repository, it needs to be verified/cleaned automatically using [pre-commit](https://pre-commit.com/).
`pre-commit` does the following:

* Cleans Jupyter notebooks. Only empty Jupyter notebooks should be pushed to the repo.
* Runs [black](https://github.com/psf/black) code formatter for all of the Python files.
* Runs [flake8](https://flake8.pycqa.org/en/latest/) code formatter for all of the Python files.

The `conda` environments in the [conda](./conda/README.md) have the `pre-commit` Python package installed. If you use any other
environment, you can install `pre-commit` using pip:

```bash
pip install pre-commit
```

Once `pre-commit` has been installed, execute the following in the `root` directory of this repo in order to install the `git` hooks:

```bash
pre-commit install
```

After having installed the `git`hooks, all the source files are automatically verified upon running `git commit -m`. You can verify
any time if the files are properly formatted, after having added the files to the stating area using `git add`, by running the following code:

```bash
pre-commit run --all-files
```

## 3.2 Python Code Style

All of the Python functions, classes etc. need to contain type-hinting for the input and return arguments, and contain Numpy-style docstring.
For more information regarding type-hinting in Python, take a look at [typing](https://docs.python.org/3/library/typing.html).
Following is an example of a function with proper type-hinting and a docstring.

```python
from typing import List, Tuple
import numpy as np

def calculate_mean_and_variance(data: List[float]) -> Tuple[float, float]:
    """
    Calculate the mean and variance of a list of numbers.

    Parameters
    ----------
    data : list of float
        A list of floating point numbers for which the mean and variance
        need to be calculated.

    Returns
    -------
    mean : float
        The mean of the input data.
    variance : float
        The variance of the input data.

    Raises
    ------
    ValueError
        If the input list is empty.

    Examples
    --------
    >>> calculate_mean_and_variance([1.0, 2.0, 3.0, 4.0])
    (2.5, 1.25)

    >>> calculate_mean_and_variance([10.0, 10.0, 10.0])
    (10.0, 0.0)
    """

    if len(data) == 0:
        raise ValueError("The input data list cannot be empty.")

    mean = np.mean(data)
    variance = np.var(data)

    return mean, variance
```

## 3.3 Building and Installing the Python Package

You can build and install the Python package found in the `src` directory. Following command builds the package into the `dist` directory.

```bash
python -m build .
```

After the package has been built, it can be installed using `pip install dist/<NAME-OF-THE-WHEEL-PACKAGE>`. Replace `NAME-OF-THE-WHEEL-PACKAGE` with the file name
of the actual wheel package that was built.

However, when you are developing functionality in the `src/library` modules, and you want to test the code immediately, instead of building and installing the package,
it's easier to install it in `editable` mode as follows:

```bash
pip install --editable .
```

In order to understand what installing a package in  `editable` mode means, take a look at [development_mode](https://setuptools.pypa.io/en/latest/userguide/development_mode.html).

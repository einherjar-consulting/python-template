from pathlib import Path


def get_project_root() -> Path:
    """Returns root-directory of the package.

    Returns
    -------
    Path
        Path to the root directory of the package.
    """

    return Path(__file__).parent.parent.parent

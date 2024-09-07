from torchvision.transforms import v2
from typing import Dict, Union
import ast
import torch
import numpy as np


def deserialize_compose_transformation(
    configuration: Dict, verbose: bool = False
) -> v2.Compose:
    """Deserializes a composed transformation from a dictionary. Supports also cases where the
    configuration has been read from a YAML file. YAML does not support tuples, these get converted
    into strings. For example (123, 123) becomes "(123, 123)".

    {"RandomResizedCrop":{"size" : (123, 123)}}
    {"RandomResizedCrop":{"size" : "123, 123"}}

    Parameters
    ----------
    configuration : Dict
        Configuration describing the composed transformation
    verbose : bool, optional
        In vebose mode parameters are shown to the user, by default False

    Returns
    -------
    v2.Compose
        Composed transformation

    Raises
    ------
    RuntimeError
        Raises an error if the configuration argument is not of type Dict
    RuntimeError
        Raises and error if a transformation cannot be constructed

    Examples
    --------
    >>> deserialize_compose_transformation({"RandomResizedCrop":{"size" : (123, 123)}})
    >>> deserialize_compose_transformation({"RandomResizedCrop":{"size" : "123, 123"}})
    """

    if not isinstance(configuration, dict):
        raise RuntimeError("configuration must be a dictionary")

    transforms = []

    for function in configuration:
        if verbose:
            print(f"Tranformation function: {function}")
            print(f"Original parameters: {configuration[function]}")

        # Reconstruct the parameters using Python types
        # Yaml does not support, for example, tuples. In the following:
        # size: (1, 2)
        # (1, 2) is interpreted as string, i.e. '(1, 2)' and passing that as a
        # parameter to the transformation function would not work. Therefore, we try
        # to reconstruct the parameters using Python types.
        reconstructed_parameters = {}
        for parameter in configuration[function]:
            parameter_value = literal_evaluation(configuration[function][parameter])
            reconstructed_parameters.update({parameter: parameter_value})

        if verbose:
            print(f"Reconstructed parameters: {reconstructed_parameters}")

        try:
            transform = getattr(v2, function)(**reconstructed_parameters)
        except Exception as e:
            raise RuntimeError(
                f"Failed to instantiate function '{function}' with parameters {reconstructed_parameters}: {e}"
            )

        transforms.append(transform)

    return v2.Compose(transforms)


def literal_evaluation(literal: Union[str, int, float]) -> any:
    """Evaluates the given literal and if required, tries to convert it to corresponding type.
    Note that this function can be potentially dangerous, since we use eval() to convert torch
    and numpy types!

    For example:
    - '(1, 2)' is converted into tuple (1, 2)
    - 'torch.float32' is converted into torch.float32
    - 'np.int16' is converted into np.int16
    - 'xyxy' stays the same, i.e. 'xyxy'

    Parameters
    ----------
    literal : Union[str, int, float]
        The object that is to be converted into proper type

    Returns
    -------
    any
        Given literal converted into the corresponding type

    Raises
    ------
    RuntimeError
        Failed to convert the literal into valid torch.dtype
    """

    if isinstance(literal, str):
        # Verify if we have torch dtype
        if literal.startswith("torch."):
            try:
                dtype_value = eval(literal)
                if isinstance(dtype_value, torch.dtype):
                    return dtype_value
            except Exception as e:
                raise RuntimeError(
                    f"Failed to convert '{literal}' into a valid torch.dtype: {e}"
                )
        elif literal.startswith("np."):
            try:
                dtype_value = eval(literal)
                # Check if the result is a NumPy data type (e.g., np.float32, np.int64, etc.)
                if isinstance(dtype_value, type) and issubclass(
                    dtype_value, np.generic
                ):
                    # Convert the data type to a NumPy dtype
                    return np.dtype(dtype_value)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to convert '{literal}' into a valid np.dtype: {e}"
                )
        else:
            try:
                python_type_value = ast.literal_eval(literal)
            except (ValueError, SyntaxError):
                python_type_value = literal
            return python_type_value

    return literal

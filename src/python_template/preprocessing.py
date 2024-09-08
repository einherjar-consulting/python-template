from torchvision.transforms import v2
from typing import Dict, Union, Tuple
import ast
import torch
import numpy as np


def deserialize_compose_transformation(
    configuration: Dict, verbose: bool = False
) -> v2.Compose:
    """Deserializes a composed transformation from a given dictionary. Supports transformations from
    torchvision.transforms.v2 plus the following operations:
    - "From_numpy", calls torch.from_numpy and converts the input into torch.Tensor
    - "Unsqueeze", calls torch.unsqueeze function on the input tensor.
    - "Permute", calls torch.permute function on the input tensor.

    For supported transformations, take a look at https://pytorch.org/vision/main/transforms.html#v2-api-ref.

    Supports also cases where the configuration has been read from a YAML file. For example,
    YAML does not support tuples, these get converted into strings. For example (123, 123) becomes "(123, 123)".
    In the following examples:

    {"RandomResizedCrop":{"size" : (123, 123)}}
    {"RandomResizedCrop":{"size" : "123, 123"}}

    (123, 123) and "123, 123" are converted into a tuple.

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

        if configuration[function] is not None:
            reconstructed_parameters = {}
            for parameter in configuration[function]:
                parameter_value = literal_evaluation(configuration[function][parameter])
                reconstructed_parameters.update({parameter: parameter_value})

            if verbose:
                print(f"Reconstructed parameters: {reconstructed_parameters}")

            try:
                if function == "From_numpy":
                    transform = From_numpy(**reconstructed_parameters)
                elif function == "Unsqueeze":
                    transform = Unsqueeze(**reconstructed_parameters)
                elif function == "Permute":
                    transform = Permute(**reconstructed_parameters)
                else:
                    transform = getattr(v2, function)(**reconstructed_parameters)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to instantiate function '{function}' with parameters {reconstructed_parameters}: {e}"
                )
        else:
            try:
                if function == "From_numpy":
                    transform = From_numpy()
                elif function == "Unsqueeze":
                    transform = Unsqueeze()
                elif function == "Permute":
                    transform = Permute()
                else:
                    transform = getattr(v2, function)()
            except Exception as e:
                raise RuntimeError(f"Failed to instantiate function '{function}': {e}")

        transforms.append(transform)

    return v2.Compose(transforms)


class From_numpy:
    """A class that converts numpy ndarrays into torch.Tensor format"""

    def __init__(self, clone: bool = True):
        """Constructor.

        Parameters
        ----------
        clone : bool, optional
            Clone the converted tensor, by default True
        """
        self.clone = clone

    def __call__(self, ndarray: np.ndarray) -> torch.Tensor:
        """Converts the given ndarray into Torch tensor, by default makes a clone.

        Parameters
        ----------
        ndarray : np.ndarray
            NDarray that is converted into tensor

        Returns
        -------
        torch.Tensor
            Array converted into tensor
        """
        if self.clone:
            my_tensor: torch.Tensor = torch.from_numpy(ndarray).clone()
        else:
            my_tensor: torch.Tensor = torch.from_numpy(ndarray)

        return my_tensor

    def __repr__(self):
        return f"{self.__class__.__name__}(clone={self.clone})"


class Unsqueeze:
    def __init__(self, dim: int = 0):
        """Constructor.

        Parameters
        ----------
        dim : int, optional
            The dimension along which to unsqueeze, by default 0
        """
        self.dim = dim

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """Applies the unsqueeze operation to the given tensor.

        Parameters
        ----------
        tensor : torch.Tensor
            The input tensor to apply unsqueeze to.

        Returns
        -------
        torch.Tensor
            The tensor with an additional dimension.
        """

        return torch.unsqueeze(tensor, self.dim)

    def __repr__(self):
        return f"{self.__class__.__name__}(dim={self.dim})"


class Permute:
    """
    A transformation class to permute the dimensions of a tensor, which can
    be added to a torchvision.transforms.v2.Compose pipeline.

    Parameters
    ----------
    dims : tuple of int
        The desired ordering of dimensions after permutation.

    Examples
    --------
    >>> import torch
    >>> from torchvision import transforms
    >>> transform = transforms.v2.Compose([
    ...     transforms.v2.ToTensor(),
    ...     Permute((2, 0, 1))  # Permute from (H, W, C) -> (C, H, W)
    ... ])
    >>> example_tensor = torch.rand(224, 224, 3)
    >>> transformed_tensor = transform(example_tensor)
    >>> print(transformed_tensor.shape)
    torch.Size([3, 224, 224])
    """

    def __init__(self, dims: Tuple[int]):
        """
        Initializes the transformation with the desired dimension order.

        Parameters
        ----------
        dims : tuple of int
            The desired ordering of dimensions after permutation.
        """
        self.dims = dims

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Applies the permute transformation to the given tensor.

        Parameters
        ----------
        tensor : torch.Tensor
            The input tensor to be permuted.

        Returns
        -------
        torch.Tensor
            The tensor with permuted dimensions.
        """
        return tensor.permute(*self.dims)

    def __repr__(self):
        """
        Returns a string representation of the transform for easy debugging and printing.

        Returns
        -------
        str
            A string describing the transformation and the permutation dimensions.
        """
        return f"{self.__class__.__name__}(dims={self.dims})"


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

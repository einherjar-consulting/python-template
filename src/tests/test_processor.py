from python_template.preprocessing import deserialize_compose_transformation
import torch
import numpy as np


def test_deserialize_compose_transformation_resize():

    # Create a test tensor
    image = torch.ones((3, 50, 50))

    # deserialize composed transformation based on dictionaries
    processor1 = deserialize_compose_transformation(
        configuration={"Resize": {"size": (20, 30)}}
    )
    processor2 = deserialize_compose_transformation(
        configuration={"Resize": {"size": "(20, 30)"}}
    )
    processor3 = deserialize_compose_transformation(
        configuration={"Resize": {"size": "20, 30"}}
    )

    result1 = processor1(image)
    result2 = processor2(image)
    result3 = processor3(image)

    assert result1.shape == (3, 20, 30)
    assert result2.shape == (3, 20, 30)
    assert result3.shape == (3, 20, 30)


def test_deserialize_compose_transformation_dtype():

    # Create a test tensor
    image = torch.ones((3, 50, 50), dtype=torch.float32)

    assert image.dtype == torch.float32

    processor = deserialize_compose_transformation(
        configuration={"ToDtype": {"dtype": "torch.float16"}}
    )

    result1 = processor(image)

    assert result1.dtype == torch.float16


def test_deserialize_compose_transformation_from_numpy():

    image = np.ones((20, 30), dtype=np.float32)

    # This version should make a clone of the numpy array, so memory
    # addresses should be different
    processor = deserialize_compose_transformation(configuration={"From_numpy": ""})

    image_tensor = processor(image)

    numpy_memory_address = image.ctypes.data
    torch_memory_address = image_tensor.data_ptr()

    assert isinstance(image_tensor, torch.Tensor)
    assert image_tensor.dtype == torch.float32
    assert numpy_memory_address != torch_memory_address

    # This version should not make a clone of the numpy array, so memory
    # addresses should be the same
    processor = deserialize_compose_transformation(
        configuration={"From_numpy": {"clone": False}}
    )

    image_tensor = processor(image)

    assert isinstance(image_tensor, torch.Tensor)
    assert image_tensor.dtype == torch.float32
    assert numpy_memory_address != torch_memory_address

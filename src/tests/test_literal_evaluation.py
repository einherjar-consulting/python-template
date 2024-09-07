from python_template.preprocessing import literal_evaluation
import torch
import numpy as np


# Test literal evaluation for torch.dtype
def test_literal_evalution_torch_dtype():

    dtype = literal_evaluation("torch.float16")
    assert dtype == torch.float16


# Test literal evaluation for torch.dtype
def test_literal_evalution_np_dtype():

    my_dtype = literal_evaluation("np.float32")
    assert my_dtype == np.float32


# Test literal evaluation for tuple
def test_literal_evalution_torch_tuple():

    my_tuple = literal_evaluation("(1, 2)")
    assert isinstance(my_tuple, tuple)

    my_tuple = literal_evaluation((1, 2))
    assert isinstance(my_tuple, tuple)


# Test literal evaluation for list
def test_literal_evalution_torch_list():

    my_list = literal_evaluation("[1,2,3]")
    assert isinstance(my_list, list)

    my_list = literal_evaluation([1, 2, 3])
    assert isinstance(my_list, list)


# Test literal evaluation for numeric values
def test_literal_evalution_torch_numeric():

    my_int = literal_evaluation(1)
    assert isinstance(my_int, int)

    my_int = literal_evaluation("1")
    assert isinstance(my_int, int)

    my_float = literal_evaluation(1.0)
    assert isinstance(my_float, float)

    my_float = literal_evaluation("1.0")
    assert isinstance(my_float, float)


# Test literal evaluation for string
def test_literal_evalution_torch_string():

    my_string = literal_evaluation("xyxy")
    assert isinstance(my_string, str)

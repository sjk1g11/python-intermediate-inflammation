"""Tests for statistics functions within the Model layer."""

import numpy as np
import numpy.testing as npt

from inflammation.models import daily_mean
from inflammation.models import patient_normalise

def test_daily_mean_zeros():
    """Test that mean function works for an array of zeros."""
    

    test_input = np.array([[0, 0],
                           [0, 0],
                           [0, 0]])
    test_result = np.array([0, 0])

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_mean(test_input), test_result)


def test_daily_mean_integers():
    """Test that mean function works for an array of positive integers."""

    test_input = np.array([[1, 2],
                           [3, 4],
                           [5, 6]])
    test_result = np.array([3, 4])

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_mean(test_input), test_result)

def test_sjk():
    """Manually check the means are right."""
    test_input = np.array([[1,2,3],[4,5,6],[7,8,9]])
    test_result = np.nansum(test_input, axis=0)/3
    npt.assert_array_equal(daily_mean(test_input), test_result)

def test_patient_normalise(test, expected):
    """Test min function works for array of 0s and +ve integers"""
    result = patient_normalise(np.array([test]))
    print(result)
    npt.assert_allclose(result, np.array([expected]), rtol=1e-2, atol=1e-2)#


def test_negative_inputs_patient_normalise(test, expected):
    """Test for type error"""
    test_data = [[-1,0,0], [0,0,0], [0,0,0]]
    with pytest.raises(ValueError):
        error_expected = patient_normalise(np.array(test_data))

import astropy.units as u
from fuzzyquantity.quantity import FuzzyQuantity
import numpy as np


# noinspection PyUnresolvedReferences
class TestFuzzyQuantityCreation:

    def test_has_expected_value_int(self):
        fuzz = FuzzyQuantity(5, 1)
        assert fuzz.value == 5

    def test_has_expected_value_float(self):
        fuzz = FuzzyQuantity(5.0, 1.0)
        assert fuzz.value == 5.0

    def test_has_expected_value_list(self):
        fuzz = FuzzyQuantity([5.0, 3.0], [2.0, 1.0])
        assert np.array_equal(fuzz.value, np.array([5.0, 3.0]))

    def test_has_expected_value_array(self):
        fuzz = FuzzyQuantity(np.array([5.0, 3.0]), np.array([2.0, 1.0]))
        assert np.array_equal(fuzz.value, np.array([5.0, 3.0]))

    def test_has_expected_uncertainty_int(self):
        fuzz = FuzzyQuantity(5, 1)
        assert fuzz.uncertainty == 1

    def test_has_expected_uncertainty_float(self):
        fuzz = FuzzyQuantity(5.0, 1.0)
        assert fuzz.uncertainty == 1.0

    def test_has_expected_uncertainty_list(self):
        fuzz = FuzzyQuantity([5.0, 3.0], [2.0, 1.0])
        assert np.array_equal(fuzz.uncertainty, np.array([2.0, 1.0]))

    def test_has_expected_uncertainty_array(self):
        fuzz = FuzzyQuantity(np.array([5.0, 3.0]), np.array([2.0, 1.0]))
        assert np.array_equal(fuzz.uncertainty, np.array([2.0, 1.0]))

    def test_has_unit_if_none_specified(self):
        fuzz = FuzzyQuantity(5, 1)
        assert fuzz.unit == u.dimensionless_unscaled

    def test_has_expected_unit_if_attached_to_value(self):
        fuzz = FuzzyQuantity(5*u.m, 1*u.m)
        assert fuzz.unit == u.m

    def test_has_expected_unit_if_kwarg(self):
        fuzz = FuzzyQuantity(5, 1, unit='m')
        assert fuzz.unit == u.m

    def test_has_expected_unit_if_convertible_but_different_inputs(self):
        fuzz = FuzzyQuantity(5*u.m, 1*u.cm)
        assert fuzz.unit == u.m

    def test_unit_kwarg_overrides_value_unit(self):
        fuzz = FuzzyQuantity(5*u.cm, 1*u.cm, unit='m')
        assert fuzz.unit == u.m

    def test_unit_kwarg_converts_value(self):
        fuzz = FuzzyQuantity(5*u.cm, 1*u.cm, unit='m')
        assert fuzz.value == 0.05


class TestFuzzyQuantityAddition:

    def test_sum_has_expected_value_int(self):
        fuzz1 = FuzzyQuantity(5, 2)
        fuzz2 = FuzzyQuantity(3, 1)
        fuzz3 = fuzz1 + fuzz2
        assert fuzz3.value == 8

    def test_sum_has_expected_value_float(self):
        fuzz1 = FuzzyQuantity(5.3, 2.4)
        fuzz2 = FuzzyQuantity(3.8, 1.1)
        fuzz3 = fuzz1 + fuzz2
        assert fuzz3.value == 5.3 + 3.8

    def test_sum_has_expected_value_list(self):
        fuzz1 = FuzzyQuantity([5.0, 3.0], [2.0, 1.0])
        fuzz2 = FuzzyQuantity([15.0, 13.0], [12.0, 11.0])
        fuzz3 = fuzz1 + fuzz2
        assert np.array_equal(fuzz3.value, np.array([20.0, 16.0]))

    def test_sum_has_expected_value_array(self):
        fuzz1 = FuzzyQuantity(np.array([5.0, 3.0]), np.array([2.0, 1.0]))
        fuzz2 = FuzzyQuantity(np.array([15.0, 13.0]), np.array([12.0, 11.0]))
        fuzz3 = fuzz1 + fuzz2
        assert np.array_equal(fuzz3.value, np.array([20.0, 16.0]))
        
    def test_sum_has_expected_uncertainty_int(self):
        fuzz1 = FuzzyQuantity(5, 2)
        fuzz2 = FuzzyQuantity(3, 1)
        fuzz3 = fuzz1 + fuzz2
        assert fuzz3.uncertainty == np.sqrt(2**2 + 1**2)

    def test_sum_has_expected_uncertainty_float(self):
        fuzz1 = FuzzyQuantity(5.3, 2.4)
        fuzz2 = FuzzyQuantity(3.8, 1.1)
        fuzz3 = fuzz1 + fuzz2
        assert fuzz3.uncertainty == np.sqrt(2.4**2 + 1.1**2)

    def test_sum_has_expected_uncertainty_list(self):
        value1 = [5.0, 3.0]
        uncertainty1 = [2.0, 1.0]
        value2 = [15.0, 13.0]
        uncertainty2 = [12.0, 11.0]
        fuzz1 = FuzzyQuantity(value1, uncertainty1)
        fuzz2 = FuzzyQuantity(value2, uncertainty2)
        fuzz3 = fuzz1 + fuzz2
        unc_expected = np.sqrt(np.array(uncertainty1)**2 +
                               np.array(uncertainty2)**2)
        assert np.array_equal(fuzz3.uncertainty, unc_expected)

    def test_sum_has_expected_uncertainty_array(self):
        value1 = np.array([5.0, 3.0])
        uncertainty1 = np.array([2.0, 1.0])
        value2 = np.array([15.0, 13.0])
        uncertainty2 = np.array([12.0, 11.0])
        fuzz1 = FuzzyQuantity(value1, uncertainty1)
        fuzz2 = FuzzyQuantity(value2, uncertainty2)
        fuzz3 = fuzz1 + fuzz2
        unc_expected = np.sqrt(uncertainty1**2 + uncertainty2**2)
        assert np.array_equal(fuzz3.uncertainty, unc_expected)
        
class TestFuzzyQuantitySubtraction:
    def test_difference_has_expected_value_int(self):
        fuzz1 = FuzzyQuantity(5, 2)
        fuzz2 = FuzzyQuantity(3, 1)
        fuzz3 = fuzz1 - fuzz2
        assert fuzz3.value == 2

    def test_difference_has_expected_value_float(self):
        fuzz1 = FuzzyQuantity(5.3, 2.4)
        fuzz2 = FuzzyQuantity(3.8, 1.1)
        fuzz3 = fuzz1 - fuzz2
        assert fuzz3.value == 5.3 - 3.8

    def test_difference_has_expected_value_list(self):
        fuzz1 = FuzzyQuantity([5.0, 3.0], [2.0, 1.0])
        fuzz2 = FuzzyQuantity([15.0, 13.0], [12.0, 11.0])
        fuzz3 = fuzz1 - fuzz2
        assert np.array_equal(fuzz3.value, np.array([-10.0, -10.0]))

    def test_difference_has_expected_value_array(self):
        fuzz1 = FuzzyQuantity(np.array([5.0, 3.0]), np.array([2.0, 1.0]))
        fuzz2 = FuzzyQuantity(np.array([15.0, 13.0]), np.array([12.0, 11.0]))
        fuzz3 = fuzz1 - fuzz2
        assert np.array_equal(fuzz3.value, np.array([-10.0, -10.0]))

    def test_difference_has_expected_uncertainty_int(self):
        fuzz1 = FuzzyQuantity(5, 2)
        fuzz2 = FuzzyQuantity(3, 1)
        fuzz3 = fuzz1 - fuzz2
        assert fuzz3.uncertainty == np.sqrt(2 ** 2 + 1 ** 2)

    def test_difference_has_expected_uncertainty_float(self):
        fuzz1 = FuzzyQuantity(5.3, 2.4)
        fuzz2 = FuzzyQuantity(3.8, 1.1)
        fuzz3 = fuzz1 - fuzz2
        assert fuzz3.uncertainty == np.sqrt(2.4 ** 2 + 1.1 ** 2)

    def test_difference_has_expected_uncertainty_list(self):
        value1 = [5.0, 3.0]
        uncertainty1 = [2.0, 1.0]
        value2 = [15.0, 13.0]
        uncertainty2 = [12.0, 11.0]
        fuzz1 = FuzzyQuantity(value1, uncertainty1)
        fuzz2 = FuzzyQuantity(value2, uncertainty2)
        fuzz3 = fuzz1 - fuzz2
        unc_expected = np.sqrt(np.array(uncertainty1) ** 2 +
                               np.array(uncertainty2) ** 2)
        assert np.array_equal(fuzz3.uncertainty, unc_expected)

    def test_difference_has_expected_uncertainty_array(self):
        value1 = np.array([5.0, 3.0])
        uncertainty1 = np.array([2.0, 1.0])
        value2 = np.array([15.0, 13.0])
        uncertainty2 = np.array([12.0, 11.0])
        fuzz1 = FuzzyQuantity(value1, uncertainty1)
        fuzz2 = FuzzyQuantity(value2, uncertainty2)
        fuzz3 = fuzz1 - fuzz2
        unc_expected = np.sqrt(uncertainty1 ** 2 + uncertainty2 ** 2)
        assert np.array_equal(fuzz3.uncertainty, unc_expected)
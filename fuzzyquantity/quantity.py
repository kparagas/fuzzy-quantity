from astropy.units import Quantity, dimensionless_unscaled
from astropy.units.typing import QuantityLike
import numpy as np
from fuzzyquantity.string_formatting import (_terminal_string,
                                             _make_siunitx_string,
                                             _make_oldschool_latex_string)

from fuzzyquantity.derivatives import propagate_1, propagate_2

class FuzzyQuantity(Quantity):
    """
    A subclass of Astropy's `Quantity` which includes uncertainties and handles
    standard error propagation.
    """
    def __new__(cls,
                value: QuantityLike,
                uncertainty: QuantityLike,
                unit=None,
                **kwargs):
        """
        Parameters
        ----------
        value : QuantityLike
            The quantity's measured value.
        uncertainty : QuantityLike=
            The quantity's measured uncertainty. Assumed to be 1-sigma
            Gaussian uncertainty.
        unit : unit-like, optional
            The units associated with the value. If you specify a unit here
            which is different than units attached to `value`, this will
            override the `value` units.
        kwargs
            Additional keyword arguments are passed to `Quantity`.
        """
        obj = super().__new__(cls, value=value, unit=unit, **kwargs)
        if isinstance(uncertainty, Quantity):
            obj.uncertainty = Quantity(uncertainty).to(obj.unit).value
        else:
            obj.uncertainty = uncertainty
        return obj

    def __str__(self) -> str:
        return _terminal_string(self.value, self.uncertainty, self.unit)

    def _prop_err_add_sub(self, uncertainty, unit):
        return np.sqrt((self.uncertainty * self.unit)**2 +
                       (uncertainty * unit).to(self.unit)**2)

    def _prop_err_mul_truediv(self, out_value, value, uncertainty):
        frac_unc = np.sqrt((self.uncertainty / self.value)**2 +
                           (uncertainty / value)**2)
        return np.abs(out_value) * frac_unc

    @staticmethod
    def _parse_input(other):
        if isinstance(other, FuzzyQuantity):
            value = other.value
            uncertainty = other.uncertainty
            unit = other.unit
        elif isinstance(other, Quantity):
            value = other.value
            uncertainty = 0
            unit = other.unit
        else:
            value = other
            uncertainty = 0
            unit = dimensionless_unscaled
        return value, uncertainty, unit

    def __array_function__(self, func, types, args, kwargs):
        """Wrap numpy functions.

        Parameters
        ----------
        func: callable
            Arbitrary callable exposed by NumPy’s public API.
        types: list
            Collection of unique argument types from the original NumPy
            function call that implement ``__array_function__``.
        args: tuple
            Positional arguments directly passed on from the original call.
        kwargs: dict
            Keyword arguments directly passed on from the original call.
        """
        if func not in HANDLED_AFUNCS:
            return NotImplemented

        return HANDLED_AFUNCS[func](*args, **kwargs)

    def __add__(self, other):
        value, uncertainty, unit = self._parse_input(other)
        out_value = self.value * self.unit + value * unit
        out_uncertainty = propagate_2('add', out_value, self.value, value, self.uncertainty, uncertainty)
        return FuzzyQuantity(value=out_value, uncertainty=out_uncertainty)

    __radd__ = __add__

    def __sub__(self, other):
        value, uncertainty, unit = self._parse_input(other)
        out_value = self.value * self.unit - value * unit
        out_uncertainty = propagate_2('sub', out_value, self.value, value, self.uncertainty, uncertainty)
        return FuzzyQuantity(value=out_value, uncertainty=out_uncertainty)

    __rsub__ = __sub__

    def __mul__(self, other):
        value, uncertainty, unit = self._parse_input(other)
        out_value = self.value * self.unit * value * unit
        out_uncertainty = propagate_2('mul', out_value, self.value, value, self.uncertainty, uncertainty)
        return FuzzyQuantity(value=out_value, uncertainty=out_uncertainty)

    __rmul__ = __mul__

    def __truediv__(self, other):
        value, uncertainty, unit = self._parse_input(other)
        out_value = self.value * self.unit / (value * unit)
        out_uncertainty = propagate_2('truediv', out_value, self.value, value, self.uncertainty, uncertainty)
        return FuzzyQuantity(value=out_value, uncertainty=out_uncertainty)

    __rtruediv__ = __truediv__

    def __pow__(self, other):
        value, uncertainty, unit = self._parse_input(other)
        if unit != dimensionless_unscaled:
            raise ValueError('u r dumb. exponent must be unitless.')
        out_value = (self.value * self.unit) ** (value)
        out_uncertainty = propagate_2('pow', out_value, self.value, value, self.uncertainty, uncertainty)
        return FuzzyQuantity(value=out_value, uncertainty=out_uncertainty)

    # TODO: implement list/array version as indicated in the docstring
    def latex(self,
              sci_thresh: int = 3,
              siunitx: bool = True) -> str:
        r"""
        Generate a LaTeX string of the FuzzyQuantity. Assumes the use of the
        `siunitx` package by default, which you should really be using if you
        are writing numbers with units and uncertainty.

        If the FuzzyQuantity is a list or array of values, this will output a
        similarly-shaped list or array of LaTeX strings.

        Parameters
        ----------
        sci_thresh : int
            The threshold for returning output in scientific notation. The
            default is 3, so any value equal to or larger than 1000 or equal
            to or smaller than 1/1000 will be returned in scientific notation
            form.

            For example, 999 ± 10 will return as `'999 ± 10'` but 1000 ± 10
            will return as `'(1.00 ± 0.01)e+03'`.
        siunitx : bool
            If `True`, return the string in either `\num{}` or `\SI{}{}` format
            for automatic parsing by the `siunitx` package. If `False`, return
            an old-school manual form. Does not include `$`, so you'll have to
            add those yourself!

        Returns
        -------
        str
            The LaTeX-formatted string.
        """
        if siunitx:
            return _make_siunitx_string(
                self.value, self.uncertainty.value, self.unit, sci_thresh)
        else:
            return _make_oldschool_latex_string(
                self.value, self.uncertainty.value, self.unit, sci_thresh)

HANDLED_AFUNCS = {}
HANDLED_UFUNCS = {}  # must be func(method, *inputs, **kwargs)


def _implements_array_func(numpy_function):
    """Register an __array_function__ implementation for QFloat objects."""
    def decorator_array_func(func):
        HANDLED_AFUNCS[numpy_function] = func
        return func
    return decorator_array_func


def _implements_ufunc(numpy_ufunc):
    """Register an ufunc implementation for QFloat objects."""
    def decorator_ufunc(func):
        HANDLED_UFUNCS[numpy_ufunc] = func
        return func
    return decorator_ufunc


@_implements_array_func(np.shape)
def _np_shape(fuzzy_quantity: FuzzyQuantity) -> tuple[int, ...]:
    """Implement np.shape for FuzzyQuantity objects."""
    return fuzzy_quantity.shape


@_implements_array_func(np.size)
def _np_size(fuzzy_quantity: FuzzyQuantity) -> int:
    return fuzzy_quantity.size


@_implements_array_func(np.clip)
def _np_clip(fuzzy_quantity: FuzzyQuantity, a_min, a_max, *args, **kwargs) -> FuzzyQuantity:
    value = np.clip(fuzzy_quantity.value, a_min, a_max, *args, **kwargs)
    return FuzzyQuantity(value, fuzzy_quantity.uncertainty, fuzzy_quantity.unit)


def _array_func_simple_wrapper(numpy_func):
    """Wraps simple array functions.

    Notes
    -----
    - Functions elegible for these are that ones who applies for nominal and
      std_dev values and return a new QFloat with the applied values.
    - No conversion or special treatment is done in this wrapper.
    - Only for one array ate once.
    """
    def wrapper(fuzzy_quantity, *args, **kwargs):
        value = numpy_func(fuzzy_quantity.value, *args, **kwargs)
        uncertainty = numpy_func(fuzzy_quantity.uncertainty, *args, **kwargs)
        return FuzzyQuantity(value, uncertainty, fuzzy_quantity.unit)
    _implements_array_func(numpy_func)(wrapper)


_array_func_simple_wrapper(np.delete)
_array_func_simple_wrapper(np.expand_dims)
_array_func_simple_wrapper(np.flip)
_array_func_simple_wrapper(np.fliplr)
_array_func_simple_wrapper(np.flipud)
_array_func_simple_wrapper(np.moveaxis)
_array_func_simple_wrapper(np.ravel)
_array_func_simple_wrapper(np.repeat)
_array_func_simple_wrapper(np.reshape)
_array_func_simple_wrapper(np.resize)
_array_func_simple_wrapper(np.roll)
_array_func_simple_wrapper(np.rollaxis)
_array_func_simple_wrapper(np.rot90)
_array_func_simple_wrapper(np.squeeze)
_array_func_simple_wrapper(np.swapaxes)
_array_func_simple_wrapper(np.take)
_array_func_simple_wrapper(np.tile)
_array_func_simple_wrapper(np.transpose)


@_implements_array_func(np.round)
@_implements_array_func(np.around)
def _np_round(fuzzy_quantity: FuzzyQuantity, *args, **kwargs) -> FuzzyQuantity:
    """Implement np.round for FuzzyQuantity objects."""
    value = np.round(fuzzy_quantity.value, *args, **kwargs)
    uncertainty = np.round(fuzzy_quantity.uncertainty, *args, **kwargs)
    return FuzzyQuantity(value, uncertainty, fuzzy_quantity.unit)

if __name__ == '__main__':
    fuzz = FuzzyQuantity(7, 2, unit = 'Jy')

    print(np.clip(fuzz, 8.5, 9.1))
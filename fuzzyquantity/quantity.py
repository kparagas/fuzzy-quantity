from functools import partial

import numpy as np
import astropy.units as u
from astropy.units.typing import QuantityLike

from fuzzyquantity.derivatives import propagate_1, propagate_2
from fuzzyquantity.exceptions import UnitsError
from fuzzyquantity.string_formatting import (_terminal_string,
                                             _make_siunitx_string,
                                             _make_oldschool_latex_string)

class FuzzyQuantity(u.Quantity):
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
            override the `value` 
        kwargs
            Additional keyword arguments are passed to `Quantity`.
        """
        obj = super().__new__(cls, value=value, unit=unit, **kwargs)
        if isinstance(uncertainty, u.Quantity):
            obj.uncertainty = u.Quantity(uncertainty).to(obj.unit).value
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
        elif isinstance(other, u.Quantity):
            value = other.value
            uncertainty = 0
            unit = other.unit
        else:
            value = other
            uncertainty = 0
            unit = u.dimensionless_unscaled
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
        if unit != u.dimensionless_unscaled:
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
    - Functions elegible for these are that ones who applies for value and
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

@_implements_array_func(np.append)
def _np_append(fuzzy_quantity1: FuzzyQuantity, fuzzy_quantity2: FuzzyQuantity, *args, **kwargs) -> FuzzyQuantity:
    """Implement np.append for FuzzyQuantity objects."""
    # First, convert to the same unit.
    fuzzy_quantity2 = fuzzy_quantity2.to(fuzzy_quantity1.unit)
    value = np.append(fuzzy_quantity1.value, fuzzy_quantity2.value, *args, **kwargs)
    uncertainty = np.append(fuzzy_quantity1.uncertainty, fuzzy_quantity2.uncertainty, *args, **kwargs)
    return FuzzyQuantity(value, uncertainty, fuzzy_quantity1.unit)


@_implements_array_func(np.insert)
def _np_insert(fuzzy_quantity1: FuzzyQuantity, fuzzy_quantity2: FuzzyQuantity, index: int, *args, **kwargs) -> FuzzyQuantity:
    """Implement np.insert for FuzzyQuantity objects."""
    # Same unit needed too
    fuzzy_quantity2 = fuzzy_quantity2.to(fuzzy_quantity1.unit)
    value = np.insert(fuzzy_quantity1.value, index, fuzzy_quantity2.value, *args, **kwargs)
    uncertainty = np.insert(fuzzy_quantity1.uncertainty, index, fuzzy_quantity2.uncertainty, *args, **kwargs)
    return FuzzyQuantity(value, uncertainty, fuzzy_quantity1.unit)


@_implements_array_func(np.sum)
def _np_sum(fuzzy_quantity: FuzzyQuantity, *args, **kwargs) -> FuzzyQuantity:
    """Implement np.sum for FuzzyQuantity objects."""
    value = np.sum(fuzzy_quantity.value, *args, **kwargs)
    uncertainty = np.sqrt(np.sum(np.square(fuzzy_quantity.uncertainty), *args, **kwargs))
    return FuzzyQuantity(value, uncertainty, fuzzy_quantity.unit)


@_implements_array_func(np.mean)
def _np_mean(fuzzy_quantity: FuzzyQuantity, *args, **kwargs) -> FuzzyQuantity:
    """Implement np.mean for FuzzyQuantity objects."""
    value = np.mean(fuzzy_quantity.value, *args, **kwargs)
    uncertainty = np.sqrt(np.sum(np.square(fuzzy_quantity.uncertainty), *args, **kwargs)) / len(value)
    return FuzzyQuantity(value, uncertainty, fuzzy_quantity.unit)


@_implements_array_func(np.nanmean)
def _np_nanmean(fuzzy_quantity: FuzzyQuantity, *args, **kwargs) -> FuzzyQuantity:
    """Implement np.mean for FuzzyQuantity objects."""
    value = np.nanmean(fuzzy_quantity.value, *args, **kwargs)
    uncertainty = np.sqrt(np.sum(np.square(fuzzy_quantity.uncertainty), *args, **kwargs)) / len(value)
    return FuzzyQuantity(value, uncertainty, fuzzy_quantity.unit)

#TODO: need to figure out uncertainty for median

# @_implements_array_func(np.median)
# def _np_median(fuzzy_quantity: FuzzyQuantity, *args, **kwargs) -> FuzzyQuantity:
#     """Implement np.median for FuzzyQuantity objects."""
#     value = np.median(fuzzy_quantity.value, *args, **kwargs)
#     uncertainty = np.median(np.abs(fuzzy_quantity.value - value), *args, **kwargs)
#     return FuzzyQuantity(value, uncertainty, fuzzy_quantity.unit)
#
#
# @_implements_array_func(np.nanmedian)
# def _np_nanmedian(fuzzy_quantity: FuzzyQuantity, *args, axis=None, **kwargs) -> FuzzyQuantity:
#     """Implement np.median for FuzzyQuantity objects."""
#     value = np.nanmedian(fuzzy_quantity.value, *args, axis=axis, **kwargs)
#     # error of average = std_dev/sqrt(N)
#     std = np.nanstd(fuzzy_quantity.value, axis=axis)
#     # N is determined by the number of elements in the axis
#     uncertainty = std / np.sqrt(np.nansum(fuzzy_quantity.value, axis=axis)/value)
#     return FuzzyQuantity(value, uncertainty, fuzzy_quantity.unit)


@_implements_array_func(np.std)
def _np_std(fuzzy_quantity: FuzzyQuantity, *args, **kwargs) -> u.Quantity:
    """Implement np.std for FuzzyQuantity objects."""
    return np.std(fuzzy_quantity.value, *args, **kwargs) * fuzzy_quantity.unit


@_implements_array_func(np.nanstd)
def _np_nanstd(fuzzy_quantity: FuzzyQuantity, *args, **kwargs) -> u.Quantity:
    """Implement np.std for FuzzyQuantity objects."""
    return np.nanstd(fuzzy_quantity.value, *args, **kwargs) * fuzzy_quantity.unit


@_implements_array_func(np.var)
def _np_var(fuzzy_quantity: FuzzyQuantity, *args, **kwargs) -> u.Quantity:
    """Implement np.var for FuzzyQuantity objects."""
    return np.var(fuzzy_quantity.value, *args, **kwargs) * fuzzy_quantity.unit


@_implements_array_func(np.nanvar)
def _np_nanvar(fuzzy_quantity: FuzzyQuantity, *args, **kwargs) -> u.Quantity:
    """Implement np.var for FuzzyQuantity objects."""
    return np.nanvar(fuzzy_quantity.value, *args, **kwargs) * fuzzy_quantity.unit

################################################################

def _implements_ufunc_on_value(func):
    """Wraps ufuncs only on the value and don't return FuzzyQuantity object."""
    def wrapper(fuzzy_quantity, *args, **kwargs):
        return func(fuzzy_quantity.value, *args, **kwargs)
    _implements_ufunc(func)(wrapper)


_implements_ufunc_on_value(np.isnan)
_implements_ufunc_on_value(np.isinf)
_implements_ufunc_on_value(np.isfinite)
_implements_ufunc_on_value(np.isneginf)
_implements_ufunc_on_value(np.isposinf)
_implements_ufunc_on_value(np.isreal)
_implements_ufunc_on_value(np.iscomplex)
_implements_ufunc_on_value(np.isscalar)
_implements_ufunc_on_value(np.signbit)
_implements_ufunc_on_value(np.sign)


def _np_exp_log(fuzzy_quantity, func):
    """General implementation for exp and log functions."""
    if fuzzy_quantity.unit != u.dimensionless_unscaled:
        raise UnitsError(f'{func.__name__} is only defined for dimensionless quantities.')
    value = func(fuzzy_quantity.value)
    uncertainty = propagate_1(func.__name__, value, fuzzy_quantity.value, fuzzy_quantity.uncertainty)
    return FuzzyQuantity(value, uncertainty, u.dimensionless_unscaled)


_implements_ufunc(np.exp)(partial(_np_exp_log, func=np.exp))
_implements_ufunc(np.exp2)(partial(_np_exp_log, func=np.exp2))
_implements_ufunc(np.expm1)(partial(_np_exp_log, func=np.expm1))
_implements_ufunc(np.log)(partial(_np_exp_log, func=np.log))
_implements_ufunc(np.log2)(partial(_np_exp_log, func=np.log2))
_implements_ufunc(np.log10)(partial(_np_exp_log, func=np.log10))
_implements_ufunc(np.log1p)(partial(_np_exp_log, func=np.log1p))


def _np_floor_wrapper(fuzzy_quantity, func):
    """General implementation for floor, ceil and trunc."""
    return FuzzyQuantity(func(fuzzy_quantity.value), np.round(fuzzy_quantity.uncertainty, 0), fuzzy_quantity.unit)


_implements_ufunc(np.floor)(partial(_np_floor_wrapper, func=np.floor))
_implements_ufunc(np.ceil)(partial(_np_floor_wrapper, func=np.ceil))
_implements_ufunc(np.trunc)(partial(_np_floor_wrapper, func=np.trunc))


def _np_only_value_wrapper(fuzzy_quantity, func):
    """General implementation for isfinite, isinf, isnan."""
    return func(fuzzy_quantity.value)


_implements_ufunc(np.isfinite)(partial(_np_only_value_wrapper,
                                       func=np.isfinite))
_implements_ufunc(np.isinf)(partial(_np_only_value_wrapper,
                                    func=np.isinf))
_implements_ufunc(np.isnan)(partial(_np_only_value_wrapper,
                                    func=np.isnan))


@_implements_ufunc(np.radians)
@_implements_ufunc(np.deg2rad)
def _np_radians(fuzzy_quantity, *args, **kwargs):
    """Convert any qfloat angle to radian."""
    return fuzzy_quantity.to(u.radian)


@_implements_ufunc(np.degrees)
@_implements_ufunc(np.rad2deg)
def _np_degrees(fuzzy_quantity, *args, **kwargs):
    return fuzzy_quantity.to(u.degree)


def _trigonometric_simple_wrapper(numpy_ufunc):
    def trig_wrapper(fuzzy_quantity, *args, **kwargs):
        # check if qf is angle
        if fuzzy_quantity.unit not in (u.degree, u.radian):
            raise UnitsError('qfloat unit is not degree or radian.')

        # if degree, convert to radian as required for numpy inputs.
        if fuzzy_quantity.unit == u.degree:
            qf = fuzzy_quantity.to(u.radian)

        value = numpy_ufunc(fuzzy_quantity.value)
        std = propagate_1(numpy_ufunc.__name__, value,
                          fuzzy_quantity.value, fuzzy_quantity.std_dev)
        return FuzzyQuantity(value, std, u.dimensionless_unscaled)
    _implements_ufunc(numpy_ufunc)(trig_wrapper)


def _inverse_trigonometric_simple_wrapper(numpy_ufunc):
    def inv_wrapper(fuzzy_quantity, *args, **kwargs):
        if fuzzy_quantity.unit != u.dimensionless_unscaled:
            raise UnitsError('inverse trigonometric functions require '
                             'dimensionless unscaled variables.')

        value = numpy_ufunc(fuzzy_quantity.value)
        std = propagate_1(numpy_ufunc.__name__, value,
                          fuzzy_quantity.value, fuzzy_quantity.std_dev)

        return FuzzyQuantity(value, std, u.radian)
    _implements_ufunc(numpy_ufunc)(inv_wrapper)


_trigonometric_simple_wrapper(np.sin)
_trigonometric_simple_wrapper(np.cos)
_trigonometric_simple_wrapper(np.tan)
_trigonometric_simple_wrapper(np.sinh)
_trigonometric_simple_wrapper(np.cosh)
_trigonometric_simple_wrapper(np.tanh)
_inverse_trigonometric_simple_wrapper(np.arcsin)
_inverse_trigonometric_simple_wrapper(np.arccos)
_inverse_trigonometric_simple_wrapper(np.arctan)
_inverse_trigonometric_simple_wrapper(np.arcsinh)
_inverse_trigonometric_simple_wrapper(np.arccosh)
_inverse_trigonometric_simple_wrapper(np.arctanh)


@_implements_ufunc(np.arctan2)
def _np_arctan2(qf1, qf2):
    """Compute the arctangent of qf1/qf2."""
    # The 2 values must be in the same unit.
    qf2 = qf2.to(qf1.unit)
    value = np.arctan2(qf1.value, qf2.value)
    std = propagate_2('arctan2', value, qf1.value, qf2.value,
                      qf1.std_dev, qf2.std_dev)
    return FuzzyQuantity(value, std, u.radian)


_ufunc_translate = {
    'add': FuzzyQuantity.__add__,
    'absolute': FuzzyQuantity.__abs__,
    'divide': FuzzyQuantity.__truediv__,
    'float_power': FuzzyQuantity.__pow__,
    'floor_divide': FuzzyQuantity.__floordiv__,
    'multiply': FuzzyQuantity.__mul__,
    'negative': FuzzyQuantity.__neg__,
    'positive': FuzzyQuantity.__pos__,
    'power': FuzzyQuantity.__pow__,
    'mod': FuzzyQuantity.__mod__,
    'remainder': FuzzyQuantity.__mod__,
    'subtract': FuzzyQuantity.__sub__,
    'true_divide': FuzzyQuantity.__truediv__,
    'divmod': lambda x, y: (FuzzyQuantity.__floordiv__(x, y), FuzzyQuantity.__mod__(x, y)),
}


def _general_ufunc_wrapper(numpy_ufunc):
    """Implement ufuncs for general math operations.

    Notes
    -----
    - These functions will not operate with kwarg.
    - These functions will just wrap QFloat math methods.
    """
    ufunc_name = numpy_ufunc.__name__
    true_func = _ufunc_translate[ufunc_name]

    def ufunc_wrapper(*inputs):
        return true_func(*inputs)
    _implements_ufunc(numpy_ufunc)(ufunc_wrapper)


_general_ufunc_wrapper(np.add)
_general_ufunc_wrapper(np.absolute)
_general_ufunc_wrapper(np.divide)
_general_ufunc_wrapper(np.divmod)
_general_ufunc_wrapper(np.float_power)
_general_ufunc_wrapper(np.floor_divide)
_general_ufunc_wrapper(np.mod)
_general_ufunc_wrapper(np.multiply)
_general_ufunc_wrapper(np.negative)
_general_ufunc_wrapper(np.positive)
_general_ufunc_wrapper(np.power)
_general_ufunc_wrapper(np.remainder)
_general_ufunc_wrapper(np.subtract)
_general_ufunc_wrapper(np.true_divide)


@_implements_ufunc(np.copysign)
def _np_copysign(qf1, qf2):
    """Return the first argument with the sign of the second argument."""
    value = np.copysign(qf1.value, qf2.value)
    std = propagate_2('copysign', value, qf1.value, qf2.value,
                      qf1.std_dev, qf2.std_dev)
    return FuzzyQuantity(value, std, qf1.unit)


@_implements_ufunc(np.square)
def _np_square(qf):
    return qf * qf


@_implements_ufunc(np.sqrt)
def _np_sqrt(qf):
    return qf**0.5


@_implements_ufunc(np.hypot)
def _np_hypot(qf1, qf2):
    qf2 = qf2.to(qf1.unit)
    value = np.hypot(qf1.value, qf2.value)
    std = propagate_2('hypot', value, qf1.value, qf2.value,
                      qf1.std_dev, qf2.std_dev)
    return FuzzyQuantity(value, std, qf1.unit)

##########################

if __name__ == '__main__':
    fuzz = FuzzyQuantity(7, 2, unit = 'Jy')

    print(np.clip(fuzz, 8.5, 9.1))
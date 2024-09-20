from astropy.units import Quantity, dimensionless_unscaled
from astropy.units.typing import QuantityLike
import numpy as np
from fuzzyquantity.string_formatting import (_terminal_string,
                                             _make_siunitx_string,
                                             _make_oldschool_latex_string)


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

    def __add__(self, other):
        value, uncertainty, unit = self._parse_input(other)
        out_value = self.value * self.unit + value * unit
        out_uncertainty = self._prop_err_add_sub(uncertainty, unit)
        return FuzzyQuantity(value=out_value, uncertainty=out_uncertainty)

    __radd__ = __add__

    def __sub__(self, other):
        value, uncertainty, unit = self._parse_input(other)
        out_value = self.value * self.unit - value * unit
        out_uncertainty = self._prop_err_add_sub(uncertainty, unit)
        return FuzzyQuantity(value=out_value, uncertainty=out_uncertainty)

    __rsub__ = __sub__

    def __mul__(self, other):
        value, uncertainty, unit = self._parse_input(other)
        out_value = self.value * self.unit * value * unit
        out_uncertainty = self._prop_err_mul_truediv(out_value, value,
                                                     uncertainty)
        return FuzzyQuantity(value=out_value, uncertainty=out_uncertainty)

    __rmul__ = __mul__

    def __truediv__(self, other):
        value, uncertainty, unit = self._parse_input(other)
        out_value = self.value * self.unit / (value * unit)
        out_uncertainty = self._prop_err_mul_truediv(out_value, value,
                                                     uncertainty)
        return FuzzyQuantity(value=out_value, uncertainty=out_uncertainty)

    __rtruediv__ = __truediv__

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

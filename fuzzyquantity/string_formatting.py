import numpy as np
import astropy.units as u


def _truncate_to_appropriate_digits(
        value,
        uncertainty) -> tuple[float, float, float, int, float, int]:
    """
    Calculate the appropriate number of significant digits for a given
    value/uncertainty pair.

    Parameters
    ----------
    value : int or float
        The value.
    uncertainty : int or float
        The uncertainty associated with the value.

    Returns
    -------
    tuple[float, float, float, int, float, int]
        The sign of the value, the truncated value, the truncated uncertainty,
        the power of 10, the scale factor for each number and the number of
        decimal places.
    """
    sign = np.sign(value)
    value = np.abs(value)

    unc_power = int(f'{uncertainty:e}'.split('e')[1])
    val_power = int(f'{value:e}'.split('e')[1])

    uncertainty_formatted = f'{uncertainty:.1e}'
    plus_one = True
    decimals = 1
    if uncertainty_formatted[0] != '1':
        uncertainty_formatted = f'{uncertainty:.0e}'
        decimals = 0
        plus_one = False
    decimals = decimals + val_power - unc_power
    if decimals <= 0:
        decimals = 0
        if plus_one:
            decimals += 1

    if unc_power <= val_power:
        scale = float(f'1e{val_power}')
        power = val_power
    else:
        scale = float(f'1e{unc_power}')
        power = unc_power
    value_scaled = float(f'{np.round(value / scale, decimals) * scale}')
    uncertainty_scaled = float(uncertainty_formatted)

    if np.abs(value_scaled) == 0:
        value_scaled = 0
        sign = 1.0

    return sign, value_scaled, uncertainty_scaled, power, scale, decimals


def _make_value_string(sign,
                       value,
                       scale,
                       decimals,
                       apply_scale: bool = True):
    if apply_scale:
        return f'{sign*value/scale:.{decimals}f}'
    else:
        decimals = int(decimals - np.log10(scale))
        return f'{sign*value:.{decimals}f}'


def _make_uncertainty_string(uncertainty,
                             scale,
                             decimals,
                             apply_scale: bool = True):
    if apply_scale:
        return f'{uncertainty/scale:.{decimals}f}'
    else:
        decimals = int(decimals - np.log10(scale))
        return f'{uncertainty:.{decimals}f}'


def _terminal_string(value,
                     uncertainty,
                     unit,
                     sci_thresh: int = 3) -> str:
    """
    Produce a terminal output string for a value and uncertainty formatted with
    proper significant figures.

    Parameters
    ----------
    value : int or float
        The value.
    uncertainty : int or float
        The uncertainty associated with the value.
    unit : u.Unit
        The units associated with the value.
    sci_thresh : int, optional
        The threshold for returning output in scientific notation. The default
        is 3, so any number equal to or larger than 1000 or equal to or smaller
        than 1/1000 will be returned in scientific notation form.

        For example, 999 ± 10 will return as `'999 ± 10'` but 1000 ± 10 will
        return as `'(1.00 ± 0.01)e+03'`.

    Returns
    -------
    str
        The properly-formatted terminal output string.
    """
    if unit != u.dimensionless_unscaled:
        unit = f' {unit}'
        prefix, suffix = '(', ')'
    else:
        unit, prefix, suffix = '', '', ''
    sign, value, uncertainty, power, scale, decimals = (
        _truncate_to_appropriate_digits(value, uncertainty))
    if np.abs(power) > sci_thresh:
        prefix, suffix = '(', ')'
        if power >= 0:
            magnitude = f'e+{power:02d}'
        else:
            magnitude = f'e{power:03d}'
    else:
        magnitude = ''
        decimals = int(decimals - np.log10(scale))
        scale = 1.0
    value_str = _make_value_string(sign, value, scale, decimals)
    uncertainty_str = _make_uncertainty_string(uncertainty, scale, decimals)
    return f'{prefix}{value_str} ± {uncertainty_str}{suffix}{magnitude}{unit}'


def _make_siunitx_string(value,
                         uncertainty,
                         unit,
                         sci_thresh) -> str:
    sign, value, uncertainty, power, scale, decimals = (
        _truncate_to_appropriate_digits(value, uncertainty))
    value_str = _make_value_string(sign, value, scale, decimals,
                                   apply_scale=False)
    uncertainty_str = _make_uncertainty_string(uncertainty, scale, decimals,
                                               apply_scale=False)
    unit_str = unit.to_string('latex', fraction=False)
    replacements = {r'$\mathrm{': '',
                    '}$': '',
                    r'\,': '.'}
    for key, value in replacements.items():
        unit_str = unit_str.replace(key, value)
    if np.abs(power) > sci_thresh:
        thresh = f'[exponent-mode=scientific]'
    else:
        thresh = ''
    if unit == u.dimensionless_unscaled:
        return fr'\num{thresh}{{{value_str}({uncertainty_str})}}'
    else:
        return fr"\SI{thresh}{{{value_str}({uncertainty_str})}}{{{unit_str}}}"


def _make_oldschool_latex_string(value,
                                 uncertainty,
                                 unit,
                                 sci_thresh) -> str:
    sign, value, uncertainty, power, scale, decimals = (
        _truncate_to_appropriate_digits(value, uncertainty))
    unit_str = unit.to_string('latex', fraction=False)
    out_str = _terminal_string(value, uncertainty, unit, sci_thresh)
    replacements = {'±': r'\pm',
                    ') ': r')\,',
                    f'e+{power:02d}': fr'\times 10^{{{power}}}',
                    f'e{power:03d}': fr'\times 10^{{{power}}}'}
    if unit != u.dimensionless_unscaled:
        replacements[unit.to_string()] = unit_str
        replacements['$'] = ''
    for key, value in replacements.items():
        out_str = out_str.replace(key, value)
    return out_str

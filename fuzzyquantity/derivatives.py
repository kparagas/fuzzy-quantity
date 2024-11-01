import sys

import numpy as np

step_size = np.sqrt(sys.float_info.epsilon)

@np.vectorize
def _pow_dxdy(x, y):
    """∂(xʸ)/∂y"""
    if (x == 0) and (y > 0):
        return 0.0
    if (x == 0) and (y <= 0):
        return np.nan
    return np.log(np.abs(x)) * x ** y

@np.vectorize
def _pow_dydx(x, y):
    """∂(xʸ)/∂x"""
    if y == 0:
        return 0.0
    if (x != 0) or (y % 1 == 0):
        return y * x ** (y - 1)
    return _numerical_derivative(np.power, x)

@np.vectorize
def _deriv_mod_dxdy(x, y):
    """∂(x%y)/∂y"""
    if x % y < step_size:
        return np.inf
    else:
        return _numerical_derivative(np.mod, y)

@np.vectorize
def _deriv_copysign(x, y):
    if x >= 0:
        return np.copysign(1, y)
    return -np.copysign(1, y)

# noinspection PyRedundantParentheses
derivatives = {
    # Python built-in operations
    'add': (lambda x, y: 1.0,
            lambda x, y: 1.0),
    'sub': (lambda x, y: 1.0,
            lambda x, y: -1.0),
    'div': (lambda x, y: 1/y,
            lambda x, y: -x/(y**2)),
    'truediv': (lambda x, y: 1/y,
                lambda x, y: -x/(y**2)),
    'floordiv': (lambda x, y: 0.0,
                 lambda x, y: 0.0),
    'mod': (lambda x, y: 1.0,
            _deriv_mod_dxdy),
    'mul': (lambda x, y: y,
            lambda x, y: x),
    'pow': (_pow_dydx,
            _pow_dxdy),
    # Numpy derivatives
    'arccos': (lambda x: -1/np.sqrt(1-x**2)),
    'arccosh': (lambda x: 1/np.sqrt(x**2-1)),
    'arcsin': (lambda x: 1/np.sqrt(1-x**2)),
    'arcsinh': (lambda x: 1/np.sqrt(1+x**2)),
    'arctan': (lambda x: 1/(1+x**2)),
    'arctan2': (lambda y, x: x/(x**2+y**2),  # Correct for x == 0
                lambda y, x: -y/(x**2+y**2)),  # Correct for x == 0
    'arctanh': (lambda x: 1/(1-x**2)),
    'cos': (lambda x: -np.sin(x)),
    'cosh': (np.sinh),
    'copysign': (_deriv_copysign, lambda x, y: 0),
    'exp': (np.exp),
    'expm1': (np.exp),
    'exp2': (lambda x: np.exp2(x)*np.log(2)),
    'hypot': [lambda x, y: x/np.hypot(x, y),
              lambda x, y: y/np.hypot(x, y)],
    'log': (lambda x: 1/x),  # for np, log=ln
    'log10': (lambda x: 1/(x*np.log(10.0))),
    'log2': (lambda x: 1/(x*np.log(2.0))),
    'log1p': (lambda x: 1/(1+x)),
    'sin': (np.cos),
    'sinh': (np.cosh),
    'tan': (lambda x: 1+np.tan(x)**2),
    'tanh': (lambda x: 1-np.tanh(x)**2)
}
"""Derivatives of mathematical functions. Copied from `astropop` package."""

def _propagate_1(func, fx, x, sx):
    """
    Propagate errors using single-variable function derivatives.

    Parameters
    ----------
    func: string
        Function name to perform error propagation. Must be in derivatives
        keys.
    fx: float or array_like
        Numerical result of f(x, y).
    x: float or array_like
        Variable of the function.
    sx: float or array_like
        1-sigma errors of the function variable.

    Returns
    -------
    sf: float or array_like
        1-sigma uncorrelated error associated with the function operation.
    """
    if func not in derivatives.keys():
        raise NotImplementedError(f'Functional derivative for `{func}` not yet '
                                  f'implemented..')

    if np.size(derivatives[func]) != 1:
        raise TypeError(f'Function `{func}` is not a single-variable '
                         f'function.')

    try:
        deriv = derivatives[func](x)
        sf = deriv*sx
        return sf
    except (ValueError, ZeroDivisionError, OverflowError):
        shape = np.shape(fx)
        if len(shape) == 0:
            return np.nan
        else:
            return np.full(shape, fill_value=np.nan)

def _propagate_2(func, fxy, x, y, sx, sy):
    """
    Propagate errors using two-variable function derivatives.

    Parameters
    ----------
    func: string
        Function name to perform error propagation. Must be in derivatives
        keys.
    fxy: float or array_like
        Numerical result of f(x, y).
    x, y: float or array_like
        Variables of the function.
    sx, sy: float or array_like
        1-sigma errors of the function variables.

    Returns
    -------
    sf: float or array_like
        1-sigma uncorrelated error associated to the operation.
    """
    if func not in derivatives.keys():
        raise NotImplementedError(f'Functional derivative for `{func}` not yet '
                                  f'implemented..')

    if np.size(derivatives[func]) != 2:
        raise TypeError(f'Function `{func}` is not a two-variable '
                        f'function.')

    deriv_x, deriv_y = derivatives[func]
    try:
        del_x2 = np.square(deriv_x(x, y))
        del_y2 = np.square(deriv_y(x, y))
        sx2 = np.square(sx)
        sy2 = np.square(sy)
        sf = np.sqrt(del_x2*sx2 + del_y2*sy2)
        return sf
    except (ValueError, ZeroDivisionError, OverflowError):
        shape = np.shape(fxy)
        if len(shape) == 0:
            return np.nan
        else:
            return np.full(shape, fill_value=np.nan)

def _numerical_derivative(func, value) -> float:
    """
    Calculate numerical derivative of function evaluated for a given value
    using the ±epsilon method.

    Parameters
    ----------
    func: callable
        The function to compute the numerical derivative.
    value: int or float
        Value at which to evaluate the function.

    Returns
    -------
    float
        The numerical derivative of the function `func` evaluated at `value`.
    """
    if not callable(func):
        raise TypeError(f'Function {func} not callable.')

    h = step_size*np.abs(value)
    f_plus = func(value + h)
    f_minus = func(value - h)
    return (f_plus - f_minus) / (2 * h)

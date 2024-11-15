r"""
Poisson equation with source term.

Find :math:`u` such that:

.. math::
    \int_{\Omega} c \nabla v \cdot \nabla u
    = - \int_{\Omega_L} b v = - \int_{\Omega_L} f v p
    \;, \quad \forall v \;,

where :math:`b(x) = f(x) p(x)`, :math:`p` is a given FE field and :math:`f` is
a given general function of space.

This example demonstrates use of functions for defining material parameters,
regions, parameter variables or boundary conditions. Notably, it demonstrates
the following:

1. How to define a material parameter by an arbitrary function - see the
   function :func:`get_pars()` that evaluates :math:`f(x)` in quadrature
   points.
2. How to define a known function that belongs to a given FE space (field) -
   this function, :math:`p(x)`, is defined in a FE sense by its nodal values
   only - see the function :func:`get_load_variable()`.

In order to define the load :math:`b(x)` directly, the term ``dw_dot``
should be replaced by ``dw_integrate``.
"""
from __future__ import absolute_import
import numpy as nm

filename_mesh = 'bezier2.mesh'

options = {
    'nls' : 'newton',
    'ls' : 'ls',
}

materials = {
    'm' : ({'c' : 1.0},)
}

regions = {
    'Omega' : 'all',
    'Gamma' : ('vertex 475', 'vertex'),
}

fields = {
    'temperature' : ('real', 1, 'Omega', 1)
}

variables = {
    'u' : ('unknown field', 'temperature', 0),
    'v' : ('test field',    'temperature', 'u'),
    'b' : ('parameter field', 'temperature',
           {'setter' : 'get_load_variable'}),
}

ebcs = {
    'u' : ('Gamma', {'u.0' : 0})
}

integrals = {
    'i' : 1,
}

equations = {
    'Laplace equation' :
    """dw_laplace.i.Omega( m.c, v, u )
     = - dw_dot.i.Omega( v, b )"""
}

solvers = {
    'ls' : ('ls.scipy_direct', {}),
    'newton' : ('nls.newton', {
        'i_max'      : 1,
        'eps_a'      : 1e-10,
    }),
}

def get_load_variable(ts, coors, region=None, variable=None, **kwargs):
    """
    Define nodal values of 'p' in the nodal coordinates `coors`.
    """
    y = coors[:,1]

    val = y
    return val

functions = {
    'get_load_variable' : (get_load_variable,)
}
"""Hypothesis strategies."""
import random

import hypothesis.strategies as st
import numpy as np
from hypothesis import assume
from hypothesis.extra.pandas import columns, data_frames

from baybe.exceptions import NumericalUnderflowError
from baybe.parameters.categorical import (
    CategoricalEncoding,
    CategoricalParameter,
    TaskParameter,
)
from baybe.parameters.custom import CustomDiscreteParameter
from baybe.parameters.numerical import (
    NumericalContinuousParameter,
    NumericalDiscreteParameter,
)
from baybe.parameters.substance import SubstanceEncoding, SubstanceParameter
from baybe.utils import DTypeFloatNumpy

_largest_lower_interval = np.nextafter(
    np.nextafter(np.inf, 0, dtype=DTypeFloatNumpy), 0, dtype=DTypeFloatNumpy
)
"""
The largest possible value for the lower end of a continuous interval such that there
still exists a larger but finite number for the upper interval end.
"""

_decorrelate = st.one_of(
    st.booleans(),
    st.floats(min_value=0.0, max_value=1.0, exclude_min=True, exclude_max=True),
)
"""A strategy that creates decorrelation settings."""

parameter_name = st.text(min_size=1)
"""A strategy that creates parameter names."""

categories = st.lists(st.text(min_size=1), min_size=2, unique=True)
"""A strategy that creates parameter categories."""


@st.composite
def smiles(draw: st.DrawFn):
    """Generates short SMILES strings."""
    n_atoms = draw(st.integers(min_value=0, max_value=19))
    string = "C"
    for _ in range(n_atoms):
        next_atom = random.choice("CNO") if string[-1] == "C" else random.choice("C")
        string += next_atom
    return string


@st.composite
def substance_data(draw: st.DrawFn):
    """Generates data for class:`baybe.parameters.substance.SubstanceParameter`."""
    names = draw(st.lists(st.text(min_size=1), min_size=2, max_size=10, unique=True))
    substances = draw(st.lists(smiles(), min_size=len(names), max_size=len(names)))
    return dict(zip(names, substances))


@st.composite
def custom_encodings(draw: st.DrawFn):
    """Generates data for class:`baybe.parameters.custom.CustomDiscreteParameter`."""
    index = st.lists(st.text(min_size=1), min_size=2, max_size=10, unique=True)
    cols = columns(
        names_or_number=10,
        elements=st.floats(allow_nan=False, allow_infinity=False),
        unique=True,
        dtype=DTypeFloatNumpy,
    )
    return draw(data_frames(index=index, columns=cols))


@st.composite
def numerical_discrete_parameter(  # pylint: disable=inconsistent-return-statements
    draw: st.DrawFn,
):
    """Generates class:`baybe.parameters.numerical.NumericalDiscreteParameter`."""
    name = draw(parameter_name)
    values = draw(
        st.lists(
            st.one_of(
                st.integers(),
                st.floats(allow_infinity=False, allow_nan=False),
            ),
            min_size=2,
            unique=True,
        )
    )

    # Reject examples where the tolerance validator cannot be satisfied
    try:
        return NumericalDiscreteParameter(name=name, values=values)
    except NumericalUnderflowError:
        assume(False)


@st.composite
def numerical_continuous_parameter(draw: st.DrawFn):
    """Generates class:`baybe.parameters.numerical.NumericalContinuousParameter`."""
    name = draw(parameter_name)
    lower = draw(st.floats(max_value=_largest_lower_interval, allow_infinity=False))
    upper = draw(st.floats(min_value=lower, exclude_min=True, allow_infinity=False))
    return NumericalContinuousParameter(name=name, bounds=(lower, upper))


@st.composite
def categorical_parameter(draw: st.DrawFn):
    """Generates class:`baybe.parameters.categorical.CategoricalParameter`."""
    name = draw(parameter_name)
    values = draw(categories)
    encoding = draw(st.sampled_from(CategoricalEncoding))
    return CategoricalParameter(name=name, values=values, encoding=encoding)


@st.composite
def task_parameter(draw: st.DrawFn):
    """Generates class:`baybe.parameters.categorical.TaskParameter`."""
    name = draw(parameter_name)
    values = draw(categories)
    active_values = random.sample(values, random.randint(0, len(values)))
    return TaskParameter(name=name, values=values, active_values=active_values)


@st.composite
def substance_parameter(draw: st.DrawFn):
    """Generates class:`baybe.parameters.substance.SubstanceParameter`."""
    name = draw(parameter_name)
    data = draw(substance_data())
    decorrelate = draw(_decorrelate)
    encoding = draw(st.sampled_from(SubstanceEncoding))
    return SubstanceParameter(
        name=name, data=data, decorrelate=decorrelate, encoding=encoding
    )


@st.composite
def custom_parameter(draw: st.DrawFn):
    """Generates class:`baybe.parameters.custom.CustomDiscreteParameter`."""
    name = draw(parameter_name)
    data = draw(custom_encodings())
    decorrelate = draw(_decorrelate)
    return CustomDiscreteParameter(name=name, data=data, decorrelate=decorrelate)


parameter = st.one_of(
    [
        numerical_discrete_parameter(),
        numerical_continuous_parameter(),
        categorical_parameter(),
        task_parameter(),
        substance_parameter(),
        custom_parameter(),
    ]
)
"""A strategy that creates parameters."""

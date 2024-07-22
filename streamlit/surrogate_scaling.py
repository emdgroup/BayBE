"""# Surrogate Model Scaling

The purpose of this script is to compare different surrogate models and verify that
their predictions are invariant to changes in location/scale of the underlying target
function.

This means that the displayed function approximation should always look the same when
the input and output locations/scales are changed.
"""  # noqa: D415

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from funcy import rpartial

from baybe.parameters.numerical import NumericalDiscreteParameter
from baybe.recommenders.pure.bayesian.botorch import BotorchRecommender
from baybe.searchspace import SearchSpace
from baybe.surrogates import CustomONNXSurrogate
from baybe.surrogates.base import Surrogate
from baybe.targets.numerical import NumericalTarget
from baybe.utils.basic import get_subclasses
from baybe.utils.random import set_random_seed

# Number of values used for the input parameter
N_PARAMETER_VALUES = 1000


def cubic(
    x: np.ndarray, x_min: float, x_max: float, amplitude: float, bias: float
) -> np.ndarray:
    """Cubic test function."""
    out = amplitude * np.power((x - x_min) / (x_max - x_min), 3) + bias
    return out


def sin(
    x: np.ndarray, x_min: float, x_max: float, amplitude: float, bias: float
) -> np.ndarray:
    """Sinusoid test function."""
    out = amplitude * np.sin((x - x_min) / (x_max - x_min) * 2 * np.pi) + bias
    return out


def constant(
    x: np.ndarray, x_min: float, x_max: float, amplitude: float, bias: float
) -> np.ndarray:
    """Constant test function."""
    out = np.full(x.shape, fill_value=bias)
    return out


def linear(
    x: np.ndarray, x_min: float, x_max: float, amplitude: float, bias: float
) -> np.ndarray:
    """Linear test function."""
    out = amplitude * np.linspace(0, 1, len(x)) + bias
    return out


def main():
    """Create the streamlit dashboard."""
    # Show module docstring in dashboard
    st.info(__doc__)

    # Collect all available test functions
    test_functions = {
        "Sine": sin,
        "Constant": constant,
        "Linear": linear,
        "Cubic": cubic,
    }

    # Collect all available surrogate models
    surrogate_model_classes = {
        cls.__name__: cls
        for cls in get_subclasses(Surrogate)
        if not issubclass(cls, CustomONNXSurrogate)
    }

    # Streamlit simulation parameters
    st_random_seed = int(st.sidebar.number_input("Random seed", value=1337))
    st_function_name = st.sidebar.selectbox(
        "Test function", list(test_functions.keys())
    )
    st_surrogate_name = st.sidebar.selectbox(
        "Surrogate model", list(surrogate_model_classes.keys())
    )
    st_n_training_points = st.sidebar.slider("Number of training points", 1, 20, 5)
    st_n_recommendations = st.sidebar.slider("Number of recommendations", 1, 20, 5)
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        When scaling is implemented correctly, the plot should remain static (except for
        the axis labels) when changing the following parameters:
        """
    )
    st_upper_parameter_limit = st.sidebar.slider(
        "Upper parameter limit", 0.0, 100.0, 1.0
    )
    st_lower_parameter_limit = st.sidebar.slider(
        "Lower parameter limit", -100.0, 0.0, 0.0
    )
    st_function_amplitude = st.sidebar.slider("Function amplitude", 1.0, 100.0, 1.0)
    st_function_bias = st.sidebar.slider("Function bias", -100.0, 100.0, 0.0)

    # Set the chosen random seed
    set_random_seed(st_random_seed)

    # Construct the specific test function
    fun = rpartial(
        test_functions[st_function_name],
        st_lower_parameter_limit,
        st_upper_parameter_limit,
        st_function_amplitude,
        st_function_bias,
    )

    # Create the training data
    train_x = np.random.uniform(
        st_lower_parameter_limit, st_upper_parameter_limit, st_n_training_points
    )
    train_y = fun(train_x)
    measurements = pd.DataFrame({"x": train_x, "y": train_y})

    # Create the plotting grid and corresponding target values
    test_x = np.linspace(
        st_lower_parameter_limit, st_upper_parameter_limit, N_PARAMETER_VALUES
    )
    test_y = fun(test_x)
    candidates = pd.DataFrame({"x": test_x, "y": test_y})

    # Create the searchspace and objective
    parameter = NumericalDiscreteParameter(
        name="x",
        values=np.linspace(
            st_lower_parameter_limit, st_upper_parameter_limit, N_PARAMETER_VALUES
        ),
    )
    searchspace = SearchSpace.from_product(parameters=[parameter])
    objective = NumericalTarget(name="y", mode="MAX").to_objective()

    # Create the surrogate model and the recommender
    surrogate_model = surrogate_model_classes[st_surrogate_name]()
    recommender = BotorchRecommender(surrogate_model=surrogate_model)

    # Get the recommendations and extract the posterior mean / standard deviation
    recommendations = recommender.recommend(
        st_n_recommendations, searchspace, objective, measurements
    )
    posterior = surrogate_model.posterior(candidates)
    mean = posterior.mean.squeeze().detach().numpy()
    std = posterior.stddev.detach().numpy()

    # Visualize the test function, training points, model predictions, recommendations
    fig = plt.figure()
    plt.plot(test_x, test_y, color="tab:blue", label="Test function")
    plt.plot(train_x, train_y, "o", color="tab:blue")
    plt.plot(test_x, mean, color="tab:red", label="Surrogate model")
    plt.fill_between(test_x, mean - std, mean + std, alpha=0.2, color="tab:red")
    plt.vlines(
        recommendations, *plt.gca().get_ylim(), color="k", label="Recommendations"
    )
    plt.legend()
    st.pyplot(fig)


if __name__ == "__main__":
    main()

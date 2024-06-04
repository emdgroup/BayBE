"""Base class for all Bayesian recommenders."""

from abc import ABC
from typing import Callable, Optional

import pandas as pd
from attrs import define, field

from baybe.acquisition.acqfs import qExpectedImprovement
from baybe.acquisition.base import AcquisitionFunction
from baybe.acquisition.utils import convert_acqf
from baybe.exceptions import DeprecationError
from baybe.objectives.base import Objective
from baybe.recommenders.pure.base import PureRecommender
from baybe.searchspace import SearchSpace
from baybe.surrogates import _ONNX_INSTALLED, GaussianProcessSurrogate
from baybe.surrogates.base import Surrogate
from baybe.utils.dataframe import to_tensor

if _ONNX_INSTALLED:
    from torch import Tensor

    from baybe.surrogates import CustomONNXSurrogate


@define
class BayesianRecommender(PureRecommender, ABC):
    """An abstract class for Bayesian Recommenders."""

    surrogate_model: Surrogate = field(factory=GaussianProcessSurrogate)
    """The used surrogate model."""

    acquisition_function: AcquisitionFunction = field(
        converter=convert_acqf, factory=qExpectedImprovement, kw_only=True
    )
    """The used acquisition function class."""

    _botorch_acqf = field(default=None, init=False)
    """The current acquisition function."""

    _searchspace: SearchSpace | None = field(default=None, init=False)

    acquisition_function_cls: bool = field(default=None)
    "Deprecated! Raises an error when used."

    @acquisition_function_cls.validator
    def _validate_deprecated_argument(self, _, value) -> None:
        """Raise DeprecationError if old acquisition_function_cls parameter is used."""
        if value is not None:
            raise DeprecationError(
                "Passing 'acquisition_function_cls' to the constructor is deprecated. "
                "The parameter has been renamed to 'acquisition_function'."
            )

    def _setup_botorch_acqf(
        self,
        searchspace: SearchSpace,
        objective: Objective,
        measurements: pd.DataFrame,
    ) -> None:
        """Create the acquisition function for the current training data."""  # noqa: E501
        # TODO: Transition point from dataframe to tensor needs to be refactored.
        #   Currently, surrogate models operate with tensors, while acquisition
        #   functions with dataframes.
        train_x = searchspace.transform(measurements)
        train_y = objective.transform(measurements)
        self.surrogate_model._fit(searchspace, *to_tensor(train_x, train_y))
        self._botorch_acqf = self.acquisition_function.to_botorch(
            self.surrogate_model, train_x, train_y
        )

    def recommend(  # noqa: D102
        self,
        batch_size: int,
        searchspace: SearchSpace,
        objective: Optional[Objective] = None,
        measurements: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        # See base class.

        if objective is None:
            raise NotImplementedError(
                f"Recommenders of type '{BayesianRecommender.__name__}' require "
                f"that an objective is specified."
            )

        if (measurements is None) or (len(measurements) == 0):
            raise NotImplementedError(
                f"Recommenders of type '{BayesianRecommender.__name__}' do not support "
                f"empty training data."
            )

        if _ONNX_INSTALLED and isinstance(self.surrogate_model, CustomONNXSurrogate):
            CustomONNXSurrogate.validate_compatibility(searchspace)

        self._setup_botorch_acqf(searchspace, objective, measurements)

        recommendation = super().recommend(
            batch_size=batch_size,
            searchspace=searchspace,
            objective=objective,
            measurements=measurements,
        )

        self._searchspace = searchspace

        return recommendation

    def get_surrogate(self) -> Callable[[pd.DataFrame], tuple[Tensor, Tensor]]:  # noqa: D102
        def behaves_like_surrogate(exp_rep: pd.DataFrame, /) -> tuple[Tensor, Tensor]:
            comp_rep = self._searchspace.transform(exp_rep)
            return self.surrogate_model.posterior(to_tensor(comp_rep))

        return behaves_like_surrogate

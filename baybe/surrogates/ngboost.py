"""NGBoost surrogates.

Currently, the documentation for this surrogate is not available. This is due to a bug
in our documentation tool, see https://github.com/sphinx-doc/sphinx/issues/11750.

Since we plan to refactor the surrogates, this part of the documentation will be
available in the future. Thus, please have a look in the source code directly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Literal

from attr import define, field
from ngboost import NGBRegressor

from baybe.parameters.base import Parameter
from baybe.surrogates.base import GaussianSurrogate
from baybe.surrogates.utils import batchify, catch_constant_targets
from baybe.surrogates.validation import get_model_params_validator
from baybe.utils.scaling import ScalerProtocol

if TYPE_CHECKING:
    from torch import Tensor


@catch_constant_targets
@define
class NGBoostSurrogate(GaussianSurrogate):
    """A natural-gradient-boosting surrogate model."""

    # Class variables
    joint_posterior: ClassVar[bool] = False
    # See base class.

    supports_transfer_learning: ClassVar[bool] = False
    # See base class.

    _default_model_params: ClassVar[dict] = {"n_estimators": 25, "verbose": False}
    """Class variable encoding the default model parameters."""

    # Object variables
    model_params: dict[str, Any] = field(
        factory=dict,
        converter=dict,
        validator=get_model_params_validator(NGBRegressor.__init__),
    )
    """Optional model parameter that will be passed to the surrogate constructor."""

    _model: NGBRegressor | None = field(init=False, default=None, eq=False)
    """The actual model."""

    def __attrs_post_init__(self):
        self.model_params = {**self._default_model_params, **self.model_params}

    @staticmethod
    def _make_parameter_scaler(
        parameter: Parameter,
    ) -> ScalerProtocol | Literal["passthrough"]:
        # See base class.

        # Tree-like models do not require any input scaling
        return "passthrough"

    @batchify
    def _estimate_moments(self, candidates: Tensor) -> tuple[Tensor, Tensor]:
        # See base class.

        import torch

        # Get predictions
        dists = self._model.pred_dist(candidates)

        # Split into posterior mean and variance
        mean = torch.from_numpy(dists.mean())
        var = torch.from_numpy(dists.var)

        return mean, var

    def _fit(self, train_x: Tensor, train_y: Tensor, context: Any) -> None:
        # See base class.
        self._model = NGBRegressor(**(self.model_params)).fit(train_x, train_y.ravel())

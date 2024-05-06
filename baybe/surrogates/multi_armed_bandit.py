"""Multi-armed bandit surrogate."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import numpy as np
from attrs import define, field
from attrs.validators import ge, instance_of
from scipy.stats import beta

from baybe.parameters import CategoricalParameter
from baybe.searchspace.core import SearchSpace
from baybe.surrogates.base import Surrogate

if TYPE_CHECKING:
    from torch import Tensor


@define
class BernoulliMultiArmedBanditSurrogate(Surrogate):
    """A multi-armed bandit model with bernoulli likelihood and a beta prior."""

    joint_posterior: ClassVar[bool] = False
    supports_transfer_learning: ClassVar[bool] = False
    # see base class.

    n_arms: int = field(validator=[instance_of(int), ge(1)])
    """ Number of arms for the bandit """

    prior_alpha_beta: np.ndarray[float] = field(
        converter=lambda x: np.asarray(x, float)
    )
    """ Prior parameters for the bandit of shape (n_arms, 2) """

    _win_lose_counts: np.ndarray[int] = field(init=False)
    """ Storing win and lose counts for updating the prior"""

    @prior_alpha_beta.default
    def _default_prior_alpha_beta(self) -> np.ndarray[float]:
        return np.ones((self.n_arms, 2))

    @prior_alpha_beta.validator
    def _validate_prior_alpha_beta(self, attribute, value) -> None:
        if value.shape != (self.n_arms, 2):
            raise ValueError(
                f"The shape of '{attribute.name}' must match with the number of bandit "
                f"arms. The bandit was configured with {self.n_arms} arm(s), hence the "
                f"expected shape is ({self.n_arms}, 2). Given: {value.shape}."
            )
        if not np.all(value > 0.0):
            raise ValueError(
                f"All values in '{attribute.name}' must be strictly positive."
            )

    @_win_lose_counts.default
    def _default_win_lose_counts(self) -> np.ndarray[int]:
        return np.zeros_like(self.prior_alpha_beta, dtype=int)

    @property
    def _posterior_alpha_beta(self) -> np.ndarray[float]:
        return self._win_lose_counts + self.prior_alpha_beta

    @property
    def means(self) -> np.ndarray[float]:
        """Posterior means."""
        return beta(*self._posterior_alpha_beta.T).mean()

    @property
    def variances(self) -> np.ndarray[float]:
        """Posterior variance."""
        return beta(*self._posterior_alpha_beta.T).var()

    def _posterior(self, candidates: Tensor) -> tuple[Tensor, Tensor]:
        import torch

        candidate_arms = candidates.argmax(dim=-1)
        posterior_mean = self.means[candidate_arms]
        posterior_variance = self.variances[candidate_arms]
        return torch.tensor(posterior_mean), torch.tensor(posterior_variance)

    def _fit(self, searchspace: SearchSpace, train_x: Tensor, train_y: Tensor) -> None:
        if len(searchspace.parameters) != 1:
            raise Exception(
                "BernoulliMultiArmedBanditSurrogate only supports one"
                " categorical parameter in the search space."
            )
        if not isinstance(searchspace.parameters[0], CategoricalParameter):
            raise Exception(
                "BernoulliMultiArmedBanditSurrogate only supports one"
                " categorical parameter in the search space."
            )

        # win counts per arm
        self._win_lose_counts[:, 0] = (train_x * train_y).sum(axis=0)
        # lose counts per arm
        self._win_lose_counts[:, 1] = (train_x * (1 - train_y)).sum(axis=0)

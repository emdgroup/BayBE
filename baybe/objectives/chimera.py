"""Functionality for chimera objectives."""

import gc
import warnings
from enum import Enum
from typing import TypeGuard

import cattrs
import numpy as np
import numpy.typing as npt
import pandas as pd
from attrs import define, field
from attrs.validators import deep_iterable, ge, gt, instance_of, min_len
from typing_extensions import override

from baybe.objectives.base import Objective
from baybe.targets.base import Target
from baybe.targets.numerical import NumericalTarget
from baybe.utils.basic import to_tuple
from baybe.utils.dataframe import get_transform_objects, pretty_print_df
from baybe.utils.plotting import to_string
from baybe.utils.validation import finite_float


def _is_all_numerical_targets(
    x: tuple[Target, ...], /
) -> TypeGuard[tuple[NumericalTarget, ...]]:
    """Typeguard helper function."""
    return all(isinstance(y, NumericalTarget) for y in x)


class ThresholdType(Enum):
    """Available types for target thresholds."""

    ABSOLUTE = "ABSOLUTE"
    """The target threshold is an absolute value."""

    PERCENTILE = "PERCENTILE"
    """The target threshold is a percentile value."""

    FRACTION = "FRACTION"
    """The target threshold is a fraction value."""


@define(frozen=True, slots=False)
class ChimeraObjective(Objective):
    """An objective scalarizing multiple targets using Chimera Merits.

    This class implements the Chimera function for multi-objective optimization,
    allowing users to establish a hierarchy of targets with thresholds. The score
    is minimized.

    For details, see: https://pubs.rsc.org/ko/content/articlelanding/2018/sc/c8sc02239a#!divAbstract
    """

    _targets: tuple[Target, ...] = field(
        converter=to_tuple,
        validator=[min_len(2), deep_iterable(member_validator=instance_of(Target))],
        alias="targets",
    )
    "The targets considered by the objective."

    targets_threshold_values: tuple[float, ...] = field(
        converter=lambda w: cattrs.structure(w, tuple[float, ...]),
        validator=deep_iterable(member_validator=[finite_float, ge(0.0)]),
    )
    """The target degradation thresholds for each target from its optimum."""

    targets_threshold_types: tuple[ThresholdType, ...] | None = field(
        converter=lambda x: None
        if x is None
        else tuple(
            ThresholdType(value) if isinstance(value, str) else value for value in x
        )
    )
    """An optional tuple of target threshold types."""

    softness: float = field(
        converter=float,
        validator=gt(0.0),
    )
    """The softness parameter regulating the Heaviside function."""

    @targets_threshold_values.default
    def _default_targets_threshold_values(self) -> tuple[float, ...]:
        default_values = (0.0,) * len(self._targets)  # TODO: intepretation?
        warnings.warn(
            f"The values for targets thresholds have not been specified. "
            f"Setting the target threshold values to {default_values}.",
            UserWarning,
        )
        return default_values

    @targets_threshold_types.default
    def _default_targets_threshold_types(self) -> tuple[ThresholdType, ...]:
        default_values = (ThresholdType.FRACTION,) * len(self._targets)
        warnings.warn(
            f"The types for target thresholds have not been specified. "
            f"Setting the target threshold types to {default_values}.",
            UserWarning,
        )
        return default_values

    @softness.default  # TODO: do we need to add warning here?
    def _default_softness(self) -> float:
        default_value = 1e-3
        return default_value

    @_targets.validator
    def _validate_targets(self, _, targets) -> None:  # noqa: DOC101, DOC103
        if not _is_all_numerical_targets(targets):
            raise TypeError(
                f"'{self.__class__.__name__}' currently only supports targets "
                f"of type '{NumericalTarget.__name__}'."
            )
        if len({t.name for t in targets}) != len(targets):
            raise ValueError("All target names must be unique.")
        if not all(target._is_transform_normalized for target in targets):
            raise ValueError(
                "All targets must have normalized computational representations to "
                "enable the computation of desirability values. This requires having "
                "appropriate target bounds and transformations in place."
            )

    @targets_threshold_values.validator
    def _validate_targets_threshold_values(self, _, values) -> None:
        if (lv := len(values)) != (lt := len(self._targets)):
            raise ValueError(
                f"If custom threshold values are specified, there must be one for each target. "  # noqa: E501
                f"Specified number of targets: {lt}. Specified number of threshold values: {lv}."  # noqa: E501
            )

    @targets_threshold_types.validator
    def _validate_targets_threshold_types(self, _, types) -> None:
        if (lt := len(types)) != (ltg := len(self._targets)):
            raise ValueError(
                f"If custom threshold types are specified, there must be one for each target. "  # noqa: E501
                f"Specified number of targets: {ltg}. Specified number of threshold types: {lt}."  # noqa: E501
            )

    def _soft_heaviside(self, value: float, softness: float) -> float:
        arg = -value / softness
        return np.exp(-np.logaddexp(0, arg))

    def _hard_heaviside(self, value: float) -> float:
        return (value >= 0.0).astype(
            float
        )  # Pandas handles booleans as floats automatically

    def step(self, value: float, softness: float = 1e-6) -> float:
        """Apply a step function to the given value based on the specified softness.

        Args:
            value: The input value to apply the step function to.
            softness: The softness parameter for the step function.
                If less than 1e-5, a hard Heaviside step function is used.
                Otherwise, a soft Heaviside step function is used.
                Default is 1e-6.

        Returns:
            The result of the step function applied to the input value.
        """
        if softness < 1e-5:
            return self._hard_heaviside(value)

        return self._soft_heaviside(value, softness)

    def _invert_binary(self, a: float) -> float:
        return 1 - a

    def _shift(
        self,
        transformed: pd.DataFrame,
        transformed_threshold_values: list[float],
    ) -> tuple[np.ndarray, np.ndarray]:
        # Initialize with the first column of transformed
        shifted_values = [transformed.values[:, 0]]
        shifted_thresholds = []
        # Initialize the shift, where the primary target is unshifted
        shift = 0.0
        # Initialize the domain with the index of transformed
        domain = transformed.index

        for target, threshold_value, threshold_type in zip(
            self.targets, transformed_threshold_values, self.targets_threshold_types
        ):
            if threshold_type == ThresholdType.FRACTION:
                _threshold = threshold_value
            elif threshold_type == ThresholdType.PERCENTILE:
                _threshold = transformed[target.name].quantile(
                    threshold_value, interpolation="linear"
                )
            elif threshold_type == ThresholdType.ABSOLUTE:
                _threshold = threshold_value
            else:
                raise ValueError(f"Unsupported ThresholdType: {threshold_type}")

            # Compute and store shifted threshold
            _shifted_threshold = _threshold - shift
            shifted_thresholds.append(_shifted_threshold)

            # Adjust to region of interest for the next (lower-)level objective
            interest = transformed[target.name][domain] < _shifted_threshold
            if interest.any():
                domain = domain[interest]
            #     print(target, "| New domain: ", domain, "\n")
            # else:
            #     print(target, "| No interest", "\n")
            #     continue

            # Compute new shift
            current_idx = self.targets.index(target)
            next_idx = (current_idx + 1) % len(
                self.targets
            )  # Loop back to target_idx == 0
            shift = transformed.values[:, next_idx].max() - min(shifted_thresholds)
            # TODO: Explanation
            # We ensure no value of lower-level target can exceed the baseline
            # defined (minimum) by the cumulative minima from higher-level target
            # Apply shift directly to the corresponding target values
            _shifted_value = transformed.values[:, next_idx] - shift
            shifted_values.append(_shifted_value)
            # print("next_idx: ", next_idx, "\n")

        return np.array(shifted_values), np.asarray(shifted_thresholds)

    def _scalarize(
        self, shifted_values: npt.ArrayLike, shifted_thresholds: npt.ArrayLike
    ) -> np.ndarray:
        # Start with the last term in the shifted_transformed (the fallback term)
        # TODO: explain a bit what is this fallback term
        merits = shifted_values[-1].copy()

        # Reverse iterate through all but the last target
        for idx in reversed(range(shifted_values.shape[0] - 1)):
            current_obj = shifted_values[idx]
            current_tol = shifted_thresholds[idx]

            # Compute step functions / positive and negative masks
            pos_mask = self.step(current_obj - current_tol)
            neg_mask = self._invert_binary(pos_mask)
            # TODO: here typecasting happening

            # Scalarize through inversely updating merits:
            # (kept if within threshold, else replaced by higher-level)
            merits = merits * neg_mask + pos_mask * current_obj

        # Normalize CHIMERA merits
        if merits.max() > 0:
            merits = (merits - merits.min()) / (merits.max() - merits.min())

        return merits

    @override
    @property
    def targets(self) -> tuple[Target, ...]:
        return self._targets

    @override
    def __str__(self) -> str:
        targets_list = [target.summary() for target in self.targets]
        targets_df = pd.DataFrame(targets_list)
        targets_df["Threshold values"] = self.targets_threshold_values
        targets_df["Threshold types"] = [t.value for t in self.targets_threshold_types]

        fields = [
            to_string("Type", self.__class__.__name__, single_line=True),
            to_string("Targets", pretty_print_df(targets_df)),
            # to_string("Scalarizer", "Chimera", single_line=True),
        ]

        return to_string("Objective", *fields)

    @override
    def transform(
        self,
        df: pd.DataFrame | None = None,
        /,
        *,
        allow_missing: bool = False,
        allow_extra: bool | None = None,
        data: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        # >>>>>>>>>> Deprecation
        if not ((df is None) ^ (data is None)):
            raise ValueError(
                "Provide the dataframe to be transformed as argument to `df`."
            )

        if data is not None:
            df = data
            warnings.warn(
                "Providing the dataframe via the `data` argument is deprecated and "
                "will be removed in a future version. Please pass your dataframe "
                "as positional argument instead.",
                DeprecationWarning,
            )

        # Mypy does not infer from the above that `df` must be a dataframe here
        assert isinstance(df, pd.DataFrame)

        if allow_extra is None:
            allow_extra = True
            if set(df.columns) - {p.name for p in self.targets}:
                warnings.warn(
                    "For backward compatibility, the new `allow_extra` flag is set "
                    "to `True` when left unspecified. However, this behavior will be "
                    "changed in a future version. If you want to invoke the old "
                    "behavior, please explicitly set `allow_extra=True`.",
                    DeprecationWarning,
                )
        # <<<<<<<<<< Deprecation

        # Extract the relevant part of the dataframe
        targets = get_transform_objects(
            df, self.targets, allow_missing=allow_missing, allow_extra=allow_extra
        )
        transformed = df[[t.name for t in targets]].copy()

        # Transform all targets individually
        for target in self.targets:
            transformed[target.name] = target.transform(df[target.name])
            # All values in transformed become "closer to 1, the better" here
            # TODO: for non-numerical targets? for MODE="MATCH"?

        # Transform threshold values for each target
        _threshold_values_transformed = list(self.targets_threshold_values)
        # TODO: typecasting happening

        def transform_threshold_value(x: float, target: NumericalTarget) -> float:
            """Transform the threshold value using the target's transform method."""
            return target.transform(pd.Series([x])).values[0]

        # TODO: for non-numerical targets? for MODE="MATCH"?

        for target, threshold_type, threshold_value in zip(
            targets, self.targets_threshold_types, self.targets_threshold_values
        ):
            # Invert maximization problems to minimization problems
            transformed[target.name] = 1.0 - transformed[target.name]
            # TODO: for non-numerical targets? for MODE="MATCH"?
            _threshold_values_transformed[targets.index(target)] = threshold_value
            # Invert the threshold value if it is an absolute threshold
            if threshold_type == ThresholdType.ABSOLUTE:
                _threshold_values_transformed[targets.index(target)] = (
                    1.0 - transform_threshold_value(threshold_value, target)
                )
            # TODO: everything becomes a minimization problem, meaning value < threshold

        # Shift objectives and thresholds
        shifted_values, shifted_thresholds = self._shift(
            transformed, _threshold_values_transformed
        )
        # TODO: caching?

        # Scalarize the shifted targets into CHIMERA merit values
        vals = self._scalarize(shifted_values, shifted_thresholds)
        # TODO: How do we intepretate this value? Examples + clarification
        # TODO: Is normalization needed? Is this closer to 1 the better or the opposite?
        # TODO: What happens if we reformulate this to a maximization problem?

        # Store the total Chimera merit in a dataframe column
        transformed = pd.DataFrame({"Merit": vals}, index=transformed.index)

        return transformed


# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()

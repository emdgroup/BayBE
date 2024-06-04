"""Demo for surrogate model access."""

import numpy as np
import pandas as pd

from baybe import Campaign
from baybe.parameters.numerical import NumericalDiscreteParameter
from baybe.searchspace.core import SearchSpace
from baybe.targets.numerical import NumericalTarget

# Define setup
parameters = [
    NumericalDiscreteParameter(name="p1", values=[1, 2, 3]),
    NumericalDiscreteParameter(name="p2", values=[4, 5, 6]),
]
objective = NumericalTarget(name="target", mode="MAX")
searchspace = SearchSpace.from_product(parameters)
campaign = Campaign(searchspace, objective)

# Generate data
inputs = pd.DataFrame.from_records([(1, 4)], columns=["p1", "p2"])
targets = pd.DataFrame({"target": np.random.rand(len(inputs))})
measurements = pd.concat([inputs, targets], axis=1)
campaign.add_measurements(measurements)

# Recommend to train surrogate
campaign.recommend(3)

# Extract surrogate and evaluate it on the search space
surrogate = campaign.recommender.recommender.get_surrogate()
mean, covar = surrogate(campaign.searchspace.discrete.exp_rep)
print(mean, covar)

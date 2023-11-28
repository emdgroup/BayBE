<div align="center">
  <br/>
  
[![ci](https://github.com/emdgroup/BayBE/actions/workflows/ci.yml/badge.svg)](https://github.com/emdgroup/BayBE/actions?query=workflow%3Aci.yml)
[![regular](https://github.com/emdgroup/BayBE/actions/workflows/regular.yml/badge.svg)](https://github.com/emdgroup/BayBE/actions?query=workflow%3Aregular.yml)
<a href="https://pypi.org/project/baybe/"><img src="https://img.shields.io/pypi/pyversions/baybe?style=flat-square&label=Supported%20Python%20versions&color=%23ffb86c"/></a>
<a href="https://anaconda.org/conda-forge/baybe"><img src="https://img.shields.io/conda/vn/conda-forge/baybe.svg?style=flat-square&label=Conda%20Forge%20Version&color=%23bd93f9" alt="Conda Version"/></a>
<a href="https://pypi.org/project/baybe/"><img src="https://img.shields.io/pypi/v/baybe.svg?style=flat-square&label=PyPI%20version&color=%23bd93f9"/></a>
<a href="https://github.com/emdgroup/BayBE/issues"><img src="https://img.shields.io/github/issues/emdgroup/BayBE?style=flat-square&color=%23ff79c6"/></a>
<a href="https://github.com/emdgroup/BayBE/pulls"><img src="https://img.shields.io/github/issues-pr/emdgroup/BayBE?style=flat-square&color=%23ff79c6"/></a>
<a href="http://www.apache.org/licenses/LICENSE-2.0"><img src="https://shields.io/badge/License-Apache%202.0-green.svg?style=flat-square&color=%234c1"/></a>

[![Logo](https://raw.githubusercontent.com/emdgroup/baybe/main/docs/images/banner2.svg)](https://github.com/emdgroup/baybe)

  <p><a href="https://emdgroup.github.io/baybe">Documentation</a></p>
</div>

# BayBE — A Bayesian Back End for Design of Experiments

The **Bay**esian **B**ack **E**nd provides a general-purpose toolbox for Bayesian Design of 
Experiments, focusing on additions that enable real-world experimental campaigns.

Besides functionality to perform a typical recommend-measure loop, BayBE's highlights are:
- Custom parameter encodings: Improve your campaign with domain knowledge
- Built-in chemical encodings: Improve your campaign with chemical knowledge
- Single and multiple targets with min, max and match objectives
- Custom surrogate models: For specialized problems or active learning
- Hybrid (mixed continuous and discrete) spaces
- Transfer learning: Mix data from multiple campaigns and accelerate optimization
- Comprehensive backtest, simulation and imputation utilities: Benchmark and find your best settings
- Fully typed and hypothesis-tested: Robust code base
- The entire state of your campaign is fully de-/serializable: Useful for storing results in databases

## Installation
### From Package Index
The easiest way to install BayBE is via PyPI or Conda:

```bash
pip install baybe
```

```bash
conda install --channel=conda-forge baybe
```

### From Repository
If you need finer control and would like to install a specific commit that has not been
released under a certain version tag, you can do so by installing BayBE directly from
the repository.
First, clone the repository, navigate to the repository root folder, check out the
desired commit, and run:

```bash
pip install .
```

There are additional dependencies that can be installed corresponding to linters, 
formatters etc. (e.g. `dev`). A developer would typically also install the package in 
editable mode ('-e') which ensures that changes to the code do not require a 
reinstallation.

```bash
pip install -e '.[dev]'
```

### Optional Dependencies
There are several dependency groups that can be installed during pip installation like
```bash
pip install baybe[test,lint] # will install baybe with additional dependency groups `test` and `lint`
```
To get the most out of `baybe` we recommend to install at least
```bash
pip install baybe[chem,simulation]
```

The available groups are:
- `chem`: Cheminformatics utilities (e.g. for the [`SubstanceParameter`](baybe.parameters.substance.SubstanceParameter)).
- `docs`: Required for creating the documentation.
- `examples`: Required for running the examples/streamlit.
- `lint`: Required for linting and formatting.
- `mypy`: Required for static type checking.
- `onnx`: Required for using custom surrogate models in ONNX format.
- `simulation`: Enabling the [`simulation`](baybe.simulation) module.
- `test`: Required for running the tests.
- `dev`: All of the above plus `tox` and `pip-audit`. For code contributors.


## Getting Started

BayBE is a DOE software built to streamline your experimental process.
It can process measurement data from previous experiments and, based on these, provide
optimal experimental designs to further improve your target quantities.

In order to make use of BayBE's optimization capabilities, you need to translate your
real-world optimization problem into mathematical language.
To do so, you should ask yourself the following questions:

* What should be optimized?
* What are the degrees of freedom?
* (Optional) What optimization strategy should be used?

Conveniently, the answer to each of these questions can be directly expressed in the
form of objects in BayBE's ecosystem that can be easily mixed and matched:

| Part of the Problem Specification                     | Defining BayBE Objects    |
|:------------------------------------------------------|:--------------------------|
| What should be optimized?                             | `Objective`, `Target`     |
| What are the degrees of freedom?                      | `Parameter`, `Constraint` | 
| (Optional) What optimization strategy should be used? | `Strategy`, `Recommender` |

The objects in the first two table rows can be regarded as embodiments of the
**mathematical DOE specifications** in digital form, as they fully define the underlying
optimization problem.
By contrast, the objects in the last row rather provide **algorithmic details**
on how the DOE problem should be solved.
In that sense, the former carry information that **must be** provided by the user,
whereas the latter are **optional** settings that can also be set automatically
by BayBE.

A key element in the design of BayBE is the [`Campaign`](baybe.campaign.Campaign) object.
It acts as a central container for all the necessary information and objects
associated with an experimentation process, ensuring that all independent model
components (e.g. the objective function, the search space, etc.) are properly combined.

The following example provides a step-by-step guide to what this translation process
should look like, and how we can subsequently use BayBE to generate optimal sets of
experimental conditions.

### Defining the Optimization Objective

We start by defining an optimization objective.
While BayBE ships with the necessary functionality to optimize multiple targets
simultaneously, as an introductory example, we consider a simple scenario where our
goal is to **maximize** a single numerical target that represents the yield of a
chemical reaction.

In BayBE's language, the reaction yield can be represented as a [`NumericalTarget`](baybe.targets.numerical)
object:

```python
from baybe.targets import NumericalTarget

target = NumericalTarget(
    name="Yield",
    mode="MAX",
)
```

We wrap the target object in an optimization [`Objective`](baybe.objective.Objective), to inform BayBE
that this is the only target we would like to consider:

```python
from baybe.objective import Objective

objective = Objective(mode="SINGLE", targets=[target])
```

In cases where we need to consider multiple (potentially competing) targets, the
role of the [`Objective`](baybe.objective.Objective) is to define how these targets should be balanced.
For more details, see [the targets section of the user guide](docs/userguide/targets.md).

### Defining the Search Space

Next, we inform BayBE about the available "control knobs", that is, the underlying
system parameters we can tune to optimize our targets.
This also involves specifying their ranges and other parameter-specific details.

For our reaction example, we assume that we can control the following three quantities:

```python
from baybe.parameters import CategoricalParameter, NumericalDiscreteParameter, SubstanceParameter

parameters = [
    CategoricalParameter(
        name="Granularity",
        values=["coarse", "medium", "fine"],
        encoding="OHE",
    ),
    NumericalDiscreteParameter(
        name="Pressure[bar]",
        values=[1, 5, 10],
        tolerance=0.2,
    ),
    SubstanceParameter(
        name="Solvent",
        data={"Solvent A": "COC", "Solvent B": "CCC", "Solvent C": "O",
              "Solvent D": "CS(=O)C"},
        encoding="MORDRED",
    ),
]
```

Note that each parameter is of a different **type** and thus requires its own
type-specific settings. In particular case above, for instance:

* `encoding=OHE` activates one-hot-encoding for the categorical parameter "Granularity".
* `tolerance=0.2` allows experimental inaccuracies up to 0.2 when reading values for
  "Pressure[bar]".
* `encoding=MORDRED`triggers computation of MORDRED cheminformatics descriptors for
  the substance parameter "Solvent".

For more parameter types and their details, see
[parameters section of the user guide](docs/userguide/parameters).

Additionally, we can define a set of constraints to further specify allowed ranges and
relationships between our parameters.
Details can be found in [the constraints section of the user guids](docs/userguide/constraints).
In this example, we assume no further constraints and explicitly indicate this with an
empty variable, for the sake of demonstration:

```python
constraints = None
```

With the parameter and constraint definitions at hand, we can now create our
[`SearchSpace`](baybe.searchspace):

```python
from baybe.searchspace import SearchSpace

searchspace = SearchSpace.from_product(parameters, constraints)
```

### Optional: Defining the Optimization Strategy

As an optional step, we can specify details on how the optimization should be
conducted.
If omitted, BayBE will choose a default setting.

For our chemistry example, we combine two selection strategies:

1. In cases where no measurements have been made prior to the interaction with BayBE,
   a random experiment selection strategy is used to produce initial recommendations.
2. As soon as the first measurements are available, we switch to a Bayesian approach
   where points are selected greedily from a probabilistic prediction model.

For more details on the different strategies, their underlying algorithmic
details, and their configuration settings, see
[the strategies section of the user guide](docs/userguide/strategy).

```python
from baybe.strategies import TwoPhaseStrategy
from baybe.recommenders import SequentialGreedyRecommender, RandomRecommender

strategy = TwoPhaseStrategy(
    initial_recommender=RandomRecommender(),
    recommender=SequentialGreedyRecommender(),
)
```

### The Optimization Loop

Having provided the answers to [all questions above](#getting-started), we can now
construct a BayBE object that brings all
pieces of the puzzle together:

```python
from baybe import Campaign

campaign = Campaign(searchspace, objective, strategy)
```

With this object at hand, we can start our experimentation cycle.
In particular:

* We can ask BayBE to `recommend` new experiments.
* We can `add_measurements` for certain experimental settings to BayBE's database.

Note that these two steps can be performed in any order.
In particular, available measurement data can be submitted at any time.
Also, we can start the interaction with either command and repeat the same type of
command immediately after its previous call, e.g., if the required number of
recommendations has changed.

The following illustrates one such possible sequence of interactions.
Let us first ask for an initial set of recommendations:

```python
df = campaign.recommend(batch_quantity=5)
```

For a particular random seed, the result could look as follows:

| Granularity   | Pressure[bar]   | Solvent   |
|---------------|-----------------|-----------|
| medium        | 1               | Solvent B |
| medium        | 5               | Solvent D |
| fine          | 5               | Solvent C |
| fine          | 5               | Solvent A |
| medium        | 10              | Solvent B |

After having conducted the corresponding experiments, we can add our measured
yields to the table and feed it back to BayBE:

```python
df["Yield"] = [79, 54, 59, 95, 84]
campaign.add_measurements(df)
```

With the newly arrived data, BayBE will update its internal state and can produce a
refined design for the next iteration.


## Known Issues
A list of know issues can be found [here](docs/known_issues.md).


## License

Copyright 2022-2023 Merck KGaA, Darmstadt, Germany
and/or its affiliates. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

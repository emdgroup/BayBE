from baybe.acquisition.acqfs import qLogNoisyExpectedHypervolumeImprovement
from baybe.objectives.multi import MultiTargetObjective
from baybe.parameters.numerical import NumericalContinuousParameter
from baybe.recommenders.pure.bayesian.botorch import BotorchRecommender
from baybe.searchspace.core import SearchSpace
from baybe.targets.numerical import NumericalTarget
from baybe.utils.dataframe import add_fake_measurements

parameters = [
    NumericalContinuousParameter("p1", (0, 1)),
    NumericalContinuousParameter("p2", (0, 1)),
]
targets = [
    NumericalTarget("t1", "MAX"),
    NumericalTarget("t2", "MAX"),
]
objective = MultiTargetObjective(targets)
searchspace = SearchSpace.from_product(parameters)
measurements = searchspace.continuous.sample_uniform(10)
add_fake_measurements(measurements, targets)
recommender = BotorchRecommender(
    acquisition_function=qLogNoisyExpectedHypervolumeImprovement(
        ref_point=measurements[[t.name for t in targets]].min()
    )
)
recs = recommender.recommend(5, searchspace, objective, measurements)
print(recs)

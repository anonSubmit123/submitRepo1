
from __future__ import annotations

from typing import Any

from core.algorithms.logical_controller import (
    AgentState,
    IndicativeComponent,
    SurrogateEfficacyProvider,
)
from core.distribution import AStarRefocusAssignmentOptimizer, GreedyAssignmentOptimizer

from .policy import InMemoryPolicy, InMemoryPolicyBank


class MetadataSurrogateEfficacyProvider:

    def __init__(
        self,
        default_score: float = 0.5,
        capability_weight: float = 0.1,
        demand_penalty_weight: float = 0.05,
        config: Any = None,
        factory: Any = None,
    ) -> None:
        self.default_score = default_score
        self.capability_weight = capability_weight
        self.demand_penalty_weight = demand_penalty_weight
        self.config = config
        self.factory = factory

    def score(self, agent: AgentState, component: IndicativeComponent) -> float:
        required = component.required_capabilities
        capability_names = agent.capabilities.keys()
        if required and not required.issubset(capability_names):
            return 0.0

        base_score = float(component.metadata.get("base_efficacy", self.default_score))
        capability_bonus = self.capability_weight * len(
            capability_names.intersection(required)
        )
        demand_penalty = (
            min(max(component.demand, 0.0), 1.0) * self.demand_penalty_weight
        )
        return max(0.0, min(1.0, base_score + capability_bonus - demand_penalty))

import typing

if typing.TYPE_CHECKING:
    from core.algorithms.logical_controller import AssignmentOptimizer

_istype_AStarRefocusAssignmentOptimizer: typing.Type[AssignmentOptimizer] = (
    AStarRefocusAssignmentOptimizer
)
_istype_GreedyAssignmentOptimizer: typing.Type[AssignmentOptimizer] = (
    GreedyAssignmentOptimizer
)
_istype_MetadataSurrogateEfficacyProvider: typing.Type[SurrogateEfficacyProvider] = (
    MetadataSurrogateEfficacyProvider
)

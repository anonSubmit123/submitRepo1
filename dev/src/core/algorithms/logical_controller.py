from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol


AgentId = str
ActivityId = str
ComponentId = str


@dataclass(frozen=True)
class CapabilityProfile:

    name: str
    values: tuple[tuple[str, float], ...] = ()

    @classmethod
    def from_mapping(
        cls,
        name: str,
        values: Mapping[str, Any],
    ) -> "CapabilityProfile":
        return cls(
            name=name,
            values=tuple(
                sorted((str(key), float(value)) for key, value in values.items())
            ),
        )

    @classmethod
    def from_names(
        cls,
        name: str,
        capabilities: Any,
    ) -> "CapabilityProfile":
        return cls.from_mapping(
            name,
            {str(item): 1.0 for item in capabilities},
        )

    @classmethod
    def coerce(cls, value: Any, name: str = "inline") -> "CapabilityProfile":
        if isinstance(value, CapabilityProfile):
            return value
        if isinstance(value, Mapping):
            return cls.from_mapping(name, value)
        return cls.from_names(name, value)

    def as_dict(self) -> dict[str, float]:
        return dict(self.values)

    def keys(self) -> frozenset[str]:
        return frozenset(key for key, _ in self.values)

    def items(self) -> tuple[tuple[str, float], ...]:
        return self.values

    def get(self, key: str, default: float = 0.0) -> float:
        return self.as_dict().get(key, default)

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and key in self.keys()

    def __iter__(self):
        return iter(sorted(self.keys()))

    def __len__(self) -> int:
        return len(self.values)


@dataclass(frozen=True)
class ComponentLocation:

    x: float
    y: float


@dataclass(frozen=True)
class ComponentSize:

    width: float
    height: float


@dataclass(frozen=True)
class IndicativeComponent:

    component_id: ComponentId
    demand: float
    since: int
    ic_type: int = 0
    required_capabilities: frozenset[str] = frozenset()
    location: ComponentLocation | None = None
    size: ComponentSize | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentState:

    agent_id: AgentId
    activity_id: ActivityId
    workload: float
    capabilities: CapabilityProfile = field(
        default_factory=lambda: CapabilityProfile(name="default")
    )
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "capabilities",
            CapabilityProfile.coerce(self.capabilities, self.agent_id),
        )


@dataclass(frozen=True)
class Assignment:

    agent_id: AgentId
    activity_id: ActivityId
    component_id: ComponentId


@dataclass(frozen=True)
class ControllerPush:

    agent_id: AgentId
    activity_id: ActivityId
    component_id: ComponentId
    surrogate_score: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EnvironmentSnapshot:

    time: int
    components: dict[ComponentId, IndicativeComponent]
    agents: dict[AgentId, AgentState]


@dataclass(frozen=True)
class RedistributionPlan:

    assignments: dict[AgentId, Assignment]
    surrogate_score: float = 0.0
    baseline_score: float = 0.0
    monotonic: bool = True

    def improves_baseline(self) -> bool:
        return self.monotonic and self.surrogate_score >= self.baseline_score


class SurrogateEfficacyProvider(Protocol):

    def score(self, agent: AgentState, component: IndicativeComponent) -> float:
        ...


@dataclass(frozen=True)
class AssignmentContext:

    snapshot: EnvironmentSnapshot
    current_assignment: dict[str, Assignment]
    efficacy_provider: SurrogateEfficacyProvider


class AssignmentOptimizer(Protocol):

    def solve(self, context: AssignmentContext) -> RedistributionPlan:
        ...


class ControllerPushDispatcher(Protocol):

    def send_push(self, push: ControllerPush) -> None:
        ...


@dataclass
class LogicalControllerAlgorithm:

    optimizer: AssignmentOptimizer
    efficacy_provider: SurrogateEfficacyProvider
    push_dispatcher: ControllerPushDispatcher | None = None
    controller_interval: int = 25
    last_update_time: int = 0
    assignment: dict[AgentId, Assignment] = field(default_factory=dict)

    def step(
        self,
        snapshot: EnvironmentSnapshot,
    ) -> list[ControllerPush]:
        should_redistribute = (
            snapshot.time - self.last_update_time >= self.controller_interval
            or not self.assignment
        )
        if not should_redistribute:
            return []

        context = AssignmentContext(
            snapshot=snapshot,
            current_assignment=self.assignment,
            efficacy_provider=self.efficacy_provider,
        )
        plan = self.optimizer.solve(context)
        if not plan.improves_baseline():
            return []

        pushes = self._build_pushes(snapshot, plan)

        self.assignment = dict(plan.assignments)
        self.last_update_time = snapshot.time
        self._deliver_pushes(pushes)
        return pushes

    def _deliver_pushes(self, pushes: list[ControllerPush]) -> None:
        if self.push_dispatcher is None:
            return
        for push in pushes:
            self.push_dispatcher.send_push(push)

    def reset(self) -> None:
        self.last_update_time = 0
        self.assignment.clear()

    def _build_pushes(
        self,
        snapshot: EnvironmentSnapshot,
        plan: RedistributionPlan,
    ) -> list[ControllerPush]:
        pushes: list[ControllerPush] = []
        for agent_id, item in sorted(plan.assignments.items()):
            agent = snapshot.agents[agent_id]
            component = snapshot.components[item.component_id]
            surrogate_score = self.efficacy_provider.score(agent, component)
            pushes.append(
                ControllerPush(
                    agent_id=agent_id,
                    activity_id=item.activity_id,
                    component_id=item.component_id,
                    surrogate_score=surrogate_score,
                    metadata={"plan_score": plan.surrogate_score},
                )
            )
        return pushes

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from .logical_controller import (
    AgentId,
    CapabilityProfile,
    ComponentId,
    ControllerPush,
)


PolicyId = str
AlgorithmId = str


@dataclass(frozen=True)
class PolicySelection:
    algorithm_id: AlgorithmId
    policy: "RLPolicy"


class PolicyBankClient(Protocol):

    def get(self, descriptor: "ActivityDescriptor") -> PolicySelection:
        ...


class RLPolicy(Protocol):
    policy_id: PolicyId
    version: int
    algorithm_id: AlgorithmId

    def select_action(self, observation: Any) -> Any:
        ...

    def step_completed(self, step: "StepRecord") -> None:
        ...

    def get_trajectory(self) -> "Trajectory":
        ...

    def reset_trajectory(self) -> None:
        ...


PolicyHandle = RLPolicy


class AgentEnvironment(Protocol):

    @property
    def stats(self) -> Any:
        ...

    def observe(self, component_id: ComponentId) -> Any:
        ...

    def execute(self, component_id: ComponentId, action: Any) -> "ExecutionResult":
        ...

    def reset(self) -> None:
        ...


class AgentTransport(Protocol):

    def send_trajectory_report(self, report: "TrajectoryReport") -> None:
        ...


@dataclass(frozen=True)
class ActivityDescriptor:
    activity_id: str
    capabilities: CapabilityProfile
    local_context: Any = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "capabilities",
            CapabilityProfile.coerce(self.capabilities, ""),
        )


@dataclass(frozen=True)
class ExecutionResult:
    reward: float
    next_observation: Any
    realized_efficacy: float
    terminal: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class StepRecord:
    step_id: int = 0
    obs_t: Any = None
    action_t: Any = None
    reward_t: float = 0.0
    obs_next: Any = None
    done: bool = False
    reward_status_t: Any = None
    reward_status_next: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrajectoryElement:
    element_id: int = 0
    observation: Any = None
    action: Any = None
    reward: float = 0.0
    done: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SARSTrajectoryElement(TrajectoryElement):
    next_observation: Any = None


@dataclass
class TrajectoryPolicyElement(TrajectoryElement):
    log_probability: float | None = None


@dataclass
class Trajectory:
    policy_id: PolicyId = ""
    algorithm_id: AlgorithmId = "ppo"
    agent_id: AgentId | None = None
    elements: list[TrajectoryElement] = field(default_factory=list)

    def add_step(self, element: TrajectoryElement) -> None:
        if element.element_id == 0:
            element.element_id = len(self.elements) + 1
        self.elements.append(element)

    def total_reward(self) -> float:
        return sum(element.reward for element in self.elements)

    def average_reward_per_action(self) -> float:
        if not self.elements:
            return 0.0
        return self.total_reward() / len(self.elements)

    def reset(self) -> None:
        self.elements.clear()


@dataclass(frozen=True)
class TrajectoryReport:
    agent_id: AgentId
    descriptor: ActivityDescriptor
    algorithm_id: AlgorithmId = "ppo"
    trajectory: Trajectory | None = None
    surrogate_score: float = 0.0
    realized_efficacy: float = 0.0
    policy_version: int = 0


@dataclass(frozen=True)
class AgentStepResult:
    action: Any | None
    reported: bool
    shaped_reward: float | None = None


@dataclass
class PolicyGuidedAgentExpedition:
    agent_id: AgentId
    capabilities: CapabilityProfile
    policy_bank: PolicyBankClient
    environment: AgentEnvironment
    transport: AgentTransport
    report_interval: int = 25
    local_context: Any = None
    current_push: ControllerPush | None = None
    descriptor: ActivityDescriptor | None = None
    policy: PolicyHandle | None = None
    algorithm_id: AlgorithmId | None = None
    trajectory: Trajectory | None = None
    last_realized_efficacy: float = 0.0
    step_count: int = 0
    evaluation_return: float = 0.0

    def __post_init__(self) -> None:
        self.capabilities = CapabilityProfile.coerce(self.capabilities, self.agent_id)

    def receive_push(self, push: ControllerPush) -> None:
        self._report_and_clear_if_needed(force=True)
        self.current_push = push
        self.descriptor = ActivityDescriptor(
            activity_id=push.activity_id,
            capabilities=self.capabilities,
            local_context=self.local_context,
        )
        selection = self.policy_bank.get(self.descriptor)
        self.algorithm_id = selection.algorithm_id
        self.policy = selection.policy
        self.trajectory = Trajectory(
            policy_id=self.policy.policy_id,
            algorithm_id=self.algorithm_id,
            agent_id=self.agent_id,
        )

    def step(self, force_report: bool = False) -> AgentStepResult:
        if (
            self.current_push is None
            or self.descriptor is None
            or self.policy is None
            or self.algorithm_id is None
            or self.trajectory is None
        ):
            return AgentStepResult(action=None, reported=False)

        self.step_count += 1
        push = self.current_push
        observation = self.environment.observe(push.component_id)
        action = self.policy.select_action(observation)
        result = self.environment.execute(push.component_id, action)
        self.evaluation_return += result.reward

        step_record = StepRecord(
            step_id=self.step_count,
            obs_t=observation,
            action_t=action,
            reward_t=result.reward,
            obs_next=result.next_observation,
            done=result.terminal,
            metadata={
                "component_id": push.component_id,
                "surrogate_score": push.surrogate_score,
                **result.metadata,
            },
        )
        self.policy.step_completed(step_record)
        self._append_trajectory_element(step_record)

        self.last_realized_efficacy = result.realized_efficacy
        reported = self._report_and_clear_if_needed(
            force=force_report or result.terminal,
        )
        return AgentStepResult(
            action=action,
            reported=reported,
            shaped_reward=result.reward,
        )

    def _append_trajectory_element(self, step: StepRecord) -> None:
        if self.trajectory is None:
            return
        if self.algorithm_id == "dqn":
            element = SARSTrajectoryElement(
                observation=step.obs_t,
                action=step.action_t,
                reward=step.reward_t,
                done=step.done,
                next_observation=step.obs_next,
                metadata=dict(step.metadata),
            )
        else:
            element = TrajectoryPolicyElement(
                observation=step.obs_t,
                action=step.action_t,
                reward=step.reward_t,
                done=step.done,
                log_probability=step.metadata.get("log_probability"),
                metadata=dict(step.metadata),
            )
        self.trajectory.add_step(element)

    def _report_and_clear_if_needed(self, force: bool = False) -> bool:
        has_trajectory = self.trajectory is not None and bool(self.trajectory.elements)
        if (
            not has_trajectory
            or self.current_push is None
            or self.descriptor is None
        ):
            return False
        should_report = force or (
            self.report_interval > 0 and self.step_count % self.report_interval == 0
        )
        if not should_report:
            return False

        policy_version = self.policy.version if self.policy is not None else 0
        self.transport.send_trajectory_report(
            TrajectoryReport(
                agent_id=self.agent_id,
                descriptor=self.descriptor,
                algorithm_id=self.algorithm_id or "ppo",
                trajectory=self.trajectory,
                surrogate_score=self.current_push.surrogate_score,
                realized_efficacy=self.last_realized_efficacy,
                policy_version=policy_version,
            )
        )
        if self.trajectory is not None:
            self.trajectory = Trajectory(
                policy_id=self.trajectory.policy_id,
                algorithm_id=self.trajectory.algorithm_id,
                agent_id=self.agent_id,
            )
        if self.policy is not None:
            self.policy.reset_trajectory()
        return True

    def reset(self) -> None:
        self.current_push = None
        self.descriptor = None
        self.policy = None
        self.algorithm_id = None
        self.trajectory = None
        self.last_realized_efficacy = 0.0
        self.step_count = 0
        self.evaluation_return = 0.0
        environment_reset = getattr(self.environment, "reset", None)
        if environment_reset is not None:
            environment_reset()

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from .policy_guided_agent_expedition import (
    ActivityDescriptor,
    Trajectory,
    TrajectoryReport,
)
from .logical_controller import CapabilityProfile


PolicyId = str
AlgorithmId = str


class SharedPolicyBank(Protocol):

    def match_activity(
        self,
        activity_id: str,
        activity_type: str,
        capabilities: CapabilityProfile,
    ) -> str:
        ...

    def index(self, descriptor: ActivityDescriptor) -> tuple[PolicyId, int]:
        ...

    def merge_trajectories(
        self,
        policy_id: PolicyId,
        algorithm_id: AlgorithmId,
        trajectories: list[Trajectory],
    ) -> int:
        ...


@dataclass(frozen=True)
class ScoredTrajectory:
    report: TrajectoryReport
    score: float
    return_sum: float


@dataclass(frozen=True)
class PolicyUpdateStats:
    pools_seen: int
    retained_trajectories: int
    replayed_trajectories: int
    transitions_seen: int
    algorithm_updates: dict[AlgorithmId, int] = field(default_factory=dict)


class TrajectoryMerger(Protocol):

    def merge(
        self,
        manager: "SharedPolicyManager",
        policy_id: PolicyId,
        items: list[ScoredTrajectory],
    ) -> int:
        ...


@dataclass
class PpoTrajectoryMerger:

    def merge(
        self,
        manager: "SharedPolicyManager",
        policy_id: PolicyId,
        items: list[ScoredTrajectory],
    ) -> int:
        trajectories = [
            item.report.trajectory
            for item in items
            if item.report.trajectory is not None
        ]
        return manager.policy_bank.merge_trajectories(
            policy_id,
            "ppo",
            trajectories,
        )


@dataclass
class A2CTrajectoryMerger:

    def merge(
        self,
        manager: "SharedPolicyManager",
        policy_id: PolicyId,
        items: list[ScoredTrajectory],
    ) -> int:
        trajectories = [
            item.report.trajectory
            for item in items
            if item.report.trajectory is not None
        ]
        return manager.policy_bank.merge_trajectories(
            policy_id,
            "actorcritic",
            trajectories,
        )


@dataclass
class DQNTrajectoryMerger:

    def merge(
        self,
        manager: "SharedPolicyManager",
        policy_id: PolicyId,
        items: list[ScoredTrajectory],
    ) -> int:
        trajectories = [
            item.report.trajectory
            for item in items
            if item.report.trajectory is not None
        ]
        return manager.policy_bank.merge_trajectories(
            policy_id,
            "dqn",
            trajectories,
        )


@dataclass
class StrategyFilter:
    success_rate: float = 0.20
    failure_rate: float = 0.10
    cap: int = 256

    def select(self, pool: list[ScoredTrajectory]) -> list[ScoredTrajectory]:
        if not pool:
            return []

        ordered = sorted(pool, key=lambda item: item.score)
        failure_count = int(len(pool) * self.failure_rate)
        success_count = int(len(pool) * self.success_rate)
        selected = ordered[:failure_count]
        if success_count:
            selected += ordered[-success_count:]
        if not selected:
            selected = ordered

        deduped: list[ScoredTrajectory] = []
        seen: set[int] = set()
        for item in selected:
            marker = id(item)
            if marker not in seen:
                seen.add(marker)
                deduped.append(item)
        return deduped[: self.cap]


@dataclass
class SharedPolicyManager:
    policy_bank: SharedPolicyBank
    strategy_filter: StrategyFilter = field(default_factory=StrategyFilter)
    stale_version_threshold: int = 5
    beta_reward: float = 0.50
    max_replay: int = 4
    replay_epsilon: float = 1e-9
    mergers: dict[AlgorithmId, TrajectoryMerger] = field(default_factory=dict)
    pools: dict[PolicyId, list[ScoredTrajectory]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.mergers:
            self.mergers = {
                "ppo": PpoTrajectoryMerger(),
                "actorcritic": A2CTrajectoryMerger(),
                "a2c": A2CTrajectoryMerger(),
                "dqn": DQNTrajectoryMerger(),
            }

    def ingest_report(self, report: TrajectoryReport) -> bool:
        policy_id, current_version = self.policy_bank.index(report.descriptor)
        if current_version - report.policy_version > self.stale_version_threshold:
            return False

        trajectory_return = report_return(report)
        score = trajectory_score(
            trajectory_return=trajectory_return,
            surrogate_score=report.surrogate_score,
            realized_efficacy=report.realized_efficacy,
            beta_reward=self.beta_reward,
        )
        self.pools.setdefault(policy_id, []).append(
            ScoredTrajectory(
                report=report,
                score=score,
                return_sum=trajectory_return,
            )
        )
        return True

    def run_update_round(self) -> PolicyUpdateStats:
        retained_trajectories = 0
        replayed_trajectories = 0
        transitions_seen = 0
        algorithm_updates: dict[AlgorithmId, int] = {}

        for policy_id, pool in list(self.pools.items()):
            retained = self.strategy_filter.select(pool)
            retained_trajectories += len(retained)
            mean_abs_score = _mean_abs_score(retained, self.replay_epsilon)

            replay_buffer: list[ScoredTrajectory] = []
            for item in retained:
                copies = 1 + min(
                    self.max_replay,
                    int(abs(item.score) / mean_abs_score),
                )
                replay_buffer.extend([item] * copies)
                replayed_trajectories += copies

            by_algorithm: dict[AlgorithmId, list[ScoredTrajectory]] = {}
            for item in replay_buffer:
                by_algorithm.setdefault(item.report.algorithm_id, []).append(item)

            for algorithm_id, items in by_algorithm.items():
                merger = self.mergers.get(algorithm_id) or self.mergers["ppo"]
                consumed = merger.merge(self, policy_id, items)
                transitions_seen += consumed
                algorithm_updates[algorithm_id] = (
                    algorithm_updates.get(algorithm_id, 0) + consumed
                )

        pools_seen = len([pool for pool in self.pools.values() if pool])
        self.pools.clear()
        return PolicyUpdateStats(
            pools_seen=pools_seen,
            retained_trajectories=retained_trajectories,
            replayed_trajectories=replayed_trajectories,
            transitions_seen=transitions_seen,
            algorithm_updates=algorithm_updates,
        )

    def reset(self) -> None:
        self.pools.clear()
        bank_reset = getattr(self.policy_bank, "reset", None)
        if bank_reset is not None:
            bank_reset()


def trajectory_score(
    trajectory_return: float,
    surrogate_score: float,
    realized_efficacy: float,
    beta_reward: float,
) -> float:
    return trajectory_return * (beta_reward + surrogate_score * realized_efficacy)


def report_return(report: TrajectoryReport) -> float:
    if report.trajectory is not None:
        return report.trajectory.total_reward()
    return 0.0


def _mean_abs_score(pool: list[ScoredTrajectory], epsilon: float) -> float:
    if not pool:
        return epsilon
    return sum(abs(item.score) for item in pool) / len(pool) + epsilon

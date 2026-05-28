
from __future__ import annotations

import os
import pickle
import logging
import random
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.algorithms.policy_guided_agent_expedition import (
    ActivityDescriptor,
    PolicySelection,
    SARSTrajectoryElement,
    StepRecord,
    Trajectory,
    TrajectoryElement,
    TrajectoryPolicyElement,
)
from core.algorithms.logical_controller import CapabilityProfile

LOGGER = logging.getLogger(__name__)


SUPPORTED_ALGORITHMS = frozenset({"ppo", "actorcritic", "a2c", "dqn"})


def _coerce_capabilities(capabilities: Any) -> CapabilityProfile:
    return CapabilityProfile.coerce(capabilities, "")

def _capability_key(capabilities: Any) -> str:
    profile = _coerce_capabilities(capabilities)
    if profile.name:
        return profile.name
    return ",".join(f"{key}={value:g}" for key, value in profile.items())

def _observation_vector(observation: Any) -> list[float]:
    if isinstance(observation, dict):
        values = observation.values()
    elif isinstance(observation, (list, tuple)):
        values = observation
    else:
        values = (observation,)

    vector: list[float] = []
    for value in values:
        if isinstance(value, bool):
            vector.append(1.0 if value else 0.0)
        elif isinstance(value, (int, float)):
            vector.append(float(value))
    return vector or [0.0]

def _policy_class_for_algorithm(algorithm_id: str) -> type["InMemoryPolicy"]:
    normalized = algorithm_id.lower()
    if normalized == "dqn":
        return DQNPolicy
    if normalized in {"actorcritic", "a2c"}:
        return A2CPolicy
    if normalized == "ppo":
        return PPOPolicy
    return InMemoryPolicy

def _create_policy(policy_id: str, algorithm_id: str) -> "InMemoryPolicy":
    return _policy_class_for_algorithm(algorithm_id)(policy_id, algorithm_id)

class InMemoryPolicy:

    def __init__(self, policy_id: str, algorithm_id: str = "ppo") -> None:
        self.policy_id = policy_id
        self.algorithm_id = algorithm_id
        self.version = 0
        self._trajectory = Trajectory(policy_id=policy_id, algorithm_id=algorithm_id)

    def select_action(self, observation: Any) -> str:
        return "operate"

    def step_completed(self, step: StepRecord) -> None:
        element = TrajectoryPolicyElement(
            observation=step.obs_t,
            action=step.action_t,
            reward=step.reward_t,
            done=step.done,
            metadata=dict(step.metadata),
        )
        self._trajectory.add_step(element)

    def get_trajectory(self) -> Trajectory:
        return self._trajectory

    def reset_trajectory(self) -> None:
        self._trajectory = Trajectory(
            policy_id=self.policy_id,
            algorithm_id=self.algorithm_id,
        )

class BaseTorchPolicy(InMemoryPolicy):

    def __init__(
        self,
        policy_id: str,
        algorithm_id: str,
        action_count: int = 3,
        hidden_size: int = 128,
        learning_rate: float = 1e-3,
        discount: float = 0.99,
    ) -> None:
        super().__init__(policy_id, algorithm_id)
        self.action_count = action_count
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.discount = discount
        self.state_dim: int | None = None
        self._last_action_metadata: dict[str, Any] = {}

    def _state_tensor(self, observation: Any) -> torch.Tensor:
        vector = _observation_vector(observation)
        self._ensure_networks(len(vector))
        return torch.tensor(vector, dtype=torch.float32).unsqueeze(0)

    def _ensure_networks(self, state_dim: int) -> None:
        if self.state_dim is not None:
            return
        self.state_dim = state_dim
        self._init_networks(state_dim)

    def _init_networks(self, state_dim: int) -> None:
        raise NotImplementedError

    def step_completed(self, step: StepRecord) -> None:
        step.metadata.update(self._last_action_metadata)
        element = TrajectoryPolicyElement(
            observation=step.obs_t,
            action=step.action_t,
            reward=step.reward_t,
            done=step.done,
            log_probability=step.metadata.get("log_probability"),
            metadata=dict(step.metadata),
        )
        self._trajectory.add_step(element)

class PPOPolicy(BaseTorchPolicy):

    def __init__(self, policy_id: str, algorithm_id: str = "ppo") -> None:
        super().__init__(policy_id, algorithm_id, learning_rate=3e-4)
        self.clip_epsilon = 0.2
        self.ppo_epochs = 2
        self.policy_net: nn.Module | None = None
        self.value_net: nn.Module | None = None
        self.optimizer: torch.optim.Optimizer | None = None

    def _init_networks(self, state_dim: int) -> None:
        self.policy_net = nn.Sequential(
            nn.Linear(state_dim, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.action_count),
        )
        self.value_net = nn.Sequential(
            nn.Linear(state_dim, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1),
        )
        self.optimizer = torch.optim.Adam(
            list(self.policy_net.parameters()) + list(self.value_net.parameters()),
            lr=self.learning_rate,
        )

    def select_action(self, observation: Any) -> int:
        assert self.policy_net is not None or self.state_dim is None
        state = self._state_tensor(observation)
        assert self.policy_net is not None
        logits = self.policy_net(state)
        probs = torch.softmax(logits, dim=1)
        dist = torch.distributions.Categorical(probs)
        action_tensor = dist.sample()
        action = int(action_tensor.item())
        probability = float(probs[0, action].detach().item())
        self._last_action_metadata = {
            "log_probability": float(dist.log_prob(action_tensor).detach().item()),
            "old_action_probability": max(probability, 1e-9),
        }
        return action

    def step_completed(self, step: StepRecord) -> None:
        super().step_completed(step)
        if self.policy_net is None or self.value_net is None or self.optimizer is None:
            return

        state = self._state_tensor(step.obs_t)
        next_state = self._state_tensor(step.obs_next)
        old_probability = float(step.metadata.get("old_action_probability", 1.0))
        action = int(step.action_t)
        reward = torch.tensor(float(step.reward_t), dtype=torch.float32)
        done_mask = 0.0 if step.done else 1.0

        for _ in range(self.ppo_epochs):
            value = self.value_net(state).squeeze()
            next_value = self.value_net(next_state).detach().squeeze()
            target = reward + done_mask * self.discount * next_value
            advantage = target - value
            logits = self.policy_net(state)
            probs = torch.softmax(logits, dim=1)
            probability = probs[0, action]
            ratio = probability / max(old_probability, 1e-9)
            clipped = torch.clamp(
                ratio,
                1.0 - self.clip_epsilon,
                1.0 + self.clip_epsilon,
            )
            actor_loss = -torch.min(ratio * advantage.detach(), clipped * advantage.detach())
            critic_loss = F.mse_loss(value, target.detach())
            entropy = -(probs * torch.log(probs + 1e-9)).sum()
            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        self.version += 1

class A2CPolicy(BaseTorchPolicy):

    def __init__(self, policy_id: str, algorithm_id: str = "actorcritic") -> None:
        super().__init__(policy_id, algorithm_id, learning_rate=1e-3)
        self.entropy_coef = 0.01
        self.policy_net: nn.Module | None = None
        self.value_net: nn.Module | None = None
        self.actor_optimizer: torch.optim.Optimizer | None = None
        self.critic_optimizer: torch.optim.Optimizer | None = None

    def _init_networks(self, state_dim: int) -> None:
        self.policy_net = nn.Sequential(
            nn.Linear(state_dim, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.action_count),
        )
        self.value_net = nn.Sequential(
            nn.Linear(state_dim, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1),
        )
        self.actor_optimizer = torch.optim.Adam(
            self.policy_net.parameters(),
            lr=self.learning_rate,
        )
        self.critic_optimizer = torch.optim.Adam(
            self.value_net.parameters(),
            lr=self.learning_rate,
        )

    def select_action(self, observation: Any) -> int:
        state = self._state_tensor(observation)
        assert self.policy_net is not None
        logits = self.policy_net(state)
        probs = torch.softmax(logits, dim=1)
        dist = torch.distributions.Categorical(probs)
        action_tensor = dist.sample()
        action = int(action_tensor.item())
        probability = float(probs[0, action].detach().item())
        self._last_action_metadata = {
            "log_probability": float(dist.log_prob(action_tensor).detach().item()),
            "old_action_probability": max(probability, 1e-9),
        }
        return action

    def step_completed(self, step: StepRecord) -> None:
        super().step_completed(step)
        if (
            self.policy_net is None
            or self.value_net is None
            or self.actor_optimizer is None
            or self.critic_optimizer is None
        ):
            return

        state = self._state_tensor(step.obs_t)
        next_state = self._state_tensor(step.obs_next)
        action = int(step.action_t)
        reward = torch.tensor(float(step.reward_t), dtype=torch.float32)
        done_mask = 0.0 if step.done else 1.0

        value = self.value_net(state).squeeze()
        next_value = self.value_net(next_state).detach().squeeze()
        target = reward + done_mask * self.discount * next_value
        advantage = target - value

        logits = self.policy_net(state)
        log_probs = torch.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)
        entropy = -(probs * log_probs).sum()
        actor_loss = -(log_probs[0, action] * advantage.detach()) - self.entropy_coef * entropy
        critic_loss = F.mse_loss(value, target.detach())

        self.actor_optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor_optimizer.step()
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        self.version += 1

@dataclass
class ReplayBuffer:

    capacity: int = 1000
    buffer: deque[SARSTrajectoryElement] = field(default_factory=deque)

    def push(self, element: SARSTrajectoryElement) -> None:
        if len(self.buffer) >= self.capacity:
            self.buffer.popleft()
        self.buffer.append(element)

    def sample(self, batch_size: int) -> list[SARSTrajectoryElement]:
        return random.sample(list(self.buffer), batch_size)

    def __len__(self) -> int:
        return len(self.buffer)

class DQNPolicy(BaseTorchPolicy):

    def __init__(self, policy_id: str, algorithm_id: str = "dqn") -> None:
        super().__init__(policy_id, algorithm_id, learning_rate=1e-3, discount=0.95)
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995
        self.batch_size = 32
        self.target_update_freq = 100
        self.step_count = 0
        self.replay_buffer = ReplayBuffer()
        self.q_network: nn.Module | None = None
        self.target_network: nn.Module | None = None
        self.optimizer: torch.optim.Optimizer | None = None

    def _init_networks(self, state_dim: int) -> None:
        self.q_network = nn.Sequential(
            nn.Linear(state_dim, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.action_count),
        )
        self.target_network = nn.Sequential(
            nn.Linear(state_dim, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.action_count),
        )
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        self.optimizer = torch.optim.Adam(
            self.q_network.parameters(),
            lr=self.learning_rate,
        )

    def select_action(self, observation: Any) -> int:
        state = self._state_tensor(observation)
        assert self.q_network is not None
        if random.random() < self.epsilon:
            action = random.randrange(self.action_count)
        else:
            with torch.no_grad():
                q_values = self.q_network(state)
            action = int(torch.argmax(q_values, dim=1).item())
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
        self._last_action_metadata = {}
        return action

    def step_completed(self, step: StepRecord) -> None:
        step.metadata.update(self._last_action_metadata)
        element = SARSTrajectoryElement(
            observation=step.obs_t,
            action=step.action_t,
            reward=step.reward_t,
            done=step.done,
            next_observation=step.obs_next,
            metadata=dict(step.metadata),
        )
        self._trajectory.add_step(element)
        self.replay_buffer.push(element)
        self._update_policy()

    def _update_policy(self) -> None:
        if (
            self.q_network is None
            or self.target_network is None
            or self.optimizer is None
            or len(self.replay_buffer) < self.batch_size
        ):
            return

        transitions = self.replay_buffer.sample(self.batch_size)
        state_batch = torch.tensor(
            [_observation_vector(item.observation) for item in transitions],
            dtype=torch.float32,
        )
        next_state_batch = torch.tensor(
            [_observation_vector(item.next_observation) for item in transitions],
            dtype=torch.float32,
        )
        action_batch = torch.tensor(
            [int(item.action) for item in transitions],
            dtype=torch.int64,
        )
        reward_batch = torch.tensor(
            [float(item.reward) for item in transitions],
            dtype=torch.float32,
        )
        done_batch = torch.tensor(
            [bool(item.done) for item in transitions],
            dtype=torch.bool,
        )

        q_values = self.q_network(state_batch).gather(1, action_batch.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            max_next_q = torch.max(self.target_network(next_state_batch), dim=1)[0]
        target_q = reward_batch + self.discount * max_next_q * (~done_batch)
        loss = F.mse_loss(q_values, target_q.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        self.version += 1

class InMemoryPolicyBank:

    def __init__(self) -> None:
        self._policies: dict[str, InMemoryPolicy] = {}
        self._activity_to_policy_map: dict[str, str] = {}
        self._activity_to_algorithm_map: dict[str, str] = {}

    def match_activity(
        self,
        activity_id: str,
        activity_type: str,
        capabilities: CapabilityProfile,
        algorithm_id: str | None = None,
    ) -> str:
        caps_str = _capability_key(capabilities)
        algorithm = algorithm_id or self._select_algorithm(activity_type, capabilities)
        policy_id = f"{algorithm}:{activity_type}:{caps_str}"
        self._activity_to_policy_map[activity_id] = policy_id
        self._activity_to_algorithm_map[activity_id] = algorithm
        if policy_id not in self._policies:
            self._policies[policy_id] = _create_policy(policy_id, algorithm)
        return policy_id

    def get(self, descriptor: ActivityDescriptor) -> PolicySelection:
        policy_id, _ = self.index(descriptor)
        policy = self._policies[policy_id]
        return PolicySelection(
            algorithm_id=getattr(policy, "algorithm_id", "ppo"),
            policy=policy,
        )

    def index(self, descriptor: ActivityDescriptor) -> tuple[str, int]:
        policy_id = self._activity_to_policy_map.get(descriptor.activity_id)
        if not policy_id:
            algorithm = self._select_algorithm(
                descriptor.activity_id,
                descriptor.capabilities,
                descriptor.local_context,
            )
            policy_id = (
                f"{algorithm}:{descriptor.activity_id}:"
                f"{_capability_key(descriptor.capabilities)}"
            )
        algorithm = self._activity_to_algorithm_map.get(
            descriptor.activity_id,
            policy_id.split(":", 1)[0],
        )
        policy = self._policies.setdefault(
            policy_id,
            _create_policy(policy_id, algorithm),
        )
        return policy.policy_id, policy.version

    def _select_algorithm(
        self,
        activity_id: str,
        capabilities: CapabilityProfile,
        local_context: Any = None,
    ) -> str:
        hinted = _algorithm_hint(local_context)
        if hinted is not None:
            return hinted
        lowered = activity_id.lower()
        if "inspect" in lowered or "sense" in lowered:
            return "dqn"
        if "stabilize" in lowered or "track" in lowered:
            return "actorcritic"
        return "ppo"

    def merge_trajectories(
        self,
        policy_id: str,
        algorithm_id: str,
        trajectories: list[Trajectory],
    ) -> int:
        policy = self._policies.setdefault(
            policy_id,
            _create_policy(policy_id, algorithm_id),
        )
        consumed = _merge_policy_trajectories(policy, trajectories)
        if consumed:
            policy.version += 1
        return consumed

    def reset(self) -> None:
        self._policies.clear()

class PersistentPolicyBank:

    def __init__(
        self,
        policies: dict[str, dict[str, Any]] | None = None,
        save_interval: int = 5,
        mappings_path: str | None = None,
        **kwargs: Any,
    ) -> None:
        self._policies: dict[str, InMemoryPolicy] = {}
        self._paths: dict[str, str] = {}
        self.save_interval = save_interval
        self._update_counts: dict[str, int] = {}
        self.mappings_path = mappings_path or "scratch/policy_mappings.pkl"
        self._activity_to_policy_map: dict[str, str] = {}
        self._activity_to_algorithm_map: dict[str, str] = {}

        if os.path.exists(self.mappings_path):
            try:
                with open(self.mappings_path, "rb") as f:
                    self._activity_to_policy_map = pickle.load(f)
                LOGGER.info(f"Loaded activity-to-policy mappings from {self.mappings_path}")
            except Exception as e:
                LOGGER.error(f"Failed to load activity-to-policy mappings from {self.mappings_path}: {e}")

        if policies:
            for policy_id, meta in policies.items():
                path = meta.get("path")
                if not path:
                    continue
                self._paths[policy_id] = path
                self._update_counts[policy_id] = 0

                if os.path.exists(path):
                    try:
                        with open(path, "rb") as f:
                            policy = pickle.load(f)
                        if isinstance(policy, InMemoryPolicy):
                            self._policies[policy_id] = policy
                            LOGGER.info(f"Loaded persistent policy {policy_id} from {path} (version {policy.version})")
                        else:
                            LOGGER.warning(f"File at {path} did not contain InMemoryPolicy; creating new one.")
                    except Exception as e:
                        LOGGER.error(f"Failed to load persistent policy from {path}: {e}")

                if policy_id not in self._policies:
                    algorithm = meta.get("algorithm_id") or policy_id.split(":", 1)[0]
                    self._policies[policy_id] = _create_policy(policy_id, algorithm)
                    self.save_policy(policy_id)

    def save_policy(self, policy_id: str) -> None:
        path = self._paths.get(policy_id)
        if not path:
            return
        policy = self._policies.get(policy_id)
        if not policy:
            return

        try:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            with open(path, "wb") as f:
                pickle.dump(policy, f, protocol=pickle.HIGHEST_PROTOCOL)
            LOGGER.info(f"Successfully persisted policy {policy_id} to {path} (version {policy.version})")
        except Exception as e:
            LOGGER.error(f"Failed to save policy {policy_id} to {path}: {e}")

    def save_mappings(self) -> None:
        if not self.mappings_path:
            return
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.mappings_path)), exist_ok=True)
            with open(self.mappings_path, "wb") as f:
                pickle.dump(self._activity_to_policy_map, f, protocol=pickle.HIGHEST_PROTOCOL)
            LOGGER.info(f"Successfully persisted activity-to-policy mappings to {self.mappings_path}")
        except Exception as e:
            LOGGER.error(f"Failed to save activity-to-policy mappings to {self.mappings_path}: {e}")

    def match_activity(
        self,
        activity_id: str,
        activity_type: str,
        capabilities: CapabilityProfile,
        algorithm_id: str | None = None,
    ) -> str:
        caps_str = _capability_key(capabilities)
        algorithm = algorithm_id or self._select_algorithm(activity_type, capabilities)
        policy_id = f"{algorithm}:{activity_type}:{caps_str}"
        self._activity_to_policy_map[activity_id] = policy_id
        self._activity_to_algorithm_map[activity_id] = algorithm

        if policy_id not in self._policies:
            self._policies[policy_id] = _create_policy(policy_id, algorithm)
            self._update_counts[policy_id] = 0

        self.save_mappings()
        return policy_id

    def get(self, descriptor: ActivityDescriptor) -> PolicySelection:
        policy_id, _ = self.index(descriptor)
        policy = self._policies[policy_id]
        return PolicySelection(
            algorithm_id=getattr(policy, "algorithm_id", "ppo"),
            policy=policy,
        )

    def index(self, descriptor: ActivityDescriptor) -> tuple[str, int]:
        policy_id = self._activity_to_policy_map.get(descriptor.activity_id)
        if not policy_id:
            algorithm = self._select_algorithm(
                descriptor.activity_id,
                descriptor.capabilities,
                descriptor.local_context,
            )
            policy_id = (
                f"{algorithm}:{descriptor.activity_id}:"
                f"{_capability_key(descriptor.capabilities)}"
            )
        if policy_id not in self._policies:
            algorithm = policy_id.split(":", 1)[0]
            self._policies[policy_id] = _create_policy(policy_id, algorithm)
            self._update_counts[policy_id] = 0
        policy = self._policies[policy_id]
        return policy.policy_id, policy.version

    def _select_algorithm(
        self,
        activity_id: str,
        capabilities: CapabilityProfile,
        local_context: Any = None,
    ) -> str:
        hinted = _algorithm_hint(local_context)
        if hinted is not None:
            return hinted
        lowered = activity_id.lower()
        if "inspect" in lowered or "sense" in lowered:
            return "dqn"
        if "stabilize" in lowered or "track" in lowered:
            return "actorcritic"
        return "ppo"

    def merge_trajectories(
        self,
        policy_id: str,
        algorithm_id: str,
        trajectories: list[Trajectory],
    ) -> int:
        if policy_id not in self._policies:
            self._policies[policy_id] = _create_policy(policy_id, algorithm_id)
            self._update_counts[policy_id] = 0
        policy = self._policies[policy_id]
        consumed = _merge_policy_trajectories(policy, trajectories)
        if consumed:
            policy.version += 1
            if policy_id in self._paths:
                self._update_counts[policy_id] += 1
                if self._update_counts[policy_id] % self.save_interval == 0:
                    self.save_policy(policy_id)
        return consumed

    def reset(self) -> None:
        self._policies.clear()
        self._update_counts.clear()
        for policy_id, path in self._paths.items():
            self._update_counts[policy_id] = 0
            if os.path.exists(path):
                try:
                    with open(path, "rb") as f:
                        policy = pickle.load(f)
                    if isinstance(policy, InMemoryPolicy):
                        self._policies[policy_id] = policy
                except Exception:
                    pass
            if policy_id not in self._policies:
                algorithm = policy_id.split(":", 1)[0]
                self._policies[policy_id] = _create_policy(policy_id, algorithm)

def _merge_policy_trajectories(
    policy: InMemoryPolicy,
    trajectories: list[Trajectory],
) -> int:
    consumed = 0
    for trajectory in trajectories:
        for element in trajectory.elements:
            step = _step_from_trajectory_element(element)
            policy.step_completed(step)
            consumed += 1
    return consumed

def _step_from_trajectory_element(element: TrajectoryElement) -> StepRecord:
    next_observation = getattr(element, "next_observation", None)
    return StepRecord(
        obs_t=element.observation,
        action_t=element.action,
        reward_t=element.reward,
        obs_next=next_observation,
        done=element.done,
        metadata=dict(element.metadata),
    )

def _algorithm_hint(local_context: Any) -> str | None:
    if not isinstance(local_context, dict):
        return None
    raw = (
        local_context.get("policy_algorithm")
        or local_context.get("algorithm_id")
        or local_context.get("algorithm")
    )
    if raw is None:
        return None
    normalized = str(raw).lower()
    if normalized == "a2c":
        return "actorcritic"
    if normalized in SUPPORTED_ALGORITHMS:
        return normalized
    return None

import typing
if typing.TYPE_CHECKING:
    from core.algorithms.shared_policy_manager import SharedPolicyBank
_istype_InMemoryPolicyBank: typing.Type[SharedPolicyBank] = InMemoryPolicyBank
_istype_PersistentPolicyBank: typing.Type[SharedPolicyBank] = PersistentPolicyBank

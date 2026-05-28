from __future__ import annotations

from dataclasses import dataclass, field
import heapq
import itertools
from typing import Any, Mapping

from core.algorithms.logical_controller import (
    AgentState,
    Assignment,
    AssignmentContext,
    AssignmentOptimizer,
    IndicativeComponent,
    RedistributionPlan,
)


@dataclass(frozen=True)
class _AssignmentAction:

    assignment: Assignment | None
    utility: float


@dataclass
class AStarRefocusAssignmentOptimizer:

    config: Any = None
    max_workload: float = 1.0
    switch_penalty: float = 0.10
    unassigned_penalty: float = 0.25
    demand_weight: float = 1.0
    fit_weight: float = 0.25
    minimum_improvement_margin: float = 0.01
    max_expansions: int = 10000
    capability_strengths: Mapping[str, float] = field(default_factory=dict)

    def solve(self, context: AssignmentContext) -> RedistributionPlan:
        agents = [
            agent
            for _, agent in sorted(context.snapshot.agents.items())
            if agent.workload <= self.max_workload
        ]
        baseline_score = self._assignment_utility(context, context.current_assignment)
        if not agents:
            return RedistributionPlan(
                assignments={},
                surrogate_score=baseline_score,
                baseline_score=baseline_score,
                monotonic=False,
            )

        normalized_demands = self._normalized_demands(context)
        normalized_strengths = self._normalized_agent_strengths(agents)
        actions = [
            self._actions_for_agent(
                context,
                agent,
                normalized_demands,
                normalized_strengths[agent.agent_id],
            )
            for agent in agents
        ]
        suffix_bounds = self._suffix_bounds(actions)
        assignments, candidate_score = self._astar_search(
            agents,
            actions,
            suffix_bounds,
        )
        monotonic = (
            candidate_score >= baseline_score + self.minimum_improvement_margin
        )
        return RedistributionPlan(
            assignments=assignments,
            surrogate_score=candidate_score,
            baseline_score=baseline_score,
            monotonic=monotonic,
        )

    def _actions_for_agent(
        self,
        context: AssignmentContext,
        agent: AgentState,
        normalized_demands: Mapping[str, float],
        normalized_strength: float,
    ) -> list[_AssignmentAction]:
        actions: list[_AssignmentAction] = []
        for component_id, component in sorted(context.snapshot.components.items()):
            utility = self._pair_utility(
                context,
                agent,
                component,
                normalized_demands.get(component_id, 0.0),
                normalized_strength,
            )
            if utility is None:
                continue
            actions.append(
                _AssignmentAction(
                    assignment=Assignment(
                        agent_id=agent.agent_id,
                        activity_id=agent.activity_id,
                        component_id=component.component_id,
                    ),
                    utility=utility,
                )
            )
        actions.append(
            _AssignmentAction(
                assignment=None,
                utility=self._unassigned_utility(agent),
            )
        )
        return sorted(actions, key=lambda item: item.utility, reverse=True)

    def _pair_utility(
        self,
        context: AssignmentContext,
        agent: AgentState,
        component: IndicativeComponent,
        normalized_demand: float,
        normalized_strength: float,
    ) -> float | None:
        if not component.required_capabilities.issubset(agent.capabilities.keys()):
            return None
        efficacy = context.efficacy_provider.score(agent, component)
        demand_multiplier = 1.0 + self.demand_weight * normalized_demand
        fit = max(0.0, 1.0 - abs(normalized_strength - normalized_demand))
        current = context.current_assignment.get(agent.agent_id)
        switch_cost = (
            self.switch_penalty
            if current is not None and current.component_id != component.component_id
            else 0.0
        )
        return demand_multiplier * efficacy + self.fit_weight * fit - switch_cost

    def _unassigned_utility(self, agent: AgentState) -> float:
        return -self.unassigned_penalty

    def _assignment_utility(
        self,
        context: AssignmentContext,
        assignments: Mapping[str, Assignment],
    ) -> float:
        eligible_agents = [
            agent
            for _, agent in sorted(context.snapshot.agents.items())
            if agent.workload <= self.max_workload
        ]
        normalized_demands = self._normalized_demands(context)
        normalized_strengths = self._normalized_agent_strengths(eligible_agents)
        score = 0.0
        for agent in eligible_agents:
            assignment = assignments.get(agent.agent_id)
            if assignment is None:
                score += self._unassigned_utility(agent)
                continue
            component = context.snapshot.components.get(assignment.component_id)
            if component is None:
                score += self._unassigned_utility(agent)
                continue
            utility = self._pair_utility(
                context,
                agent,
                component,
                normalized_demands.get(component.component_id, 0.0),
                normalized_strengths[agent.agent_id],
            )
            score += self._unassigned_utility(agent) if utility is None else utility
        return score

    def _normalized_demands(
        self,
        context: AssignmentContext,
    ) -> dict[str, float]:
        max_demand = max(
            (max(0.0, component.demand) for component in context.snapshot.components.values()),
            default=0.0,
        )
        if max_demand <= 0.0:
            return {component_id: 0.0 for component_id in context.snapshot.components}
        return {
            component_id: max(0.0, component.demand) / max_demand
            for component_id, component in context.snapshot.components.items()
        }

    def _normalized_agent_strengths(
        self,
        agents: list[AgentState],
    ) -> dict[str, float]:
        raw = {agent.agent_id: self._agent_strength(agent) for agent in agents}
        max_strength = max(raw.values(), default=0.0)
        if max_strength <= 0.0:
            return {agent_id: 0.0 for agent_id in raw}
        return {
            agent_id: max(0.0, strength) / max_strength
            for agent_id, strength in raw.items()
        }

    def _agent_strength(self, agent: AgentState) -> float:
        for key in ("capacity", "strength", "size"):
            value = agent.metadata.get(key)
            if isinstance(value, (int, float)):
                return float(value)
        for key in ("extinguisher_capacity", "capacity", "strength", "size"):
            value = agent.capabilities.get(key)
            if value > 0.0:
                return float(value)
        if self.capability_strengths:
            return sum(
                float(self.capability_strengths.get(capability, 1.0))
                for capability in agent.capabilities.keys()
            )
        return float(max(1, len(agent.capabilities)))

    def _suffix_bounds(
        self,
        actions: list[list[_AssignmentAction]],
    ) -> list[float]:
        suffix = [0.0] * (len(actions) + 1)
        for index in range(len(actions) - 1, -1, -1):
            best = max((action.utility for action in actions[index]), default=0.0)
            suffix[index] = best + suffix[index + 1]
        return suffix

    def _astar_search(
        self,
        agents: list[AgentState],
        actions: list[list[_AssignmentAction]],
        suffix_bounds: list[float],
    ) -> tuple[dict[str, Assignment], float]:
        counter = itertools.count()
        queue: list[
            tuple[
                float,
                float,
                int,
                int,
                dict[str, Assignment],
            ]
        ] = []
        heapq.heappush(
            queue,
            (-suffix_bounds[0], 0.0, 0, next(counter), {}),
        )
        best_assignments: dict[str, Assignment] = {}
        best_score = float("-inf")
        expansions = 0

        while queue and expansions < self.max_expansions:
            negative_bound, negative_score, depth, _, partial = heapq.heappop(queue)
            bound = -negative_bound
            score = -negative_score
            if bound < best_score:
                continue
            if depth == len(agents):
                if score > best_score:
                    best_score = score
                    best_assignments = partial
                break

            agent = agents[depth]
            for action in actions[depth]:
                next_score = score + action.utility
                next_partial = dict(partial)
                if action.assignment is not None:
                    next_partial[agent.agent_id] = action.assignment
                next_depth = depth + 1
                next_bound = next_score + suffix_bounds[next_depth]
                if next_bound < best_score:
                    continue
                heapq.heappush(
                    queue,
                    (
                        -next_bound,
                        -next_score,
                        next_depth,
                        next(counter),
                        next_partial,
                    ),
                )
            expansions += 1

        if best_score == float("-inf"):
            return {}, sum(self._unassigned_utility(agent) for agent in agents)
        return best_assignments, best_score


@dataclass
class GreedyAssignmentOptimizer:

    config: Any = None
    max_workload: float = 1.0
    capability_strengths: Mapping[str, float] = field(default_factory=dict)
    unassigned_penalty: float = 0.25
    minimum_improvement_margin: float = 0.0

    def solve(self, context: AssignmentContext) -> RedistributionPlan:
        agents = [
            agent
            for _, agent in sorted(context.snapshot.agents.items())
            if agent.workload <= self.max_workload
        ]
        ordered_agents = sorted(
            agents,
            key=lambda agent: (self._agent_strength(agent), agent.agent_id),
            reverse=True,
        )
        ordered_components = sorted(
            context.snapshot.components.values(),
            key=lambda component: (component.demand, component.component_id),
            reverse=True,
        )

        assignments: dict[str, Assignment] = {}
        for agent in ordered_agents:
            component = self._first_feasible_component(
                context,
                agent,
                ordered_components,
            )
            if component is None:
                continue
            assignments[agent.agent_id] = Assignment(
                agent_id=agent.agent_id,
                activity_id=agent.activity_id,
                component_id=component.component_id,
            )

        baseline_score = self._assignment_score(context, context.current_assignment)
        candidate_score = self._assignment_score(context, assignments)
        return RedistributionPlan(
            assignments=assignments,
            surrogate_score=candidate_score,
            baseline_score=baseline_score,
            monotonic=(
                candidate_score >= baseline_score + self.minimum_improvement_margin
            ),
        )

    def _first_feasible_component(
        self,
        context: AssignmentContext,
        agent: AgentState,
        components: list[IndicativeComponent],
    ) -> IndicativeComponent | None:
        for component in components:
            if not component.required_capabilities.issubset(agent.capabilities.keys()):
                continue
            if context.efficacy_provider.score(agent, component) <= 0.0:
                continue
            return component
        return None

    def _assignment_score(
        self,
        context: AssignmentContext,
        assignments: Mapping[str, Assignment],
    ) -> float:
        score = 0.0
        eligible_agent_ids = {
            agent_id
            for agent_id, agent in context.snapshot.agents.items()
            if agent.workload <= self.max_workload
        }
        for agent_id in eligible_agent_ids:
            assignment = assignments.get(agent_id)
            if assignment is None:
                score -= self.unassigned_penalty
                continue
            agent = context.snapshot.agents.get(assignment.agent_id)
            component = context.snapshot.components.get(assignment.component_id)
            if agent is None or component is None:
                score -= self.unassigned_penalty
                continue
            if not component.required_capabilities.issubset(agent.capabilities.keys()):
                score -= self.unassigned_penalty
                continue
            efficacy = context.efficacy_provider.score(agent, component)
            score += max(0.0, component.demand) * efficacy
        return score

    def _agent_strength(self, agent: AgentState) -> float:
        for key in ("capacity", "strength", "size"):
            value = agent.metadata.get(key)
            if isinstance(value, (int, float)):
                return float(value)
        for key in ("extinguisher_capacity", "capacity", "strength", "size"):
            value = agent.capabilities.get(key)
            if value > 0.0:
                return float(value)
        if self.capability_strengths:
            return sum(
                float(self.capability_strengths.get(capability, 1.0))
                for capability in agent.capabilities.keys()
            )
        return float(max(1, len(agent.capabilities)))


import typing

if typing.TYPE_CHECKING:
    from core.algorithms.logical_controller import AssignmentOptimizer

_istype_AStarRefocusAssignmentOptimizer: typing.Type[AssignmentOptimizer] = (
    AStarRefocusAssignmentOptimizer
)
_istype_GreedyAssignmentOptimizer: typing.Type[AssignmentOptimizer] = (
    GreedyAssignmentOptimizer
)


from __future__ import annotations

import argparse

from .config import RuntimeConfig
from .entities import Agent, LogicalController, PolicyManager
from .udp_factory import UdpSocketEntityFactory


def build_entity(
    config: RuntimeConfig,
    role: str,
    entity_id: str,
    realization: str,
) -> LogicalController | Agent | PolicyManager:
    if realization != "udp":
        raise ValueError(f"unsupported realization: {realization}")

    factory = UdpSocketEntityFactory(config=config)
    if role == "logical_controller":
        return factory.create_logical_controller(entity_id)
    if role == "agent":
        return factory.create_agent(entity_id)
    if role == "policy_manager":
        return factory.create_policy_manager(entity_id)
    raise ValueError(f"unsupported entity role: {role}")

def run_entity(entity: LogicalController | Agent | PolicyManager) -> None:
    entity.start()
    try:
        entity.wait()
    finally:
        entity.stop()

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one runtime entity process.")
    parser.add_argument("--role", required=True)
    parser.add_argument("--realization", default="udp")
    parser.add_argument("--config", required=True)
    parser.add_argument("--entity-id", required=True)
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    config = RuntimeConfig.from_json_file(args.config)
    entity = build_entity(
        config=config,
        role=args.role,
        entity_id=args.entity_id,
        realization=args.realization,
    )
    run_entity(entity)

if __name__ == "__main__":
    main()

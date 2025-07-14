# running intervention / control on agentcollection
import os
import json
import pathlib
from typing import List
from datetime import datetime

import hydra
from omegaconf import DictConfig

from diary.entity.agent import Agent, AgentCollection

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)

@hydra.main(
    config_path=os.path.join(
        ROOT_DIR, "diary", "config", "treatment",
    ),
    config_name="config",
    version_base="1.3",
)
def main(cfg: DictConfig):
    
    cfg = cfg.treatment
    datetime_init = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"--> main(): Running {cfg.treatment_name} with config: {cfg}")
    
    with open(os.path.join(ROOT_DIR, cfg.environment_filepath), "r") as f:
        treatment_metadata = json.load(f)
    print(f"--> main(): running treatment with metadata: {treatment_metadata}")
    
    agent_filepath = os.path.join(
        ROOT_DIR, cfg.agent_params.agent_filepath)
    assert agent_filepath == 'None' or os.path.exists(agent_filepath), (
        "Agent must be providied. "
        f"Current agent filepath: {agent_filepath}"
    )
    agents = AgentCollection.from_jsonl(agent_filepath)
    print(f"--> main(): Loaded {len(agents.agents)} agents.")
    print(agents)
    
    agents.rollout(
        environment_params=treatment_metadata,
        response_sampling_params=cfg.sampling_params,
        critic_params=cfg.critic_params,
        agent_params=cfg.agent_params,
        interview_params=cfg.interview_params,
        system_params=cfg.system_params,
        op="treatment"
    )

    output_filepath = os.path.join(
        ROOT_DIR, cfg.output_filepath_template.format(
            treatment_name=cfg.treatment_name,
            source=(
                str(os.path.splitext(
                    os.path.basename(cfg.agent_params.agent_filepath))[0]
                ) if cfg.agent_params.agent_filepath != 'None' else "new"
            ),
            timestamp=datetime_init
        )
    )
    print(f"--> main(): Saving agents to {output_filepath}")
    agents.to_jsonl(output_filepath)

if __name__ == "__main__":
    main()
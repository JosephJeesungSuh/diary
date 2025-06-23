# running identity survey operations on agents
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
        ROOT_DIR, "diary", "config", "identity_survey",
    ),
    config_name="config",
    version_base="1.3",
)
def main(cfg: DictConfig):
    
    datetime_init = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"--> main(): Running narrative generation with config: {cfg}")
    
    with open(os.path.join(ROOT_DIR, cfg.question_filepath), "r") as f:
        questions = json.load(f)
    qkeys = [q['qkey'] for q in questions]
    print(f"--> main(): running survey on questions: {qkeys}")

    agent_filepath = os.path.join(
        ROOT_DIR, cfg.agent_params.agent_filepath)
    assert os.path.exists(agent_filepath), (
        "Agent must be providied. "
        f"Current agent filepath: {agent_filepath}"
    )
    agents = AgentCollection.from_jsonl(agent_filepath)
    print(f"--> main(): Loaded {len(agents.agents)} agents.")
    print(agents)
    
    agents.rollout(
        identity_query_prompt=questions,
        response_sampling_params=cfg.sampling_params,
        critic_params=cfg.critic_params,
        agent_params=cfg.agent_params,
        interview_params=cfg.interview_params,
        op="identity"
    )

    output_filepath = os.path.join(
        ROOT_DIR, cfg.output_filepath_template.format(
            timestamp=datetime_init
        )
    )
    print(f"--> main(): Saving agents to {output_filepath}")
    agents.to_jsonl(output_filepath)

if __name__ == "__main__":
    main()
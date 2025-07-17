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
        ROOT_DIR, "diary", "config", "narrative_gen",
    ),
    config_name="config",
    version_base="1.3",
)
def main(cfg: DictConfig):
    
    # initiate: load the config
    datetime_init = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"--> main(): Running narrative generation with config: {cfg}")
    with open(os.path.join(ROOT_DIR, cfg.question_filepath), "r") as f:
        questions = json.load(f)

    # initialize agent collection
    if cfg.agent_params.agent_filepath != 'None':
        agents = AgentCollection.from_jsonl(
            filepath=os.path.join(
                ROOT_DIR, cfg.agent_params.agent_filepath
            )
        )
    else:
        agents = AgentCollection(
            agents=[
                Agent() for _ in range(cfg.agent_params.n_agent)
            ]
        )
    
    # run the narrative generation on the agent collection
    print(f"--> main(): Running narrative gen on {len(agents)} agents.")
    agents.rollout(
        continuation_prompt=questions,
        response_sampling_params=cfg.sampling_params,
        critic_params=cfg.critic_params,
        agent_params=cfg.agent_params,
        interview_params=cfg.interview_params,
        op="narrative"
    )

    # save
    output_filepath = os.path.join(
        ROOT_DIR, cfg.output_filepath_template.format(
            question=cfg.question_filepath.split("/")[-1].split(".")[0],
            modelname=cfg.sampling_params.model_name.replace("/", "--"),
            timestamp=datetime_init
        )
    )
    print(f"--> main(): Saving agents to {output_filepath}")
    agents.to_jsonl(output_filepath)

if __name__ == "__main__":
    main()
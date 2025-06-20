import os
import json
import pathlib
import warnings
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple, Union, final, Literal, Any

import numpy as np

from diary.stats.registered_ops import Operation
from diary.utils.progress import Progress
from diary.utils.intervention import Intervention

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)


class Action:

    def __init__(self):
        pass


class History:

    def __init__(self):
        # an event is consisted of three components:
        # 1. action: the action taken by experimenter
        # 2. response: the response from the agent
        # 3. timestamp: the time when the action was taken
        self.history_list: List[Tuple[Action, str, datetime]] = []


class Agent:
    
    def __init__(self):
        self.history: History = History()
        self.progress: Progress = Progress()
        self.created_timestamp: datetime = datetime.now()
        self._assigned_to: Intervention = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Agent":
        assert isinstance(data, dict), "Data must be a dictionary."
        agent = cls()
        if "history" in data:
            agent.history = History.from_dict(data["history"])
        if "progress" in data:
            agent.progress = Progress.from_dict(data["progress"])        
        if "created_timestamp" in data:
            agent.created_timestamp = datetime.fromisoformat(
                data["created_timestamp"])
        if "assigned_to" in data:
            agent._assigned_to = Intervention.from_dict(
                data["assigned_to"])
        return agent

    @property
    def assigned_to(self) -> Optional[Intervention]:
        return self._assigned_to
    
    @assigned_to.setter
    def assigned_to(self, intervention: Intervention):
        assert self._assigned_to is None, (
            "Agent already assigned to an intervention."
        )
        self._assigned_to = intervention    
        

class AgentCollection:
    
    def __init__(self, agents: List[Agent]):
        assert isinstance(agents, list)
        self.agents = agents

    @classmethod
    def from_dict(cls, data: List[Dict]) -> "AgentCollection":
        agents = [Agent.from_dict(agent_data) for agent_data in data]
        return cls(agents)
    
    @classmethod
    def from_jsonl(cls, filepath: str) -> "AgentCollection":
        assert os.path.exists(filepath), f"File {filepath} does not exist."
        agents = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line_number, line in enumerate(f, 1):
                try:
                    data = json.loads(line)
                    agents.append(data)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Error decoding JSON on line {line_number}: {e}")
        return cls.from_dict(agents)
    
    def shuffle(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            np.random.seed(seed)
            np.random.shuffle(self.agents)
    
    def return_stats(self, op: Operation):
        assert isinstance(op, Operation)
        return op.perform_on(self.agents)
    
    def __len__(self) -> int:
        return len(self.agents)
    
    def __iter__(self):
        for agent in self.agents:
            yield agent
    
    def __getitem__(self,
                  idx: Union[int, slice]
                  ) -> Union["Agent", "AgentCollection"]:
        assert isinstance(idx, (int, slice)), (
            f"Index must be an int or slice, got {type(idx).__name__}"
        )
        if isinstance(idx, int):
            return self.agents[idx]
        if isinstance(idx, slice):
            new = self.__class__.__new__(self.__class__)
            new.agents = self.agents[idx]
            return new
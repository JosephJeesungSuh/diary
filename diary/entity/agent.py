import os
import json
import pathlib
import warnings
import multiprocessing as mp
from datetime import datetime
from typing import Optional, Dict, List, Union, Any

import numpy as np
from tqdm import tqdm

from diary.stats.registered_ops import Operation
from diary.entity.progress import Progress
from diary.entity.intervention import Intervention
from diary.entity.history import History, Event
from diary.entity.identity import Identity, Attribute
from diary.llm_engine.llm_engine import LLMEngine
from diary.utils.misc_utils import class_to_dict, random_hashtag
from diary.operation import generate_narrative, query_identity

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)


class Agent:
    
    def __init__(self):
        self.id: str = random_hashtag()
        self.history: History = History()
        self.identity: Identity = Identity(owner=self.id)
        self.progress: Progress = Progress()
        self.created_timestamp: str = datetime.now().isoformat()
        self._assigned_to: Intervention = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Agent":
        assert isinstance(data, dict), "Data must be a dictionary."
        agent = cls()
        for k,v in data.items():
            if hasattr(agent, k):
                attr = getattr(agent, k)
                if hasattr(attr, "from_dict"):
                    setattr(agent, k, type(attr).from_dict(v))
                else:
                    setattr(agent, k, v)
            else:
                raise ValueError("Unexpected key in deserializing Agent.")
        return agent
    
    def update_history(self, event: Event) -> None:
        assert isinstance(event, Event)
        self.history.update(event)

    def update_identity(self, identity: Attribute) -> None:
        assert isinstance(identity, Attribute)
        self.identity.update(identity)
    
    def rollout(self, **kwargs) -> None:
        rollout_kwargs = kwargs.copy()
        op: str = rollout_kwargs.pop("op", None)     
        assert op in ["narrative", "identity", "diary"], (
            f"--> Agent.rollout(): Invalid rollout operation {op}."
        )
        if op == "narrative":
            continuation_prompt = rollout_kwargs.pop("continuation_prompt", None)
            assert continuation_prompt is not None
            if isinstance(continuation_prompt, str):
                continuation_prompt = [continuation_prompt]
            for prompt in continuation_prompt:
                query, response = generate_narrative( # two events
                    history=self.history,
                    continuation_prompt=prompt,
                    **rollout_kwargs,
                )
                self.update_history(query)
                self.update_history(response)
        elif op == "identity":
            query_prompts: List[Dict] = (
                rollout_kwargs.pop("identity_query_prompt", None)
            )
            assert query_prompts is not None
            if isinstance(query_prompts, dict):
                query_prompts = [query_prompts]
            for query in query_prompts:
                new_attribute, event = query_identity(
                    history=self.history,
                    query=query,
                    **rollout_kwargs,
                )
                self.update_identity(new_attribute)
                self.update_history(event)
        else:
            raise NotImplementedError
        return

    @property
    def assigned_to(self) -> Optional[Intervention]:
        return self._assigned_to
    
    @assigned_to.setter
    def assigned_to(self, intervention: Intervention):
        assert self._assigned_to is None, (
            "Agent already assigned to an intervention."
        )
        self._assigned_to = intervention

    def _sanity_check(self) -> None:
        assert self.identity.owner_id == self.id
        

class AgentCollection:
    
    def __init__(self, agents: List[Agent]):
        assert isinstance(agents, list)
        self.agents = agents

    @classmethod
    def from_dictlist(cls, data: List[Dict]) -> "AgentCollection":
        assert isinstance(data, list)
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
        return cls.from_dictlist(agents)

    def to_jsonl(self,
                 filepath: Union[str, pathlib.Path]) -> None:
        assert isinstance(filepath, (str, pathlib.Path))
        if os.path.exists(filepath):
            raise FileExistsError(f"File {filepath} already exists.")
        
        if isinstance(filepath, str):
            filepath = pathlib.Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            for agent in self.agents:
                f.write(json.dumps(class_to_dict(agent)) + '\n')
        return

    def rollout(self, **kwargs) -> None:
        
        rollout_kwargs = kwargs.copy()
        n_parallel: int = rollout_kwargs.pop("n_parallel", 1)
        response_engine: LLMEngine = LLMEngine(
            llm_config=rollout_kwargs.pop("response_sampling_params", None)
        )
        critic_engine: LLMEngine = LLMEngine(
            llm_config=rollout_kwargs.pop("critic_params", None)
        )
        print(f"--> AgentCollection.rollout(): initialized LLMEngine.")

        def _worker(agent: Agent) -> None:
            try:
                agent.rollout(
                    **rollout_kwargs,
                    response_engine=response_engine,
                    critic_engine=critic_engine)
            except Exception as exc:
                warnings.warn(
                    f"[Agent {getattr(agent, 'id', 'N/A')}] rollout failed: {exc}"
                )
                return 1

        with tqdm(
            total=len(self.agents), desc="Agent Rollout",
        ) as pbar:
            if n_parallel <= 1:
                for agent in self.agents:
                    _worker(agent); pbar.update(1)
            else:
                with mp.pool.ThreadPool(n_parallel) as pool:
                    for _ in pool.imap_unordered(_worker, self.agents):
                        pbar.update(1)        
    
    def shuffle(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            np.random.seed(seed)
            np.random.shuffle(self.agents)
    
    def return_stats(self, op: Operation):
        assert isinstance(op, Operation)
        return op.perform_on(self.agents)
    
    def _run_sanity_checks(self) -> None:
        ids = self._get_agent_ids()
        assert len(ids) == len(set(ids)), (
            "AgentCollection contains duplicate agent IDs."
        )
    
    def _get_agent_ids(self) -> List[str]:
        return [agent.id for agent in self.agents]
        
    def __eq__(self, other) -> bool:
        return isinstance(other, AgentCollection) and \
            self._get_agent_ids() == other._get_agent_ids()

    def __repr__(self) -> str:
        repr_str = f"AgentCollection(n={len(self.agents)} agents)"
        repr_str += "\n" + "=" * 40 + "\n"
        repr_str += f"0th agent history [id={self.agents[0].id}] (for inspection):"
        repr_str += "\n" + "=" * 40 + "\n"
        repr_str += self.agents[0].history.format_string()
        return repr_str
    
    def __add__(self, other: "AgentCollection") -> "AgentCollection":
        assert isinstance(other, AgentCollection), (
            f"Cannot add {type(other).__name__} to AgentCollection."
        )
        new = self.__class__.__new__(self.__class__)
        new.agents = self.agents + other.agents
        new._run_sanity_checks()
        return new
    
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
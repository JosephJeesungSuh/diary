import os
import json
import pathlib
import warnings
import asyncio
from datetime import datetime
from typing import Optional, Dict, List, Union, Any

import numpy as np
from tqdm import tqdm

from diary.stats.stats import Operation
from diary.entity.progress import Progress
from diary.entity.intervention import Intervention
from diary.entity.history import History, Event
from diary.entity.identity import Identity, Attribute
from diary.llm_engine.llm_engine import LLMEngine
from diary.utils.misc_utils import (
    class_to_dict,
    random_hashtag,
    extract_placeholder
)
from diary.operation import (
    generate_narrative,
    query_identity,
    run_intervention
)
from diary.operation.system_prompt import (
    MESSAGING_SYSTEM_DESCRIPTION,
    MESSAGING_RECEIVE_DESCRIPTION
)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)


class Agent:
    
    def __init__(self, id: Optional[str] = None):
        self.id: str = random_hashtag() if id is None else id
        self.history: History = History()
        self.identity: Identity = Identity(owner=self.id)
        self.progress: Progress = Progress()
        self.created_timestamp: str = datetime.now().isoformat()
        self._assigned_to: Intervention = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Agent":
        assert isinstance(data, dict), "Data must be a dictionary."
        agent = cls(id=data.get("id", None))
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

    def update_identity(self, attribute: Attribute) -> None:
        assert isinstance(attribute, Attribute)
        self.identity.update(attribute)
    
    def update_progress(self, env: Dict) -> None:
        assert Progress.convert_timestamp_to_day(
            env.get("study_duration", "00-00-00")
        ) > 0
        if self.progress.progress == []:
            self.progress = Progress(env)
        else:
            self.progress.update(env)

    def update_assignment(self, env: Dict) -> None:
        assert isinstance(env, dict)
        self.assigned_to = Intervention(env)
        
    def rollout(self, **kwargs) -> None:
        rollout_kwargs = kwargs.copy()
        op: str = rollout_kwargs.pop("op", None)     
        assert op in ["narrative", "identity", "treatment"], (
            f"--> Agent.rollout(): Invalid rollout operation {op}."
        )
        if op == "narrative":
            """
            Semantics of narrative generation rollout:
            1. Load a list of 'interview questions' to continuation_prompt
            2. For each 'question', generate a response conditioned on history
            3. Update agent history with the 'question' (entity = query_module)
               and the response (entity = agent)
            """
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
            """
            Semantics of identity query rollout:
            1. Load a list of 'identity query questions' to query_prompts
             - metadata (qkey, question body, categories) defined in query_prompts
            2. For each 'question', generate an Attribute and update identity
             - event is a null event (entity = "", action = "") indicating
               the identity survey is performed but not added to textual history
            """
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
        elif op == "treatment":
            """
            Semantics of treatment rollout:
            """
            env: Dict = rollout_kwargs.pop("environment_params", None)
            assert env is not None
            # register the intervention environment to Progress and Intervention
            self.update_progress(env)
            self.update_assignment(env)
            while self.progress.is_finished() is False:
                # advance the progress by one timestep
                # temporal advance info ([Day 1]) is recorded as event.
                ops_to_invoke: List[str] = self.progress.forward()                
                self.update_history(
                    Event(
                        entity="",
                        action=self.progress.current_printable(),
                    )
                )
                for op_name in ops_to_invoke:
                    # each invoked operation is consisted of:
                    # 1. prompt
                    # 2. who invokes (system or interviewer)
                    # 3. whether it requires a response from agent
                    prompt, entity, critic_fn, require_response = \
                        self._assigned_to.return_ops(op_name)
                    # sometimes prompt is contextualized,
                    # ex. with agent's name or the current status (how many weeks left)
                    placeholders = extract_placeholder(prompt)
                    if placeholders:
                        info = [(p, self.agent_information(p)) for p in placeholders]
                        prompt = prompt.format(**dict(info))
                    if require_response:
                        # run the sampling engine to make agent rollout
                        query, response = run_intervention(
                            intervention=self._assigned_to,
                            history=self.history,
                            continuation_prompt=prompt,
                            critic_fn=critic_fn,
                            **rollout_kwargs,
                        )
                        self.update_history(query)
                        self.update_history(response)
                    else:
                        if entity == "interview":
                            # if it is invoked by interviewer but not requires response
                            # just record the prompt as an event.
                            self.update_history(
                                Event(
                                    entity=rollout_kwargs.get("interview_params").entity,
                                    action=prompt,
                                    metadata=None,
                                )
                            )
                        elif entity == "system":
                            # invoked by the system (ex. regular reminder)
                            self.update_history(
                                Event(
                                    entity=rollout_kwargs.get("system_params").entity,
                                    action=MESSAGING_SYSTEM_DESCRIPTION[op_name],
                                    metadata=None,
                                )
                            )
                            self.update_history(
                                Event(
                                    entity=rollout_kwargs.get("agent_params").entity,
                                    action=MESSAGING_RECEIVE_DESCRIPTION[op_name].format(
                                        prompt=prompt
                                    ),
                                    metadata=None,
                                )
                            )
                        else:
                            raise ValueError(f"Invalid entity {entity} for Agent.rollout().")
                    self.progress.update_progress(
                        (op_name, self.progress.current)
                    )
        else:
            raise ValueError(f"Invalid operation {op} for Agent.rollout().")
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

    def agent_information(self, query: str) -> Any:
        if query == "remain_week":
            return self.progress.remaining_weeks()
        raise NotImplementedError
        

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
        interview_params: Dict = rollout_kwargs.get("interview_params", None)
        n_parallel: int = interview_params.get("n_parallel", 1)
        response_engine: LLMEngine = LLMEngine(
            llm_config=rollout_kwargs.pop("response_sampling_params", None)
        )
        critic_engine: LLMEngine = LLMEngine(
            llm_config=rollout_kwargs.pop("critic_params", None)
        )
        print(f"--> AgentCollection.rollout(): initialized LLMEngine.")

        async def _async_worker(agent: Agent):
            try:
                await asyncio.to_thread(
                    agent.rollout,
                    **rollout_kwargs,
                    response_engine=response_engine,
                    critic_engine=critic_engine
                )
                print(f"--> Agent {getattr(agent, 'id', 'N/A')} rollout complete")
            except Exception as exc:
                warnings.warn(
                    f"[Agent {getattr(agent, 'id', 'N/A')}] rollout failed: {exc}"
                )
                return 1
            
        async def _run_all():
            semaphore = asyncio.Semaphore(n_parallel)
            async def sem_task(agent: Agent):
                async with semaphore:
                    await _async_worker(agent)
                    pbar.update(1)
            tasks = [asyncio.create_task(sem_task(agent)) for agent in self.agents]
            await asyncio.gather(*tasks)

        with tqdm(
            total=len(self.agents), desc="Agent Rollout",
        ) as pbar:
            asyncio.run(_run_all())
    
    def shuffle(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            np.random.seed(seed)
        np.random.shuffle(self.agents)
    
    def return_stats(self, op: Operation, **kwargs):
        assert isinstance(op, Operation)
        return op.perform_on(
            identity_list=[agent.identity for agent in self.agents],
            **kwargs
        )
    
    def filter_to(self,
               target_demographics: np.ndarray,
               demographics_metadata: Dict[str, List[str]],
               **kwargs) -> "AgentCollection":
        """
        Demographic filtering of agent collection based on bipartite matching.
        Args:
            target_demographics: 2D array of shape (n_population, n_features)
             - each row is a demographic vector of individuals
             - each entry is an interger representing a feature
            demographics_metadata: dictionary containing
             1. List of feature names in "featureslist"
             2. for each feature name, a list of categories in "{feature_name}"
            kwargs:
             matching_scheme: str = {"hungarian", "greedy"}
        """
        assert len(self.agents) < target_demographics.shape[0], (
            "Requires (number of agents) >= (number of individuals)"
        )
        raise NotImplementedError
    
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
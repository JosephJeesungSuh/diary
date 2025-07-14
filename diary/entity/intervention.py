# a sytem managing intervention

import random
from typing import Dict, Any, List, Tuple, Optional, Union

from diary.utils.misc_utils import class_to_dict

class Intervention:
    
    def __init__(self, env: Optional[Dict] = None):
        self.intervention : Optional[str] = None
        self.intervention_uid : Optional[str] = None
        self.static_ops : Dict[str, Dict[str, Union[List[str], bool]]] = {}
        self.conditional_ops: Dict[str, Dict[str, Union[List[str], bool]]] = {}
        if env is not None:
            self.update_intervention(env)

    def update_intervention(self, env: Dict):
        assert isinstance(env, dict)
        self.intervention = env.get('treatment_name', None)
        self.intervention_uid = env.get('treatment_uid', None)
        for op_name, op_info in env.get('ops', {}).items():
            if op_name.startswith('COND'):
                self.conditional_ops[op_info['trigger']] = {
                    "formats": op_info['formats'],
                    "require_response": op_info['require_response'],
                    "entity": op_info['entity'],
                    "critic_fn": op_info["critic_fn"]
                }
            else:
                self.static_ops[op_name] = {
                    "formats": op_info['formats'],
                    "require_response": op_info['require_response'],
                    "entity": op_info['entity'],
                    "critic_fn": op_info["critic_fn"]
                }
        return
    
    def return_ops(self, invoke: str) -> Tuple[str, str, List[str], bool]:
        if invoke.startswith('TIME') or invoke.startswith('FREQ'):
            candidates = self.static_ops.get(invoke, None)
            assert candidates is not None
            return (
                random.choice(candidates['formats']),
                candidates['entity'],
                candidates['critic_fn'],
                candidates['require_response']
            )
        if invoke.startswith('<') or invoke.endswith('>'):
            candidates = self.conditional_ops.get(invoke, None)
            assert candidates is not None
            return (
                random.choice(candidates['formats']),
                candidates['entity'],
                candidates['critic_fn'],
                candidates['require_response']
            )

    @classmethod
    def from_dict(cls, env: Dict) -> 'Intervention':
        return cls(env)

    def serialize(self) -> Dict:
        return class_to_dict(self)
    
    def __repr__(self) -> str:
        repr_str = ""
        repr_str += f"Intervention: {self.intervention}\n"
        repr_str += f"Intervention UID: {self.intervention_uid}\n"
        repr_str += "Static Operations:\n"
        for op_name, formats in self.static_ops.items():
            repr_str += f"  {op_name}: {formats}\n"
        repr_str += "Conditional Operations:\n"
        for trigger, formats in self.conditional_ops.items():
            repr_str += f"  {trigger}: {formats}\n"
        return repr_str
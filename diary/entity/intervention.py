# a warpper for intervention related functions

from typing import Dict, Any, List, Tuple, Optional

from diary.utils.misc_utils import class_to_dict

class Intervention:
    
    def __init__(self):
        self.intervention = None

    def serialize(self) -> Dict:
        return class_to_dict(self)
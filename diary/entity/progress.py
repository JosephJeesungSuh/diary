# progress.py deals with the temporal progress of the diary.
# it is a wrapper for the temporal information.

from typing import Dict, Any, List, Tuple, Optional

from diary.utils.misc_utils import class_to_dict

class Progress:
    def __init__(self):
        self.progress = []

    def serialize(self) -> Dict:
        return class_to_dict(self)
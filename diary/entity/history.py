from typing import List, Dict, Any, Union

class Event:

    def __init__(self,
                 entity: str = "",
                 action: str = "",
                 metadata: Union[Dict[str, Any], None] = None):
        self.entity: str = entity
        self.action: str = action
        self.metadata: Dict[str, Any] = metadata if metadata is not None else {}
    
    def format_string(self) -> str:
        return f"{self.entity}{self.action}"

class History:

    def __init__(self):
        self.history_list: List[Event] = []

    def format_string(self) -> str:
        return "\n\n".join(
            event.format_string() for event in self.history_list
        )
    
    def update(self, event: Event) -> None:
        assert isinstance(event, Event)
        self.history_list.append(event)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "History":
        assert isinstance(data, dict), "Data must be a dictionary."
        history = cls()
        for k, v in data.items():
            if hasattr(history, k):
                if isinstance(v, list):
                    history.history_list = [Event(**item) for item in v]
                else:
                    setattr(history, k, v)
            else:
                raise ValueError("Unexpected key in unserializing History.")
        return history
    
    @classmethod
    def concat(cls, h1: "History", h2: "History") -> "History":
        new = cls()
        new.history_list = h1.history_list + h2.history_list
        return new

    def __add__(self, other: "History") -> "History":
        assert isinstance(other, History)
        return self.concat(other)
    
    def __len__(self) -> int:
        return len(self.history_list)
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
    def concat(cls, h1: "History", h2: "History") -> "History":
        new = cls()
        new.history_list = h1.history_list + h2.history_list
        return new

    def __add__(self, other: "History") -> "History":
        assert isinstance(other, History)
        return self.concat(other)
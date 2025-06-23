from typing import List, Dict, Any, Union, Literal, Optional

import numpy as np

class Attribute:
    
    def __init__(self,
                 attribute: str = None,
                 category: List[str] = None,
                 obtained_from: Literal["query", "critic"] = None,
                 stats: List[float] = None,
                 raw_data: List[str] = None,
                 run_check: bool = True):
        self.attribute: str = attribute
        self.category: List[str] = category
        self.obtained_from: Literal[
            "query", "critic"
        ] = obtained_from
        self.stats: List[float] = stats
        self.raw_data: List[str] = raw_data
        if run_check:
            self._sanity_check()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Attribute":
        assert isinstance(data, dict), "Data must be a dictionary."
        attribute = cls(run_check=False)
        for k, v in data.items():
            if hasattr(attribute, k):
                setattr(attribute, k, v)
            else:
                raise ValueError(f"Unexpected key {k} in deserializing Attribute.")
        attribute._sanity_check()
        return attribute

    def n_sampling(self) -> int:
        assert self.obtained_from == "query"
        return sum(1 for x in self.raw_data if x is not None)

    def _sanity_check(self) -> None:
        assert len(self.category) > 0 \
            and len(self.category) == len(self.stats)
        assert self.obtained_from in ["query", "critic"]
        if self.obtained_from == "query":
            assert len(self.raw_data) > 0
    
    def __eq__(self, other) -> bool:
        return isinstance(other, Attribute) and \
            self.attribute == other.attribute and \
            self.obtained_from == other.obtained_from
    
    def __repr__(self) -> str:
        return f"Attribute(attribute={self.attribute}, " \
               f"category={self.category}, " \
               f"obtained_from={self.obtained_from}, " \
               f"stats={self.stats})"


class Identity:
    
    def __init__(self,
                 owner: str,
                 attributes: Optional[List[Attribute]] = None):
        self.owner_id: str = owner
        self.attributes: List[Attribute] = attributes if attributes else []

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Identity":
        assert isinstance(data, dict), "Data must be a dictionary."
        assert "owner_id" in data, "owner_id is required in Identity."
        identity = cls(owner=data["owner_id"])
        identity.attributes = [
            Attribute.from_dict(attr)
            for attr in data.get("attributes", [])
        ]
        return identity

    def _sanity_check(self) -> None:
        for attr in self.attributes:
            attr._sanity_check()
            if self.attributes.count(attr) > 1:
                import pdb; pdb.set_trace()
                raise ValueError(
                    "Duplicate attributes in Identity. "
                    f"Attribute: {attr.attribute}"
                )
        
    def update(self, identity: "Attribute") -> None:
        self.attributes.append(identity)
        self._sanity_check()

    def __repr__(self) -> str:
        return f"Identity(owner_id={self.owner_id}, " \
               f"attributes={self.attributes})"
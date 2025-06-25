# stats: statistics on any agent-related operation. Include:
# distribution of demographics
# intervention outcomes

# usage:
# op = Operation(name="get_distribution")
# agentcollection.return_stats(
#   op=op, attribute={attribute}, 
#   regard_as={distribution or one_hot},
#   obtained_from={query or critic}
# )

from typing import Any, List, Literal, Dict

import numpy as np

from diary.entity.identity import Identity, Attribute
from diary.utils.misc_utils import class_to_dict

FAGERSTROM_QKEYS = {
    "fagerstrom_waking" : [3,2,1],
    "fagerstrom_forbidden_places" : [1,0],
    "fagerstrom_hate" : [1,0],
    "fagerstrom_cigarettes_per_day" : [0,1,2,3],
    "fagerstrom_smoke_morning" : [1,0],
    "fagerstrom_smoke_sick" : [1,0],
}


class Operation:

    def __init__(self, name: str, **kwargs: Dict):
        self.name = name
        self.routine: callable = getattr(self, name, None)
        assert self.routine is not None \
            and callable(self.routine), \
            f"Operation {name} is not callable or not exist."
        self.additional_params: Dict = kwargs

    def perform_on(self, identity_list: list[Identity], **kwargs: Any):
        self.update_params(**kwargs)
        return self.routine(identity_list)
    
    def update_params(self, **kwargs: Any):
        self.additional_params.update(kwargs)
             
    def get_distribution(self, identity_list: List[Identity]) -> List[float]:
        """
        kwargs required to perform this operation:
        - attribute: str, the attribute to be queried
        - obtained_from: Literal["query", "critic"]
        - regard_as: Literal["one_hot", "distribution"]
        """
        attribute = self.additional_params.get("attribute")
        obtained_from = self.additional_params.get("obtained_from")
        regard_as = self.additional_params.get("regard_as")
        available_attrs = [a.attribute for a in identity_list[0].attributes]
        assert attribute is not None and \
            attribute in available_attrs, (
            f"Attribute {attribute} not found in identity attributes. "
            f"Available attributes: {available_attrs}."
        )
        assert obtained_from in ["query", "critic"], \
            f"Invalid obtained_from value: {obtained_from}."
        assert regard_as in ["one_hot", "distribution"], \
            f"Invalid regard_as value: {regard_as}."

        per_agent_dist = np.array([
            id.get_attribute(attribute, obtained_from).stats
            for id in identity_list
        ])
        if regard_as == "one_hot":
            max_idx = per_agent_dist.argmax(axis=1)
            one_hot = np.zeros_like(per_agent_dist)
            one_hot[np.arange(len(max_idx)), max_idx] = 1
            per_agent_dist = one_hot
        aggr = per_agent_dist.sum(axis=0)
        return (aggr / aggr.sum()).tolist()


def calc_fagerstrom(identity: Identity,
                    regard_as: Literal["one_hot", "distribution"]) -> int:
    assert regard_as in ["one_hot", "distribution"]
    sum_score: int = 0
    for qkey in FAGERSTROM_QKEYS:
        assert qkey in [attr.attribute for attr in identity.attributes], (
            "Fagerstrom result {qkey} not found in Identity {identity.owner_id}"
        )
        stat = identity.get_attribute(qkey).stats
        if regard_as == "one_hot":
            sum_score += FAGERSTROM_QKEYS[qkey][np.argmax(stat)]
        else:
            sum_score += np.sum(np.array(FAGERSTROM_QKEYS[qkey]) * stat)
    identity.update(
        Attribute(
            attribute="fagerstrom_score",
            category=["score"],
            obtained_from="query",
            stats=[float(sum_score)],
            raw_data=[regard_as],
        )
    )
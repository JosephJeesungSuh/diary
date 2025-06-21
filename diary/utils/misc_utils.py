import random
import string
from typing import Any, Dict

def class_to_dict(obj: Any) -> Dict:
    if hasattr(obj, "serialize_fn"):
        return obj.serialize_fn()
    if hasattr(obj, "__dict__"):
        return {
            key: class_to_dict(value)
            for key, value in obj.__dict__.items()
        }
    if isinstance(obj, dict):
        return {
            key: class_to_dict(value)
            for key, value in obj.items()
        }
    if isinstance(obj, list):
        return [class_to_dict(item) for item in obj]
    return obj


def random_hashtag(length=32) -> str:
    chars = string.ascii_letters + string.digits
    tag = ''.join(random.choice(chars) for _ in range(length))
    return tag
import random
import re
import string
from typing import Any, Dict, List

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

def extract_placeholder(text: str) -> List[str]:
    return re.findall(r'\{(.*?)\}', text)

def clean_format_chat(chat: List[Dict[str, str]]) -> List[Dict[str, str]]:
    cleaned_chat = []
    idx = 0
    while idx < len(chat):
        role = chat[idx].get("role")
        assert role in ["user", "assistant", "system"], (
            "Invalid role in chat: {}".format(role)
        )
        content = chat[idx].get("content", "").strip()
        idx_cp = idx + 1
        while idx_cp < len(chat) and chat[idx_cp].get("role") == role:
            content += " " + chat[idx_cp].get("content", "").strip()
            idx_cp += 1
        cleaned_chat.append({"role": role, "content": content})
        idx = idx_cp
    return cleaned_chat
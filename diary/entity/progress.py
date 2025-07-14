# progress.py deals with the temporal progress of the diary.
# entry-level API is forward(), which progress a timestep
# and returns a list of operations that should be invoked at that timestamp.
# also records the progress (events occured so far).
# COND operations are recorded in self.progress but not invoked by forward().

from typing import Dict, Any, List, Tuple, Optional, Union

from diary.utils.misc_utils import class_to_dict

class Progress:
    def __init__(self, env: Optional[Dict] = None):
        self.progress: List[Tuple[str, int]] = []
        self.current: Optional[int] = None
        self.until: Optional[int] = None
        self.timestep: int = 1
        self.invocable_ops: Dict[str, Any] = {}
        if env is not None:
            self.update_env(env)

    def update_env(self, env: Dict) -> None:
        assert isinstance(env, dict)
        if self.current == None:
            self.current = -1
        self.until = self.convert_timestamp_to_day(
            env.get("study_duration", "00-00-00")
        )
        self.timestep = self.convert_timestamp_to_day(
            env.get("timestep", "00-00-01")
        )
        for op_name, op_info in env.get("ops", {}).items():
            assert op_name not in self.invocable_ops, (
                f"Operation {op_name} already exists in invocable_ops. "
                "Invalid attempt to overwrite."
            )
            if op_name.startswith('FREQ'): # frquency-based ops
                freq = self.convert_timestamp_to_day(op_info["freq"])
                except_times = [
                    self.convert_timestamp_to_day(t)
                    for t in op_info.get("except", [])
                ]
                self.invocable_ops[op_name] = tuple([
                    "FREQ", # type of operation
                    tuple([freq, freq]), # frequency and cooltime
                    except_times # exceptions
                ])
            elif op_name.startswith('TIME'):
                times = [self.convert_timestamp_to_day(t)
                         for t in op_info["when"]]
                self.invocable_ops[op_name] = tuple([
                    "TIME", # type of operation
                    times # list of moments event occurs
                ])
            elif op_name.startswith('COND'):
                trigger = op_info["trigger"]
                assert isinstance(trigger, str) or isinstance(trigger, list)
                self.invocable_ops[op_name] = tuple([
                    "COND", trigger
                ])
        return

    def forward(self) -> List[str]:
        """
        Progress the longitudinal progress by one timestep
        and return a list of operations that should be invoked at this timestep.
        """
        if self.current >= self.until:
            return ["<TERMINATE>"]
        self.current += self.timestep
        invoked_ops: List[str] = []
        for op_name, op_info in self.invocable_ops.items():
            if op_info[0] == "FREQ":
                freq, cooltime = op_info[1]
                cooltime -= self.timestep
                if cooltime <= 0:
                    if self.current not in op_info[2]:
                        invoked_ops.append(op_name)
                    self.invocable_ops[op_name] = (
                        "FREQ", (freq, freq), op_info[2]
                    )
                else:
                    self.invocable_ops[op_name] = (
                        "FREQ", (freq, cooltime), op_info[2]
                    )
            elif op_info[0] == "TIME":
                times = op_info[1]
                if self.current in times:
                    invoked_ops.append(op_name)
        return invoked_ops

    def is_finished(self) -> bool:
        return self.current >= self.until
    
    def current_printable(self) -> str:
        assert self.current is not None, "Progress has not been initialized."
        if self.timestep == 1:
            return "[Day {day}] ".format(day=self.current)
        elif self.timestep == 7:
            return "[Week {week}] ".format(week=self.current // 7)
        else:
            raise NotImplementedError(
                "Has not considered timestep other than daily or weekly.")
    
    def remaining_days(self) -> int:
        assert self.until is not None, "Progress has not been initialized."
        return self.until - self.current
    
    def remaining_weeks(self) -> int:
        assert self.until is not None, "Progress has not been initialized."
        return (self.until - self.current) // 7
    
    def update_progress(self, event: Tuple[str, int]) -> None:
        assert isinstance(event, tuple) and len(event) == 2, (
            "Event must be a tuple of (timestamp, value)."
        )
        self.progress.append(event)

    def serialize(self) -> Dict:
        return class_to_dict(self)
    
    @staticmethod
    def convert_timestamp_to_day(timestamp: str) -> int:
        assert isinstance(timestamp, str)
        month, week, day = map(int, timestamp.split("-"))
        return month * 30 + week * 7 + day

    @staticmethod
    def convert_day_to_timestamp(day: int) -> str:
        assert isinstance(day, int)
        month = day // 30
        week = (day % 30) // 7
        day = (day % 30) % 7
        return f"{month:02d}-{week:02d}-{day:02d}"
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Progress":
        assert isinstance(data, dict), "Data must be a dictionary."
        progress = cls()
        progress.progress = data.get("progress", [])
        progress.current = data.get("current", None)
        progress.until = data.get("until", None)
        progress.timestep = data.get("timestep", 1)
        progress.invocable_ops = data.get("invocable_ops", {})
        return progress
    
    def __repr__(self) -> str:
        repr_str = ""
        repr_str += "=" * 20 + " Progress " + "=" * 20 + "\n"
        repr_str += f"Current: {self.convert_day_to_timestamp(getattr(self, 'current', -1))}\n"
        repr_str += f"Until: {self.convert_day_to_timestamp(getattr(self, 'until', -1))}\n"
        repr_str += f"Timestep: {self.convert_day_to_timestamp(getattr(self, 'timestep', 1))}\n"
        repr_str += "Invocable Operations:\n"
        for op_name, op_info in self.invocable_ops.items():
            op_type = op_info[0]
            if op_type == "FREQ":
                repr_str += f"  {op_name}: Frequency={op_info[1][0]}, Cooltime={op_info[1][1]}, Exceptions={op_info[2]}\n"
            elif op_type == "TIME":
                repr_str += f"  {op_name}: Occurs at={', '.join(map(str, op_info[1]))}\n"
            elif op_type == "COND":
                repr_str += f"  {op_name}: Triggering event={op_info[1]}\n"
        repr_str += "Progress until now:\n"
        for idx, (action, timestamp) in enumerate(self.progress):
            repr_str += (
                f"{idx + 1}. Action: {action}, "
                f"Timestamp: {self.convert_day_to_timestamp(timestamp)}\n"
            )
        repr_str += "=" * 50 + "\n"
        return repr_str
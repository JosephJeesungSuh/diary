# progress.py deals with the temporal progress of the diary.
# it is a wrapper for the temporal information.

class Progress:
    def __init__(self, start_time: str, end_time: str):
        self.start_time = start_time
        self.end_time = end_time
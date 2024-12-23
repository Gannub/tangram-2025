import time
from datetime import datetime

class Timer:
    def __init__(self):
        self.start_time = time.time()
        self.elapsed = 0

    def elapsed_time(self):
        e = time.time() - self.start_time
        self.elapsed = e
        return e

    def elapsed_time_hr(self):
            elapsed_seconds = int(self.elapsed_time())
            hours = elapsed_seconds // 3600
            minutes = (elapsed_seconds % 3600) // 60
            seconds = elapsed_seconds % 60
            return f"{hours:02}:{minutes:02}:{seconds:02}"
import time


class Timer:

    def __init__(self):
        self._measured_times = []
        self._start_time = None

    def start(self):
        self._start_time = time.time()

    def stop(self):
        if not self._start_time:
            raise ValueError("You have to run timer.start() before timer.stop().")
        # This is inaccurate on windows (see: https://stackoverflow.com/a/1938096)
        # TODO: If we require Python 3.7, we could use time.time_ns()
        self._measured_times.append(time.time() - self._start_time)
        self._start_time = None

    def reset(self):
        self._measured_times = []

    def average(self):
        return sum(self._measured_times) / len(self._measured_times)

    def average_tail(self, tail_size=15):
        tail_size = min(tail_size, len(self._measured_times))
        return sum(self._measured_times[-tail_size:]) / tail_size

    def total(self, num_total: int = None):
        if not num_total:
            num_total = len(self._measured_times)
        return self.average() * num_total

    def last(self):
        return self._measured_times[-1]

    def format_status(self, tail_size: int = 15, num_total=None) -> str:
        return "{:8.3f} ms | {} s".format(self.average_tail(tail_size) * 1e3, int(self.total(num_total)))

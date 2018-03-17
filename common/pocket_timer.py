import time


class GoldenTimer:
    def __init__(self):
        self.start_time = time.time()

    def time(self, print_str):
        duration = time.time() - self.start_time
        print(print_str, duration)
        self.start_time = time.time()
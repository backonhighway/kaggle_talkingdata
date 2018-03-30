import time

class GoldenTimer:
    def __init__(self, logger_arg=None):
        self.start_time = time.time()
        self.logger = None
        if logger_arg is not None:
            self.logger = logger_arg

    def time(self, print_str):
        duration = time.time() - self.start_time
        print(print_str, duration)
        if self.logger is not None:
            self.logger.info(print_str, duration)
        self.start_time = time.time()
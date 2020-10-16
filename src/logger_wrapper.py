#!/usr/bin/env python3

# Adapted from Facebook Habitat Framework

import logging
import copy

class Logger(logging.Logger):
    def __init__(
        self,
        name,
        level,
        filename=None,
        filemode="a",
        stream=None,
        format=None,
        dateformat=None,
        style="%",
    ):
        super().__init__(name, level)
        if filename is not None:
            handler = logging.FileHandler(filename, filemode)
        else:
            handler = logging.StreamHandler(stream)
        self._formatter = logging.Formatter(format, dateformat, style)
        handler.setFormatter(self._formatter)
        super().addHandler(handler)
        self.stat_queue = [] # Going to be tuples

    def add_filehandler(self, log_filename):
        filehandler = logging.FileHandler(log_filename)
        filehandler.setFormatter(self._formatter)
        self.addHandler(filehandler)

    def queue_stat(self, stat_name, stat):
        self.stat_queue.append((stat_name, stat))

    def empty_queue(self):
        queue = copy.deepcopy(self.stat_queue)
        self.stat_queue = []
        return queue

    def mute(self):
        self.setLevel(logging.ERROR)


def make_logger():
    return Logger(
        name="NoisedRNNNetworks", level=logging.INFO, format="%(asctime)-15s %(message)s"
    )


__all__ = ["make_logger"]

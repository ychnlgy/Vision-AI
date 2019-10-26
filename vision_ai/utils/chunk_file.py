import os

import numpy


class ChunkFile:
    
    def __init__(self, fname, mode):
        self._fname = fname
        self._file = None
        self._mode = mode

    def __enter__(self):
        self._file = open(self._fname, self._mode)
        return self
    
    def __exit__(self, *args):
        self._file.flush()
        self._file.close()
    
    def save(self, data):
        numpy.save(self._file, data)
    
    def load(self):
        while True:
            try:
                yield numpy.load(self._file)
            except OSError:
                break

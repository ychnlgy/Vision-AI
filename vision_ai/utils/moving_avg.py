class MovingAvg:
    
    def __init__(self, decay):
        self.val = None
        self.decay = decay
    
    def update(self, val):
        self.val = val
        self.update = self._update
    
    def _update(self, val):
        self.val = self.val * self.decay + val * (1 - self.decay)
    
    def peek(self):
        return self.val

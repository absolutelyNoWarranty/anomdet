import datetime

class SimpleTimer(object):
    '''SimpleTimer
    A class for the coarse measurement of execution time.
    
    Based on MATLAB's tic/toc
    '''
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.start_t = None
        self.end_t = None
        self.diff = None
        
    def start(self):
        self.tic()
        
    def stop(self):
        self.toc()
    
    def tic(self):
        self.start_t = datetime.datetime.now()
    
    def toc(self):
        self.end_t = datetime.datetime.now()
        self.diff = (self.end_t - self.start_t).total_seconds()
        return self.diff
    
    def __str__(self):
        if self.start_t is None:
            return "SimpleTimer not started"
        
        elif self.end_t is None:
            return "SimpleTimer is running"
            
        else:
            return "{diff} seconds elapsed.".format(diff=self.diff)
            
my_timer = SimpleTimer()
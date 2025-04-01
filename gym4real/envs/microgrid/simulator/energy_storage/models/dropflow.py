# coding: utf-8
"""
Implements incremental rainflow cycle counting algorythm for fatigue analysis
according to section 5.4.4 in ASTM E1049-85 (2011).
"""
from __future__ import division
from collections import deque, defaultdict


def format_output(point1, point2, count):
            i1, x1 = point1
            i2, x2 = point2
            rng = abs(x1 - x2)
            mean = 0.5 * (x1 + x2)
            return rng, mean, count, i1, i2


class Dropflow:
    """
    Class of the incremental rainflow cycle counting algorithm.
    The name Dropflow represents the idea of dropping a point at time instead of the 
    entire "rain" of points at once.
    """
    def __init__(self) -> None:
        self._reversals = []
        self._stopper = ()
        self._closed_cycles = []

        self._mean = 0
        self._history_length = 0
        self._idx_last = None
        self._x_last = None
        self._x = None
        self._d_last = None
    
    @property
    def reversals(self):
        if self._history_length < 2:
            return []
        return self._reversals + [self._stopper] if self._stopper else self._reversals
        
    def reset(self):
        self._reversals = []
        self._stopper = ()
        self._closed_cycles = []

        self._mean = 0
        self._history_length = 0
        self._idx_last = None
        self._x_last = None
        self._x = None
        self._d_last = None
        
    def add_point(self, x: float, idx: int) -> None:
        """
        Add a point to the series.
        
        Args:
            x (float): value of the point
            idx (int): index of the point
        """
        self._check_reversal(x, idx)
        self._mean = (self._mean * (self._history_length) + x) / (self._history_length + 1)
        self._history_length += 1
    
    def _check_reversal(self, x: float, idx: int) -> None:
        """
        Check if the provided point is a reversal point.

        A reversal point is a point in the series at which the first derivative
        changes sign. Reversal is undefined at the first (last) point because the
        derivative before (after) this point is undefined. The first and the last
        points are treated as reversals.

        Parameters
        ----------
        x (float): value of the point
        idx (int): index of the point
        """
        if self._history_length == 0:
            self._x_last = x
            self._idx_last = 0
            
        elif self._history_length == 1:
            self._x = x
            self._d_last = (x - self._x_last)
            self._reversals.append((self._idx_last, self._x_last))
            self._idx_last = idx
        
        else:
            if x == self._x:
                self._idx_last = idx
                return
            
            # Here we decide if the last point is a reversal or not
            d_next = (x - self._x)
            
            if self._d_last * d_next < 0:
                self._reversals.append((self._idx_last, self._x))
            self._x_last, self._x = self._x, x
            self._d_last = d_next
            self._idx_last = idx
            
            # A new point is always a reversal until the following point is read
            self._stopper = (idx, x)
            
    def extract_all_cycles(self, ignore_stopper=False):
        """
        Iterate closed cycles and half cycles.
        In this method we append the closed cycles within the attribute _closed_cycles and pop 
        the relative points from the reversals.
        Instead, the half cycles are yielded and relative points are not popped from the reversals.

        Yields
        ------
        cycle : tuple
            Each tuple contains (range, mean, count, start index, end index).
            Count equals to 1.0 for full cycles and 0.5 for half cycles.
        """         
        self._reversals.extend([self._stopper]) if self._stopper and not ignore_stopper else None
        
        if len(self._closed_cycles) == 0 and len(self._reversals) < 1:
            return []
        
        # Yield already closed cycles
        for cycle in self._closed_cycles:
            yield cycle
                    
        i = 0
        while i < (len(self._reversals) - 2):
            # Form ranges X and Y from the three most recent points
            x1, x2, x3 = self._reversals[i][1], self._reversals[i+1][1], self._reversals[i+2][1]
            X = abs(x3 - x2)
            Y = abs(x2 - x1)
            
            if X < Y:
                # Read the next point
                i += 1
            else:
                if i == 0:
                    # Y contains the starting point
                    # Count Y as one-half cycle and discard the first point
                    self._closed_cycles.append(format_output(self._reversals[i], self._reversals[i+1], 0.5))
                    self._reversals.pop(i)
                    yield self._closed_cycles[-1]
                else:
                    # Count Y as one cycle and discard the peak and the valley of Y
                    self._closed_cycles.append(format_output(self._reversals[i], self._reversals[i+1], 1.0))
                    self._reversals.pop(i)
                    self._reversals.pop(i)
                    yield self._closed_cycles[-1]
        else:
            # Count the remaining ranges as one-half cycles 
            for i in range(len(self._reversals) - 1):
                yield format_output(self._reversals[i], self._reversals[i+1], 0.5)
                i -= 1
                
            if not ignore_stopper and self._reversals[-1] == self._stopper:
                self._reversals.pop()
                
    def extract_new_cycles(self, ignore_stopper=False):
        """
        Iterate closed cycles and half cycles.
        In this method we don't save the closed cycles and we delegate the user to save them.
        Indeed, we just yield the new closed cycles or half cycles.

        Yields
        ------
        cycle : tuple
            Each tuple contains (range, mean, count, start index, end index).
            Count equals to 1.0 for full cycles and 0.5 for half cycles.
        """         
        self._reversals.extend([self._stopper]) if self._stopper and not ignore_stopper else None
        
        if len(self._reversals) < 1:
            #print("Not enough samples")
            return []
        
        i = 0
        while i < (len(self._reversals) - 2):
            # Form ranges X and Y from the three most recent points
            x1, x2, x3 = self._reversals[i][1], self._reversals[i+1][1], self._reversals[i+2][1]
            X = abs(x3 - x2)
            Y = abs(x2 - x1)
            
            if X < Y:
                # Read the next point
                i += 1
            else:
                if i == 0:
                    # Y contains the starting point
                    # Count Y as one-half cycle and discard the first point
                    yield format_output(self._reversals[i], self._reversals[i+1], 0.5)
                    self._reversals.pop(i)
                else:
                    # Count Y as one cycle and discard the peak and the valley of Y
                    yield format_output(self._reversals[i], self._reversals[i+1], 1.0)
                    self._reversals.pop(i)
                    self._reversals.pop(i)
                
        else:
            # Count the remaining ranges as one-half cycles 
            for i in range(len(self._reversals) - 1):
                yield format_output(self._reversals[i], self._reversals[i+1], 0.5)
                i -= 1
                
            if not ignore_stopper and self._reversals[-1] == self._stopper:
                self._reversals.pop()
                
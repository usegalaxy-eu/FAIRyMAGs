"""
EVALUATION MODULE
=================
This module contains code to evaluate different regression models for resource estimation.
It includes functions to compute performance metrics, visualize results, and compare models.
"""
import numpy as np
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, Any, Callable
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, Tuple



def simple_retry_policy(
        R: int, # retry count, 0 means first attempt
        a0: float, # initial allocation
        ) -> float: # next allocation
    """Simple retry policy: increase allocation by m each time."""

    return (R+1) * a0

def failure_rate(y_true, y_safe):
    """Jobs that would OOM (allocation below actual). Lower is better."""
    return float(np.mean(y_safe < y_true))

def overallocation(y_true, y_safe):
    """Mean relative overallocation. Lower is better (but must balance failure rate)."""
    rel = (y_safe - y_true) / np.clip(y_true, 4, None)
    return float(np.mean(rel))

def binned_metrics(y_true, y_safe, n_bins=10):
    bins = np.linspace(y_true.min(), y_true.max(), n_bins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    failure_rates = []
    overallocations = []
    bin_counts = []

    for i in range(n_bins):
        bin_mask = (y_true >= bins[i]) & (y_true < bins[i + 1])
        bin_y_true = y_true[bin_mask]
        bin_y_safe = y_safe[bin_mask]
        bin_count = len(bin_y_true)
        bin_counts.append(bin_count)
        if bin_count > 0:
            fr = failure_rate(bin_y_true, bin_y_safe)
            orc = overallocation(bin_y_true, bin_y_safe)
        else:
            fr = np.nan
            orc = np.nan
        failure_rates.append(fr)
        overallocations.append(orc)

    return bin_centers, np.array(failure_rates), np.array(overallocations), np.array(bin_counts)


class JobCost:

    @classmethod
    def simulate_retries(cls, 
                        y_true:        float, # true peak memory
                        true_wall_time: float|None, # the wall time needed for job, if None it will asume that time is proportional to memory
                        retry_allocation_policy:   Callable[[int,float],float], # function to get next allocation given given a0, and attempt number. It return a float or None if no more retries
                        a0:             float, # initial memory allocation
        ) -> Tuple[int, list, float, float]: 
        """
        Simulate retries until success or no more retries.
        Returns (R, a_R, failed_mem_rate_sum) where:
          R = number of retys (0 means success on first attempt),
          alloc_time = tuple with (allocation, wall_time) of each attempt
          a_R = final allocation
          failed_mem_rate_sum = sum of (allocations*wall_time) used in failed attempts
        """
        aR = a0
        R = 0
        failed_alloc_time = []
        failed_mem_rate_sum = 0.0
        while aR < y_true:
            fraction_allocated = aR / y_true
            if true_wall_time is None:
                wall_time = fraction_allocated * 1.0 # assume time proportional to memory
            else:
                wall_time = fraction_allocated * true_wall_time

            failed_mem_rate_sum += aR * wall_time
            failed_alloc_time.append((wall_time, aR))
            
            R += 1
            
            aR = retry_allocation_policy(R, a0)
            if aR is None:
                break
            if aR <= 0:
                raise ValueError("Retry policy returned non-positive allocation.")

        return R, failed_alloc_time, aR, failed_mem_rate_sum


    @classmethod
    def succesful_attempt_waste(cls,
                                aR: float, # final allocation
                                y_true: float, # true peak memory
                                true_wall_time: float|None # the wall time needed for job, if None it will asume that time is proportional to memory
                                ) -> Tuple[list, float]:
        """Compute waste on successful attempt."""
        if aR < y_true:
            raise ValueError("Final allocation must be >= true memory for successful attempt.")
        overalloc = aR - y_true
        if true_wall_time is None:
            wall_time = float(aR) # assume time proportional to memory
        else:
            wall_time = true_wall_time
        return [(wall_time,overalloc)], overalloc * wall_time

    @classmethod
    def job_cost(cls,
                 y_true: float, # true peak memory
                 true_wall_time: float|None, # the wall time needed for job, if None it will asume that time is proportional to memory
                 retry_allocation_policy: Callable[[int,float],float], # function to get next allocation given given a0, and attempt number. It return a float or None if no more retries
                 a0: float, # initial memory allocation. From predictor or heuristic

        ) -> Dict[str, Any]:
        """Compute total cost of job with retries."""
        R, failed_alloc_time, aR, failed_mem_rate_sum = cls.simulate_retries(y_true, true_wall_time, retry_allocation_policy, a0)
        over_alloc_time, overalloc_mem_rate = cls.succesful_attempt_waste(aR, y_true, true_wall_time)
        total_mem_rate = failed_mem_rate_sum + (aR * (true_wall_time if true_wall_time is not None else float(aR)))
        total_waste_rate = overalloc_mem_rate + failed_mem_rate_sum
        waste_alloc_time = failed_alloc_time + over_alloc_time
        total_time = sum(wt for wt, _ in waste_alloc_time)
        return {
            "R": R,
            "a_R": aR,
            "failed_alloc_time": failed_alloc_time,
            "over_alloc_time": over_alloc_time,
            "waste_alloc_time": waste_alloc_time,
            "failed_mem_rate_sum": failed_mem_rate_sum,
            "overalloc_mem_rate": overalloc_mem_rate,
            "total_mem_rate": total_mem_rate,
            "total_time": total_time,
            "total_waste_rate": total_waste_rate
        }
    
    @classmethod
    def batch_job_total_waste_rate(cls,
                                      y_true: np.ndarray, # true peak memory
                                      true_wall_time: np.ndarray|None, # the wall time needed for job, if None it will asume that time is proportional to memory
                                      retry_allocation_policy: Callable[[int,float],float], # function to get next allocation given given a0, and attempt number. It return a float or None if no more retries
                                      a0: np.ndarray, # initial memory allocation. From predictor or heuristic
          ) -> np.ndarray:
          """Compute total waste rate for a batch of jobs."""
          results = [cls.job_cost(yt, twt if true_wall_time is not None else None, retry_allocation_policy, a0i)
                     for yt, twt, a0i in zip(y_true, true_wall_time if true_wall_time is not None else [None]*len(y_true), a0)]
          return np.array([res["total_waste_rate"] for res in results])
    
    @classmethod
    def batch_job_total_time(cls,
                                      y_true: np.ndarray, # true peak memory
                                      true_wall_time: np.ndarray|None, # the wall time needed for job, if None it will asume that time is proportional to memory
                                      retry_allocation_policy: Callable[[int,float],float], # function to get next allocation given given a0, and attempt number. It return a float or None if no more retries
                                      a0: np.ndarray, # initial memory allocation. From predictor or heuristic
          ) -> np.ndarray:
          """Compute total wall time for a batch of jobs."""
          results = [cls.job_cost(yt, twt if true_wall_time is not None else None, retry_allocation_policy, a0i)
                     for yt, twt, a0i in zip(y_true, true_wall_time if true_wall_time is not None else [None]*len(y_true), a0)]
          return np.array([res["total_time"] for res in results])
    

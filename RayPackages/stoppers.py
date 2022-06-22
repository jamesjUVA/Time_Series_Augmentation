import ray
from typing import Dict, Optional
import time
from collections import defaultdict, deque
import numpy as np

class CMaximumIterationStopper(ray.tune.Stopper):
    """Stop trials after reaching a maximum number of iterations

    Args:
        max_iter (int): Number of iterations before stopping a trial.
    """

    def __init__(self, max_iter: int):
        self._max_iter = max_iter
        self._iter = defaultdict(lambda: 0)

    def __call__(self, trial_id: str, result: Dict):
        self._iter[trial_id] += 1
        stop = self._iter[trial_id] >= self._max_iter
        #print('STOPPPER')
        #print(stop)
        return stop

    def stop_all(self):
        return False

class EarlyStopping(ray.tune.Stopper):
    """Early stop single trials when they reached a plateau.

    When the the `metric` result of a trial stops improving,
    the trial plateaued and will be stopped early.

    Args:
        metric_threshold (float):
            Minimum or maximum value the result has to exceed before it can
            be stopped early.
        mode (Optional[str]): If a `metric_threshold` argument has been
            passed, this must be one of [min, max]. Specifies if we optimize
            for a large metric (max) or a small metric (min). If max, the
            `metric_threshold` has to be exceeded, if min the value has to
            be lower than `metric_threshold` in order to early stop.
    """

    def __init__(self,
                 metric: str,
                 metric_threshold: float = .0001,
                 mode: str = "min",
                 warmup: int = 200,
                 patience: int = 50,
                 metric_threshold_abs: float = -.1):
                
        self._iter = defaultdict(lambda: 0)
        self.last_best_iter = defaultdict(lambda: 0)
        self.patience = patience
        self._metric = metric
        self._mode = mode
        self.warmup = warmup
        self._metric_threshold = metric_threshold
        self._metric_threshold_abs = metric_threshold_abs

        if self._metric_threshold:
            if mode not in ["min", "max"]:
                raise ValueError(
                    f"When specifying a `metric_threshold`, the `mode` "
                    f"argument has to be one of [min, max]. "
                    f"Got: {mode}")
        if self._mode == "min":
            self.last_best = defaultdict(lambda: 100000)
        else:
            self.last_best = defaultdict(lambda: -100000)

    def __call__(self, trial_id: str, result: Dict):
        metric_result = result.get(self._metric)
        self._iter[trial_id] += 1
        # If metric threshold value not reached, do not stop yet

        if self._iter[trial_id] > self.warmup:
            if self._mode == "min":

                if metric_result < self.last_best[trial_id] - self._metric_threshold:
                    self.last_best[trial_id] = metric_result
                    self.last_best_iter[trial_id] = 0
                    return False
                else:
                    if self.last_best[trial_id] > self._metric_threshold_abs:
                        return False
                    self.last_best_iter[trial_id] += 1
                    if self.last_best_iter[trial_id] >= self.patience:
                        print('THRESH STOP')
                        return True
                    else:
                        return False

            else:

                if metric_result > self.last_best[trial_id] + self._metric_threshold:
                    self.last_best[trial_id] = metric_result
                    self.last_best_iter[trial_id] = 0
                    return False
                else:
                    if self.last_best[trial_id] < self._metric_threshold_abs:
                        return False
                    self.last_best_iter[trial_id] += 1
                    if self.last_best_iter[trial_id] >= self.patience:
                        print('THRESH STOP')
                        return True
                    else:
                        return False
        else:
            return False

    def stop_all(self):
        return False
    
class LossThresholdStopper(ray.tune.Stopper):
    """Early stop single trials when they reached a plateau.

    When the standard deviation of the `metric` result of a trial is
    below a threshold `std`, the trial plateaued and will be stopped
    early.

    Args:
        metric_threshold (float):
            Minimum or maximum value the result has to exceed before it can
            be stopped early.
        mode (Optional[str]): If a `metric_threshold` argument has been
            passed, this must be one of [min, max]. Specifies if we optimize
            for a large metric (max) or a small metric (min). If max, the
            `metric_threshold` has to be exceeded, if min the value has to
            be lower than `metric_threshold` in order to early stop.
    """

    def __init__(self,
                 metric: str,
                 metric_threshold: float = .01,
                 mode: Optional[str] = None):
        self._metric = metric
        self._mode = mode
        self._metric_threshold = metric_threshold
        if self._metric_threshold:
            if mode not in ["min", "max"]:
                raise ValueError(
                    f"When specifying a `metric_threshold`, the `mode` "
                    f"argument has to be one of [min, max]. "
                    f"Got: {mode}")



    def __call__(self, trial_id: str, result: Dict):
        metric_result = result.get(self._metric)
        print(metric_result)
        # If metric threshold value not reached, do not stop yet
        if self._metric_threshold is not None:
            if self._mode == "min" and metric_result < self._metric_threshold:
                print('THRESH STOP')
                return True
            elif self._mode == "max" and \
                    metric_result > self._metric_threshold:
                print('THRESH STOP')
                return True
            else:
                print('NO THRESH')
                return False
            
    def stop_all(self):
        return False
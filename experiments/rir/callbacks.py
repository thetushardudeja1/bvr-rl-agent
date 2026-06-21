"""
callbacks.py
============
Comprehensive training logger for RIR runs. Records combat outcome metrics
(win/loss/timeout rate, returns, episode length, missiles) to TensorBoard +
CSV, and saves the best model by rolling mean return.
"""
import numpy as np
from collections import deque
from stable_baselines3.common.callbacks import BaseCallback


class RIRLogger(BaseCallback):
    def __init__(self, save_path, win_thr=50.0, loss_thr=-100.0,
                 window=100, print_freq=20, verbose=0):
        super().__init__(verbose)
        self.save_path  = save_path
        self.win_thr    = win_thr
        self.loss_thr   = loss_thr
        self.results    = deque(maxlen=window)
        self.returns    = deque(maxlen=window)
        self.lengths    = deque(maxlen=window)
        self.best_rew   = -np.inf
        self.print_freq = print_freq
        self._rollouts  = 0
        self.total_eps  = 0

    def _classify(self, r):
        if r > self.win_thr:  return 'WIN'
        if r < self.loss_thr: return 'LOSS'
        return 'TIMEOUT'

    def _on_step(self):
        for info in self.locals.get('infos', []):
            ep = info.get('episode')
            if ep is not None:
                r = float(ep['r'])
                self.returns.append(r)
                self.lengths.append(int(ep['l']))
                oc = info.get('outcome')
                if oc in ('win', 'loss', 'timeout'):
                    self.results.append(oc.upper())   # explicit env outcome
                else:
                    self.results.append(self._classify(r))  # fallback
                self.total_eps += 1
        return True

    def _on_rollout_end(self):
        self._rollouts += 1
        if not self.results:
            return
        n = len(self.results)
        win     = self.results.count('WIN')     / n
        loss    = self.results.count('LOSS')    / n
        timeout = self.results.count('TIMEOUT') / n
        mean_r  = float(np.mean(self.returns))
        mean_l  = float(np.mean(self.lengths))

        self.logger.record('combat/win_rate',       win)
        self.logger.record('combat/loss_rate',      loss)
        self.logger.record('combat/timeout_rate',   timeout)
        self.logger.record('combat/ep_return_mean', mean_r)
        self.logger.record('combat/ep_len_mean',    mean_l)
        self.logger.record('combat/total_episodes', self.total_eps)
        self.logger.record('time/total_timesteps',  self.num_timesteps)

        if mean_r > self.best_rew and n >= 10:
            self.best_rew = mean_r
            self.model.save(self.save_path + '_best')
            self.logger.record('combat/best_return', self.best_rew)

        # NOTE: do NOT call logger.dump() here — PPO dumps once per rollout
        # right after train(), so our records flush in the same clean row.

        if self._rollouts % self.print_freq == 0:
            print(f"  [{self.num_timesteps:>10,}] win={win:.0%} loss={loss:.0%} "
                  f"timeout={timeout:.0%}  ret={mean_r:.1f}  eps={self.total_eps}")

"""
selfplay_env.py
==============
Self-play BVR environment: RED is controlled by a FROZEN policy (a past
snapshot of the learner, or the BT-beating seed) instead of the behaviour tree.
BLUE is the learner. Reward/observation are returned from BLUE's perspective,
exactly like BVRBase, so our existing PPO pipeline trains unchanged.

The opponent policy is loaded ONCE per worker process (cached by path) to avoid
360 separate 125MB loads. Red acts via its frozen policy each step.
"""
import numpy as np
from stable_baselines3 import PPO
from jsb_gym.envs.BaseEnv import BVRBase
from jsb_gym.agents.config import blue_agent, red_agent
from jsb_gym.agents.agents import RLBVRAgent
from jsb_gym.bts.bts import BVRBT
from jsb_gym.agents.agents import BTBVRAgent
from jsb_gym.utils.geospatial import (
    dinstance_between_agents, bearing_between_agents, to_360
)
from jsb_gym.utils.scale import scale_between

# per-process cache: path -> loaded policy (or "BT")
_OPP_CACHE = {}


def _get_opponent(path, device='cpu'):
    if path == 'BT':
        return 'BT'
    if path not in _OPP_CACHE:
        import torch
        torch.set_num_threads(1)
        _OPP_CACHE[path] = PPO.load(path, device=device).policy
    return _OPP_CACHE[path]


class SelfPlayBVR(BVRBase):
    """conf must additionally carry `opponent_path` (str path or 'BT')."""

    def reset(self, seed=None, options=None):
        import gymnasium as gym
        gym.Env.reset(self, seed=seed)

        self.blue_agent = RLBVRAgent(blue_agent, self)
        opp = getattr(self.conf, 'opponent_path', 'BT')
        if opp == 'BT':
            self.red_agent = BTBVRAgent(red_agent, self)
            self.red_agent.load_BT(BVRBT)
            self._opp_policy = 'BT'
        else:
            self.red_agent = RLBVRAgent(red_agent, self)
            self._opp_policy = _get_opponent(opp)

        self.all_agents = [self.blue_agent, self.red_agent]
        self.blue_agent.set_target(self.red_agent)
        self.red_agent.set_target(self.blue_agent)

        self.done = False
        self.tacview_logger = None
        self.prev_dist = None
        self.missile_fired = False
        self.enemy_missile_was_active = False
        self.red_kill_reward_given = False
        self.red_state = None

        self.update_state()
        self._update_red_state()
        return self.state, {}

    # --- parameterized obs vector: `agent` viewing `target` ---
    def _obs_vec(self, agent, target):
        bearing = to_360(bearing_between_agents(agent, target))
        heading = agent.simObj.get_psi()
        mach    = agent.simObj.get_mach()
        alt     = agent.simObj.get_altitude()
        d       = dinstance_between_agents(agent, target)
        e_bear  = to_360(bearing_between_agents(target, agent))
        e_head  = target.simObj.get_psi()
        e_mach  = target.simObj.get_mach()
        e_alt   = target.simObj.get_altitude()
        lim = agent.simObj.conf.aircraft_limits
        v = np.array([
            np.sin(np.radians(bearing)), np.cos(np.radians(bearing)),
            np.sin(np.radians(heading)), np.cos(np.radians(heading)),
            scale_between(mach, 0.1, 1.5),
            scale_between(alt, lim.alt_min, lim.alt_max),
            scale_between(d, 0.0, 120e3),
            np.sin(np.radians(e_bear)), np.cos(np.radians(e_bear)),
            np.sin(np.radians(e_head)), np.cos(np.radians(e_head)),
            scale_between(e_mach, 0.1, 1.5),
            scale_between(e_alt, lim.alt_min, lim.alt_max),
            1.0 if agent.is_own_missile_active() else 0.0,
            1.0 if target.is_own_missile_active() else 0.0,
        ], dtype=np.float32)
        v = np.nan_to_num(v, nan=0.0, posinf=1.0, neginf=-1.0)
        np.clip(v, -1.0, 1.0, out=v)
        return v

    def _update_red_state(self):
        if self._opp_policy == 'BT':
            return
        obs_nn = self._obs_vec(self.red_agent, self.blue_agent)
        if self.red_state is None:
            self.red_state = np.tile(obs_nn, (self.obs_shape[0], 1))
        else:
            self.red_state = np.roll(self.red_state, shift=-1, axis=0)
            self.red_state[-1, :] = obs_nn

    def step(self, action):
        action = self.from_nn2agent(action, self.blue_agent)

        # red's action from its frozen policy (or BT)
        red_action = None
        if self._opp_policy != 'BT':
            r_a, _ = self._opp_policy.predict(self.red_state[None], deterministic=False)
            red_action = self.from_nn2agent(np.array(r_a[0], dtype=np.float32), self.red_agent)

        for _ in range(self.conf.step_length):
            self.blue_agent.apply_action(action)
            if self._opp_policy == 'BT':
                self.red_agent.apply_action()
            else:
                self.red_agent.apply_action(red_action)
            self.update_state()
            self._update_red_state()
            self.done   = self.is_done()
            self.reward = self.get_reward(self.done)
            if self.done:
                break

        truncated = self.max_episode_time_passed()
        outcome = ''
        if self.done:
            if self.red_agent.healthPoints <= 0.0:
                outcome = 'win'
            elif self.blue_agent.healthPoints <= 0.0:
                outcome = 'loss'
            else:
                outcome = 'timeout'
        return (self.state, self.reward, self.done, truncated,
                {'done': self.done, 'trunk': truncated, 'outcome': outcome})

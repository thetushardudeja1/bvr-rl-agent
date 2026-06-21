"""
BaseEnv.py
==========
BVR Gym main environment.

Fixes applied:
    Fix 1: Episode length 16 min (baseEnv_conf.py)
    Fix 2: Timeout reward = -50 not -200 (stops policy freeze in validate)
    Fix 3: is_done() waits for missile to complete after Blue OR Red dies
    Fix 4: Step rewards scaled down 10x (prevents reward hacking)
    Fix 5: 15M steps in main.py
    Fix 6: Missile observations enabled (own + enemy)
    Fix 7: Kill range reward — bonus inside 20km, penalty beyond 80km
    Fix 8: ent_coef=0.01 in main.py (prevents entropy collapse)
    Fix 9: is_done() also waits after Red dies for clean TacView
    Bug fix: BVREnv_BTvsBT agent_name = "BT_blue"
    Bug fix: done reset in reset()
"""

import numpy as np
import pymap3d as pm
import gymnasium as gym
from gymnasium import spaces

from jsb_gym.agents.config import blue_agent, red_agent
from jsb_gym.agents.agents import RLBVRAgent, BTBVRAgent

from jsb_gym.utils.geospatial import (
    dinstance_between_agents,
    bearing_between_agents,
    relative_bearing_between_agents,
    to_360
)
from jsb_gym.utils.loggers import TacviewLogger
from jsb_gym.utils.scale import scale_between_inv, scale_between
from jsb_gym.utils.wez import max_launch_range_m, no_escape_range_m
from jsb_gym.bts.bts import BVRBT


class BVRBase(gym.Env):

    def __init__(self, conf):
        super().__init__()
        self.conf      = conf
        self.obs_shape = conf.observation_shape
        self.act_shape = conf.action_shape

        self.observation_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=self.obs_shape,
            dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=self.act_shape,
            dtype=np.float32
        )

        self.state          = None
        self.done           = False
        self.tacview_logger = None
        self.observation    = {}

        # Reward shaping state
        self.prev_dist                = None
        self.missile_fired            = False
        self.enemy_missile_was_active = False
        self.red_kill_reward_given    = False
        self._caps_prev               = None
        self._wasted_shots            = 0

    def randomize_initial_conditions(self):
        """Domain randomization of the engagement geometry. Without this,
        every episode ever flown was the same frozen head-on setup at 89km —
        the agent was a one-scenario specialist. Ranges are configurable via
        conf.ic_ranges; disabled when conf.randomize_ic is False."""
        if not getattr(self.conf, 'randomize_ic', False):
            return
        rng = self.np_random
        r = self.conf.ic_ranges

        # blue anchored near the map origin; red placed at random range/bearing
        blue_lat, blue_long = 58.0, 18.0
        rng_m  = rng.uniform(*r['range_m'])
        brg    = rng.uniform(0.0, 360.0)
        # flat-earth offsets are fine at <150km scales
        dlat   = rng_m * np.cos(np.radians(brg)) / 111_320.0
        dlong  = rng_m * np.sin(np.radians(brg)) / (111_320.0 * np.cos(np.radians(blue_lat)))
        red_lat, red_long = blue_lat + dlat, blue_long + dlong

        # headings: roughly toward each other, +- aspect jitter
        jit = r['heading_jitter_deg']
        blue_head = (brg + rng.uniform(-jit, jit)) % 360.0
        red_head  = (brg + 180.0 + rng.uniform(-jit, jit)) % 360.0

        self.blue_agent.simObj.reset(
            lat=blue_lat, long=blue_long,
            alt=rng.uniform(*r['alt_m']), vel=rng.uniform(*r['vel_mps']),
            heading=blue_head)
        self.red_agent.simObj.reset(
            lat=red_lat, long=red_long,
            alt=rng.uniform(*r['alt_m']), vel=rng.uniform(*r['vel_mps']),
            heading=red_head)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.blue_agent = RLBVRAgent(blue_agent, self)
        self.red_agent  = BTBVRAgent(red_agent,  self)
        self.red_agent.load_BT(BVRBT)

        self.all_agents = [self.blue_agent, self.red_agent]
        self.blue_agent.set_target(self.red_agent)
        self.red_agent.set_target(self.blue_agent)
        self.randomize_initial_conditions()

        # Per-episode red firing doctrine: vary when red shoots so the threat
        # geometry (range/energy) differs episode to episode, forcing the learner
        # to develop more than the single drag evasion. Disabled by default;
        # training configs set conf.red_doctrine_randomize = True.
        if getattr(self.conf, 'red_doctrine_randomize', False):
            lo, hi = getattr(self.conf, 'red_launch_scale_range', (0.8, 1.4))
            self.red_agent.launch_range_scale = float(self.np_random.uniform(lo, hi))
        else:
            self.red_agent.launch_range_scale = 1.0

        self.done           = False
        self.tacview_logger = None

        self.prev_dist                = None
        self.missile_fired            = False
        self.enemy_missile_was_active = False
        self.red_kill_reward_given    = False
        self._caps_prev               = None
        self._wasted_shots            = 0
        self._phi_prev                = None   # PBRS potential of previous step
        self._blue_shots              = 0      # blue launches counted for shot bonus

        self.update_state()
        return self.state, {}

    def log_tacview(self):
        if self.conf.tacview_output_dir is not None:
            if self.tacview_logger is None:
                self.tacview_logger = TacviewLogger(self)
            elif self.done:
                self.tacview_logger.save_logs()
            else:
                self.tacview_logger.log_flight_data()

    def update_state(self):
        self.update_observation()
        obs_nn = self.from_obs2nn(self.blue_agent)

        if self.state is None:
            self.state = np.tile(obs_nn, (self.obs_shape[0], 1))
        else:
            self.state = np.roll(self.state, shift=-1, axis=0)
            self.state[-1, :] = obs_nn

    def update_observation(self):
        self.observation['bearing']  = to_360(
            bearing_between_agents(self.blue_agent, self.blue_agent.target)
        )
        self.observation['heading']  = self.blue_agent.simObj.get_psi()
        self.observation['mach']     = self.blue_agent.simObj.get_mach()
        self.observation['altitude'] = self.blue_agent.simObj.get_altitude()
        self.observation['d']        = dinstance_between_agents(
            self.blue_agent, self.blue_agent.target
        )
        self.observation['enemy_bearing']  = to_360(
            bearing_between_agents(self.red_agent, self.red_agent.target)
        )
        self.observation['enemy_heading']  = self.red_agent.simObj.get_psi()
        self.observation['enemy_mach']     = self.red_agent.simObj.get_mach()
        self.observation['enemy_altitude'] = self.red_agent.simObj.get_altitude()

        # FIX 6 — enable real missile observations (was always 0)
        self.observation['own_missile_active'] = (
            1.0 if self.blue_agent.is_own_missile_active() else 0.0
        )
        self.observation['enemy_missile_active'] = (
            1.0 if self.red_agent.is_own_missile_active() else 0.0
        )

        # Incoming-threat geometry: so the policy can SENSE an inbound missile
        # and learn to beam/notch it (it cannot defeat what it cannot perceive).
        #   threat_bearing : where the missile is, relative to blue's nose
        #   threat_range   : how close the missile is (m)
        #   threat_los     : blue's along-LOS speed to the missile (m/s); driving
        #                    this toward 0 IS the notch (perpendicular = beaming)
        mis = self.red_agent.get_active_missile()
        if mis is not None:
            b = self.blue_agent.simObj
            e, n, u = pm.geodetic2enu(
                mis.get_lat_gc_deg(), mis.get_long_gc_deg(), mis.get_altitude(),
                b.get_lat_gc_deg(), b.get_long_gc_deg(), b.get_altitude())
            self.observation['threat_bearing'] = to_360(np.degrees(np.arctan2(e, n)))
            self.observation['threat_range']   = float(np.sqrt(e*e + n*n + u*u))
            self.observation['threat_los']     = mis.target_los_speed()
        else:
            self.observation['threat_bearing'] = self.observation['heading']  # neutral (rel brg 0)
            self.observation['threat_range']   = 120e3
            self.observation['threat_los']     = 0.0

    def step(self, action):
        # CAPS temporal-smoothness penalty on RAW consecutive policy outputs
        # (Mysore et al. 2021) — punishes command flip-flopping at the source.
        caps_pen = 0.0
        caps_coef = getattr(self.conf, 'caps_coef', 0.0)
        if caps_coef > 0.0:
            if self._caps_prev is not None:
                caps_pen = caps_coef * float(np.sum(np.abs(action - self._caps_prev)))
            self._caps_prev = np.array(action, copy=True)

        action = self.from_nn2agent(action, self.blue_agent)

        for _ in range(self.conf.step_length):
            self.blue_agent.apply_action(action)
            self.red_agent.apply_action()
            self.update_state()
            self.done   = self.is_done()
            self.reward = self.get_reward(self.done)
            if self.done:
                break

        self.reward -= caps_pen

        truncated = self.max_episode_time_passed()

        # Explicit terminal outcome for logging (kept distinct even though
        # timeout and loss now share the same -200 reward).
        outcome = ''
        if self.done:
            if self.red_agent.healthPoints <= 0.0:
                outcome = 'win'
            elif self.blue_agent.healthPoints <= 0.0:
                outcome = 'loss'
            else:
                outcome = 'timeout'

        return (
            self.state,
            self.reward,
            self.done,
            truncated,
            {'done': self.done, 'trunk': truncated, 'outcome': outcome}
        )

    def from_obs2nn(self, agent):
        bearing_sin = np.sin(np.radians(self.observation['bearing']))
        bearing_cos = np.cos(np.radians(self.observation['bearing']))
        heading_sin = np.sin(np.radians(self.observation['heading']))
        heading_cos = np.cos(np.radians(self.observation['heading']))

        mach     = scale_between(self.observation['mach'], 0.1, 1.5)
        altitude = scale_between(
            self.observation['altitude'],
            agent.simObj.conf.aircraft_limits.alt_min,
            agent.simObj.conf.aircraft_limits.alt_max
        )
        d = scale_between(self.observation['d'], 0.0, 120e3)

        enemy_bearing_sin = np.sin(np.radians(self.observation['enemy_bearing']))
        enemy_bearing_cos = np.cos(np.radians(self.observation['enemy_bearing']))
        enemy_heading_sin = np.sin(np.radians(self.observation['enemy_heading']))
        enemy_heading_cos = np.cos(np.radians(self.observation['enemy_heading']))

        enemy_mach     = scale_between(self.observation['enemy_mach'], 0.1, 1.5)
        enemy_altitude = scale_between(
            self.observation['enemy_altitude'],
            agent.simObj.conf.aircraft_limits.alt_min,
            agent.simObj.conf.aircraft_limits.alt_max
        )

        # incoming-threat features (relative bearing so notch geometry is direct)
        threat_rel = np.radians(self.observation['threat_bearing']
                                - self.observation['heading'])
        threat_bearing_sin = np.sin(threat_rel)
        threat_bearing_cos = np.cos(threat_rel)
        threat_range = scale_between(self.observation['threat_range'], 0.0, 120e3)
        # along-LOS speed normalized to [-1,1]; ~ -1 == fully notched (beaming)
        threat_los = np.clip(self.observation['threat_los'] / 400.0, 0.0, 1.0) * 2.0 - 1.0

        obs = np.array([
            bearing_sin, bearing_cos,
            heading_sin, heading_cos,
            mach, altitude, d,
            enemy_bearing_sin, enemy_bearing_cos,
            enemy_heading_sin, enemy_heading_cos,
            enemy_mach, enemy_altitude,
            self.observation['own_missile_active'],
            self.observation['enemy_missile_active'],
            threat_bearing_sin, threat_bearing_cos,
            threat_range, threat_los,
        ], dtype=np.float32)
        # NaN/inf guard (JSBSim can emit NaN at envelope edges -> would
        # silently collapse training) + clip to declared Box(-1,1) space.
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        np.clip(obs, -1.0, 1.0, out=obs)
        return obs

    def from_nn2agent(self, action, agent):
        # copy first: mutating in place would corrupt the caller's array
        # (SB3 hands us a view into its rollout buffer)
        action = np.array(action, dtype=np.float64, copy=True)
        action[0] = scale_between_inv(
            action[0],
            a_min=agent.simObj.conf.aircraft_limits.psi_min,
            a_max=agent.simObj.conf.aircraft_limits.psi_max
        )
        action[1] = scale_between_inv(
            action[1],
            a_min=agent.simObj.conf.aircraft_limits.alt_min,
            a_max=agent.simObj.conf.aircraft_limits.alt_max
        )
        action[2] = 0.49 if action[2] <= 0.0 else 0.69
        return action

    def get_reward(self, is_done):
        """
        FIX 2: Timeout = -50 not -200
        FIX 4: Step rewards scaled down 10x
        FIX 7: Kill range reward
        FIX 9: Return win reward as soon as Red dies even if missile still flying
        """

        # --- Terminal rewards ---
        if is_done:
            if self.red_agent.healthPoints <= 0.0:
                # FIX: do NOT re-award if the kill bonus was already paid
                # mid-episode (was double-counting +200 -> +400 per kill)
                return 0.0 if self.red_kill_reward_given else 200.0
            elif self.blue_agent.healthPoints <= 0.0:
                return -200.0   # Blue dies
            else:
                # FIX: timeout = mission failure (BT still alive) = loss.
                # Was -50, which let the agent farm "safe" timeouts instead of
                # engaging. Matches H3E attack semantics ("opponent alive" = -1).
                return -200.0   # timeout = failed to kill = loss

        # Give win reward immediately when Red dies (before missile finishes)
        if self.red_agent.healthPoints <= 0.0 and not self.red_kill_reward_given:
            self.red_kill_reward_given = True
            return 200.0

        reward = 0.0

        # FIX: per-step time penalty — punishes dragging the episode out so the
        # agent hunts for a FAST kill instead of stalling to a (discounted) timeout.
        reward -= getattr(self.conf, 'time_penalty', 0.0)

        dist = self.observation['d']

        # === Potential-based reward shaping (Ng et al. 1999) ===========
        # The OLD reward paid +0.05 every time blue evaded a missile while the
        # offensive terms were ~0.0001 — defense out-paid offense ~500x, which
        # trained the circling survival specialist. We replace the ad-hoc terms
        # and the gameable evade bonus with a single potential Phi(s) over
        # OFFENSIVE geometry. PBRS (gamma*Phi' - Phi) is policy-invariant, so it
        # densifies the path to a kill WITHOUT being farmable by stalling
        # (the literature's "potential reward" + "anneal exploration" prescription;
        #  PBRS needs no annealing because it is invariant by construction).
        #   proximity : 1 when blue is at/inside its own launch range on red
        #   nose      : 1 when blue points at red (firing geometry), 0 tail-on
        rmax = max_launch_range_m(self.blue_agent, self.red_agent)
        bearing = abs(relative_bearing_between_agents(self.blue_agent, self.red_agent))
        prox = float(np.clip(1.0 - dist / max(rmax, 1.0), 0.0, 1.0))
        nose = 0.5 * (1.0 + np.cos(np.radians(bearing)))
        # COUPLED advantage (lit: angle x distance trains better than additive):
        # potential is high only when blue is simultaneously IN RANGE and AIMED
        # — i.e. an actual firing solution — so it cannot be farmed by being
        # nose-on-but-far or close-but-unaimed (the additive form's loophole).
        phi  = prox * nose
        if self._phi_prev is not None:
            reward += self.conf.pbrs_coef * (self.conf.pbrs_gamma * phi - self._phi_prev)
        self._phi_prev = phi
        self.prev_dist = dist

        # === Event rewards =============================================
        # High-Pk shot bonus (KAERS-style, larger at closer range): reward blue
        # for taking a shot from INSIDE the no-escape zone — a real firing
        # solution, not a marginal spray. Paid once per new launch.
        bsh = sum(1 for m in self.blue_agent.ammo.values()
                  if (not m.is_ready_to_launch()))
        if bsh > getattr(self, '_blue_shots', 0):
            rne = no_escape_range_m(self.blue_agent, self.red_agent)
            if dist <= rne:
                reward += self.conf.shot_in_wez_bonus           # decisive shot
            self._blue_shots = bsh

        # Wasted-shot penalty: a missile that dies without a kill. Teaches
        # conservation and offsets any incentive to spray for the shot bonus.
        wasted = sum(1 for m in self.blue_agent.ammo.values()
                     if (not m.is_alive()) and (not m.is_target_hit()))
        if wasted > self._wasted_shots:
            reward -= 2.0 * (wasted - self._wasted_shots)
            self._wasted_shots = wasted

        # NOTE: the repeatable evade bonus (+0.05) is intentionally GONE — it
        # was the primary driver of the passive circling behaviour. Surviving a
        # shot is now its own reward only through staying alive to win.
        self.enemy_missile_was_active = self.red_agent.is_own_missile_active()

        return reward

    def is_done(self):
        """
        FIX 3 + FIX 9:
        - Wait for missile to finish after Blue dies
        - Wait for missile to finish after Red dies (for clean TacView)
        - This ensures missile trajectory is fully logged before episode ends
        """

        # Red killed — wait for Blue's missile to finish for clean TacView
        if self.red_agent.healthPoints <= 0.0:
            if not self.blue_agent.is_own_missile_active():
                return True  # missile done — end episode
            return False     # missile still flying — keep logging

        # Blue killed — wait for missile to complete
        if self.blue_agent.healthPoints <= 0.0:
            if not self.blue_agent.is_own_missile_active():
                return True
            return False

        # Timeout — but wait if missile is still active
        if self.max_episode_time_passed():
            if self.blue_agent.is_own_missile_active():
                return False  # missile still flying — keep going
            return True

        # DRY STALEMATE: both magazines spent, nothing airborne, both alive —
        # the engagement is decided (we model no guns). Ends episodes that
        # previously circled at 1km for up to 10 more sim-minutes.
        if self.blue_agent.healthPoints > 0 and self.red_agent.healthPoints > 0:
            blue_dry = all((not m.is_ready_to_launch()) and (not m.is_active())
                           for m in self.blue_agent.ammo.values())
            if blue_dry:
                red_dry = all((not m.is_ready_to_launch()) and (not m.is_active())
                              for m in self.red_agent.ammo.values())
                if red_dry:
                    return True

        return False

    def max_episode_time_passed(self):
        return (
            self.blue_agent.simObj.get_sim_time_sec()
            >= self.conf.max_episode_time
        )


class BVREnv_BTvsBT(BVRBase):

    def __init__(self, conf):
        super().__init__(conf)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        blue_agent.agent_name = "BT_blue"
        self.blue_agent = BTBVRAgent(blue_agent, self)
        self.red_agent  = BTBVRAgent(red_agent,  self)

        self.blue_agent.load_BT(BVRBT)
        self.red_agent.load_BT(BVRBT)

        self.all_agents = [self.blue_agent, self.red_agent]
        self.blue_agent.set_target(self.red_agent)
        self.red_agent.set_target(self.blue_agent)
        self.randomize_initial_conditions()

        self.update_state()
        return self.state, {}

    def step(self, action=None):
        self.blue_agent.apply_action()
        self.red_agent.apply_action()
        self.update_state()

        # FIX: terminate on kill (via is_done, which also waits for missiles
        # to finish) — previously only the 16-min timeout ended the episode,
        # so demo datasets were polluted with post-kill flying.
        self.done   = self.is_done()
        self.reward = 0

        truncated = self.max_episode_time_passed()
        outcome = ''
        if self.done:
            if self.red_agent.healthPoints <= 0.0:
                outcome = 'win'
            elif self.blue_agent.healthPoints <= 0.0:
                outcome = 'loss'
            else:
                outcome = 'timeout'
        return (
            self.state,
            self.reward,
            self.done,
            truncated,
            {'done': self.done, 'trunk': truncated, 'outcome': outcome}
        )

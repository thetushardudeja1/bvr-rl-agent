"""
baseEnv_conf.py
===============
Environment configuration.
Fix 1: max_episode_time = 60*16 (16 minutes — confirmed from paper text)
"""

import os

tacview_output_dir = os.path.join('data_output', 'tacview')

observation_shape = (40, 19)
# 15 base features + 4 incoming-threat features (threat bearing sin/cos, range,
# along-LOS speed) so the policy can sense an inbound missile and learn to
# beam/notch it — the second evasion that breaks the monotone turn-and-run.
# 4th dim = launch intent (>0 fire): the agent OWNS the trigger (H3E weapon
# action). Before this, a hardcoded rule dumped the magazine at max range —
# measured: blue fired 4 shots at ~71km (Pk~0) while BT killed at 20-30km.
action_shape      = (4,)

max_episode_time = 60 * 16  # FIX 1 — 16 minutes (paper confirmed)

step_length = 1

# Per-step time penalty — discourages stalling to a (discounted) timeout and
# pushes for fast kills. Tunable; -0.01/step ~= -19 over a full 1920-step episode.
time_penalty = 0.01

# Domain randomization of engagement geometry (range/bearing/heading/alt/vel).
# Without it every episode is the same frozen head-on setup at 89km.
randomize_ic = True

# CAPS temporal smoothness penalty coefficient (0 disables). At 0.05/unit L1
# a typical episode pays ~5-25 — a few % of the +-200 terminal reward.
caps_coef = 0.05
ic_ranges = {
    'range_m':            (60e3, 100e3),
    'alt_m':              (5e3, 11e3),
    'vel_mps':            (250.0, 400.0),
    'heading_jitter_deg': 45.0,   # deviation from pure head-on aspect
}

# Per-episode randomization of red's firing doctrine (training only — eval keeps
# the deterministic smart-shooter at scale 1.0). Toggled ON by train_rir
# --red_doctrine so the threat arrives at varied ranges/energy and the learner
# must develop more than the single drag evasion.
red_doctrine_randomize = False
red_launch_scale_range = (0.9, 1.25)   # milder than (0.8,1.4): keep variety learnable

# Potential-based reward shaping (Ng et al. 1999) toward offensive geometry —
# replaces the ad-hoc step rewards and the gameable +0.05 evade bonus that
# trained the circling survivor. PBRS is policy-invariant so it cannot be farmed
# by stalling. Telescoped over an episode it is bounded ~pbrs_coef*(w_prox+w_nose).
pbrs_coef   = 12.0
pbrs_gamma  = 0.999
pbrs_w_prox = 1.0    # being at/inside own launch range on red
pbrs_w_nose = 1.0    # pointing at red (firing geometry)
# KAERS-style decisive-shot bonus: blue fires from inside the no-escape zone.
shot_in_wez_bonus = 5.0
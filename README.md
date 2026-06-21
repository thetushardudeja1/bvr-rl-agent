<h1 align="center">Sensor-Aware Reinforcement Learning for Beyond-Visual-Range Air Combat</h1>

<p align="center">
  <em>A PPO agent that learns to defeat radar-guided missiles through sensor exploitation (Doppler-notch beaming) and energy-maneuver tactics in a high-fidelity 6-DOF flight simulator — not by degenerate "run-away" policies.</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.11-blue.svg" alt="Python 3.11">
  <img src="https://img.shields.io/badge/PyTorch-%E2%89%A52.4-ee4c2c.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/algorithm-PPO%20%7C%20BC%20%7C%20DAPG%20%7C%20PFSP-8957e5.svg" alt="Algorithms">
  <img src="https://img.shields.io/badge/sim-JSBSim%206--DOF-1f6feb.svg" alt="JSBSim 6-DOF">
  <img src="https://img.shields.io/badge/SB3-Stable--Baselines3-44a833.svg" alt="SB3">
  <img src="https://img.shields.io/badge/license-GPL--3.0-green.svg" alt="License">
</p>

---

## Demo

<p align="center">
  <img src="media/demo.gif" alt="BVR engagement with live Weapon-Employment-Zone overlay in Tacview" width="100%">
</p>

<p align="center">
  <em>The learned policy (Blue) beams an incoming radar-guided missile into the Doppler main-lobe-clutter notch, breaking the seeker lock, while the adversary's Weapon-Employment-Zone (the colored kill-probability petal) is rendered live in the 3-D world.</em><br>
  ▶ <a href="media/demo.mp4">Full video (MP4)</a>
</p>

---

## Abstract

We study 1-v-1 Beyond-Visual-Range (BVR) air combat as a continuous-control
reinforcement-learning problem in a high-fidelity, six-degree-of-freedom flight
simulator (**JSBSim**). An ego agent ("Blue", PPO) engages a scripted
behaviour-tree adversary ("Red") armed with active-radar-homing missiles. The
core contribution is an environment in which **missile evasion requires
exploiting sensor physics**: we model the **Doppler main-lobe-clutter notch**, so
a target flying perpendicular to a missile's line of sight (a *beam* maneuver)
forces the seeker to lose lock. Under this model the previously dominant
degenerate strategy — monotone "turn and run" — is no longer optimal, and the
agent is driven to discover a repertoire of doctrine-accurate tactics:
mid-course **beaming** (sensor-level defeat), **energy dragging** (kinematic
defeat), and **terminal-beam crossing** shots. Training combines **behavior
cloning**, **DAPG-style demonstration-anchored PPO**, and **prioritized
fictitious self-play**. We additionally characterize the missile's
**Weapon-Employment-Zone (WEZ)** and **kinematic reach envelope** by Monte-Carlo
simulation, and render both inside Tacview for interpretability.

---

## Key contributions

1. **A sensor-physics-driven evasion environment.** A Doppler-notch seeker model
   (lock loss under sustained beaming) plus an incoming-threat observation,
   turning missile defense from a kinematics-only problem into a
   *sensor + kinematics* problem.
2. **A demonstration-anchored training pipeline (BC → DAPG-PPO → PFSP).**
   Behavior cloning from behaviour-tree rollouts initializes the policy; a
   decaying DAPG auxiliary loss keeps it near competent behavior during RL; a
   prioritized fictitious self-play league pushes beyond the scripted opponent.
3. **Emergent tactical diversity.** The learned policy beams in the majority of
   engagements and *mixes* beaming with energy-dragging — eliminating the
   degenerate run-away policy of radar-free baselines.
4. **Quantitative weapon characterization.** Monte-Carlo measurement of `R_max`
   and the no-escape range `R_ne` across (range × aspect × altitude × speed),
   showing `R_ne/R_max` is strongly aspect-dependent (≈0.63 head-on, ≈0.33
   off-aspect), contradicting the textbook flat-0.6 heuristic.

---

## Problem formulation

A partially observable continuous-control MDP solved with on-policy RL.

| Component | Specification |
|---|---|
| **Dynamics** | JSBSim 1.3.1, full 6-DOF F-16 for both aircraft and missile airframes. |
| **Observation** | 19-dim per step: relative engagement geometry (range, aspect, closure, altitude, energy) + an **incoming-missile threat block** (threat bearing sin/cos, threat range, along-LOS closure). Temporally stacked, shape `(40, 19)`. |
| **Action** | Continuous flight commands (target heading / altitude / throttle) plus a discrete fire gate. |
| **Reward** | Potential-based shaping (policy-invariant, Ng et al. 1999): `Φ = proximity × nose-on`, `proximity = clip(1 − d/R_max, 0, 1)`, `nose-on = ½(1 + cos θ_bearing)`; `r += k·(γΦ′ − Φ)` plus a no-escape-zone shot bonus and a wasted-shot penalty. The gameable "evade" bonus that produced run-away behavior was removed. |
| **Termination** | Kill (either side) or timeout; episodes wait for in-flight missiles to resolve. |

---

## Method

### Threat / sensor model (the Doppler notch)
An active-radar seeker rejects ground clutter by Doppler: it filters out returns
whose **along-line-of-sight closing speed** is near zero. A target beaming
(flying ⟂ to the LOS) therefore drops its return into the rejected clutter band
and **breaks the lock**. Implemented in `jsb_gym/simObjects/missiles.py`:
lock is lost when along-LOS speed `< v_notch = 75 m/s` for
`N_notch = 20` consecutive samples inside the active-seeker window. On lock loss
the missile **coasts ballistically** (motor already burned out) rather than
teleporting/vanishing, for physically faithful trajectories.

### Missile model
Boost (~2 s, → ~Mach 3.5–4) → unpowered **loft/cruise** at altitude → terminal
**dive**; **Proportional Navigation** guidance (`N = 3`); kill via a **300 m
proximity (fragmentation) fuze** evaluated at the analytic closest-point-of-
approach, so discretization never causes phantom skips or orbits.

### Algorithms & training pipeline
| Stage | Algorithm | Purpose |
|---|---|---|
| 1. Demonstrations | Parallel behaviour-tree rollouts (win-filtered) | Collect expert-ish trajectories. |
| 2. Behavior Cloning | Supervised pre-training (`bc_model_*.zip`) | Warm-start the policy. |
| 3. **RIR** | **PPO** + **DAPG**-style auxiliary BC loss with weight decaying `1.0 → 0.1` | Demonstration-anchored RL; stable improvement without forgetting. |
| 4. League | **Prioritized Fictitious Self-Play** (AlphaStar-style) | Generalize beyond the scripted opponent. |

**Representative PPO hyperparameters** (`experiments/rir/train_rir.py`):
`γ = 0.999`, `lr = 1e-4`, `n_steps = 128`, `batch = 8192`, `ent_coef = 3e-3`,
parallel envs = 128–360 (custom chunked `SubprocVecEnv`), critic value-warmup
`1e6` steps, total budget `12–15e6` steps, device CPU.

---

## Baselines & results

**Evaluation protocol.** Deterministic policy, 40 episodes vs. the standard
behaviour-tree adversary; we report win rate and the fraction of engagements in
which the notch defense is used.

| Policy | Win rate | Notch usage | Description |
|---|---:|---:|---|
| Radar-free champion (15-dim obs) | ~54% | — | No sensor model; collapses to a single run-away tactic. |
| Notch-aware v1 | 18% | 65% | First sensor-aware policy. |
| **Notch-aware v2 (released)** | **28%** | **72%** | Best model; varied beam + drag evasion. |
| Notch-aware v3 | 18% | 57% | Coupled-reward rebalance. |
| PFSP self-play league | ~32% | — | Best vs. behaviour tree. |

**Interpretation.** Introducing the sensor model makes the task strictly harder
than the radar-free setting (larger observation, defense-dominant dynamics), so
raw win rate drops relative to the 54% radar-free baseline. The scientific result
is **behavioral**: the degenerate run-away policy is eliminated and replaced by a
*tactically diverse, sensor-aware* policy that defeats missiles by sensor and
kinematic means. Reproduce with `RUN_eval.py` (or `python -m experiments.rir...`).

### Weapon-Employment-Zone characterization
Monte-Carlo launch grid over (range × aspect × altitude × shooter-speed) against
both non-maneuvering and best-evading targets yields kill-probability fields and
the boundaries **`R_max`** (kinematic reach) and **`R_ne`** (no-escape). Finding:
`R_ne/R_max ≈ 0.63` head-on but `≈ 0.33` off-aspect — i.e. the launch envelope is
strongly aspect-dependent. Data in `experiments/wez/`; sweep in
`extra_scripts/wez_sweep_fine.py`.

---

## Reproducibility

```bash
pip install -r requirements.txt        # NumPy 2.x and torch >= 2.4 required
```

| Task | Command | (or one-click) |
|---|---|---|
| Evaluate the released policy | `python -m extra_scripts.eval_notch --model experiments/rir/rir_notch2_best --eps 40` | `RUN_eval.py` |
| Train (BC-anchored PPO) | `python -m experiments.rir.train_rir --bc experiments/rir/bc_model_notch2 --data experiments/rir/bt_dataset_notch2.npz --red_doctrine ...` | `RUN_train.py` |
| Generate engagement replays | — | `RUN_tacview.py` |
| Render the WEZ overlay | — | `RUN_wez_tacview.py` |

> The `RUN_*.py` scripts are thin, zero-argument wrappers (handy for running
> from an editor on machines with restricted terminals); the underlying entry
> points are standard Python modules.

Replays are written as Tacview `.acmi` files (open in [Tacview](https://www.tacview.net/)).
Full design/parameter documentation: **[`DETAILED_README.md`](DETAILED_README.md)**.

---

## Repository structure
```
.
├─ jsb_gym/                         simulation + Gymnasium environment
│  ├─ simObjects/missiles.py        boost/loft, Proportional Navigation, CPA fuze, Doppler notch, coast
│  ├─ envs/BaseEnv.py               observation, potential-based reward, episode logic
│  ├─ envs/vec_env.py               chunked SubprocVecEnv (high-throughput parallel sim)
│  ├─ agents/agents.py              no-escape-gated launch logic; bts/bts.py  scripted adversary
│  └─ utils/                        guidance (PN), PID autopilots, Tacview logger
├─ experiments/
│  ├─ rir/                          BC → DAPG-PPO → self-play pipeline + released model (rir_notch2_best.zip)
│  └─ wez/                          Monte-Carlo WEZ / reach envelopes (JSON)
├─ extra_scripts/                   evaluation, telemetry, WEZ sweep, replay generators
├─ aircraft/ engine/ systems/ scripts/   JSBSim 6-DOF model data
└─ RUN_*.py                         one-click wrappers
```

---

## Tech stack
Python 3.11 · PyTorch (≥2.4) · Stable-Baselines3 (PPO) · Gymnasium ·
JSBSim 1.3.1 (6-DOF FDM) · NumPy 2 · pymap3d · Tacview (ACMI).

---

## Selected references
- Schulman et al., *Proximal Policy Optimization Algorithms*, 2017.
- Rajeswaran et al., *Learning Complex Dexterous Manipulation… (DAPG)*, 2018.
- Vinyals et al., *Grandmaster level in StarCraft II (AlphaStar / PFSP)*, Nature 2019.
- Ng, Harada, Russell, *Policy invariance under reward transformations
  (potential-based shaping)*, ICML 1999.
- Raffin et al., *Stable-Baselines3*, JMLR 2021.
- Berndt, *JSBSim: An Open Source Flight Dynamics Model*, 2004.

---

## Citation
```bibtex
@software{dudeja_bvr_rl_2026,
  author  = {Dudeja, Tushar},
  title   = {Sensor-Aware Reinforcement Learning for Beyond-Visual-Range Air Combat},
  year    = {2026},
  url     = {https://github.com/thetushardudeja1/Beamrider}
}
```

## License
Released under the **GNU GPL-3.0** (see [`LICENSE`](LICENSE)). Built on the
open-source **JSBSim** flight dynamics model and **Stable-Baselines3**.

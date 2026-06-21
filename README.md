<h1 align="center">BVR Air-Combat RL Agent</h1>

<p align="center">
  <em>A reinforcement-learning fighter pilot that survives missiles with sensors and maneuver — not by running away.</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.11-blue.svg" alt="Python 3.11">
  <img src="https://img.shields.io/badge/PyTorch-%E2%89%A52.4-ee4c2c.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/Stable--Baselines3-PPO-44a833.svg" alt="SB3 PPO">
  <img src="https://img.shields.io/badge/JSBSim-1.3.1-1f6feb.svg" alt="JSBSim">
  <img src="https://img.shields.io/badge/NumPy-2.x-013243.svg" alt="NumPy 2.x">
  <img src="https://img.shields.io/badge/license-GPL--3.0-green.svg" alt="License GPL-3.0">
  <img src="https://img.shields.io/badge/replays-Tacview-orange.svg" alt="Tacview">
</p>

---

## Demo

<p align="center">
  <img src="media/demo.gif" alt="BVR engagement with live WEZ overlay in Tacview" width="100%">
</p>

<p align="center">
  <em>Blue (RL agent) beams Red's missiles into the radar Doppler notch while Red's launch zone — the coloured Weapon-Employment-Zone petal — is drawn live in the world and rotates with the aircraft.</em><br>
  ▶ <a href="media/demo.mp4">Watch the full video (MP4)</a>
</p>

---

## Overview

This project trains a **1-versus-1 Beyond-Visual-Range (BVR) air-combat agent**. A
**Blue** F-16, controlled by a PPO neural-network policy, fights a **Red** F-16
driven by a behaviour tree, inside a full **JSBSim 6-DOF** flight simulation with
radar-guided missiles.

The central result: **survival in BVR is governed by sensors and maneuver, not by
range geometry alone.** The agent learns to defeat incoming missiles two
physically distinct ways — by **beaming** them into the radar **Doppler notch**
(a sensor defeat) and by forcing a **terminal crossing the missile can't out-turn**
(a kinematic defeat) — instead of the naive "turn and run" that earlier agents
collapsed to.

> Everything runs from **VS Code's ▶ Run button** — no terminal commands required
> (see [Quick start](#quick-start)).

---

## Highlights

- **Sensor-aware evasion.** A Doppler main-lobe-clutter **notch** model: a beaming
  target drops out of the seeker's clutter filter and breaks lock. The agent
  exploits this in the majority of engagements.
- **Realistic missile model.** Boost (~2 s) → high-altitude **loft/cruise** → terminal
  **dive**, guided by **Proportional Navigation**, killed by a **300 m proximity
  fuze** evaluated at the true closest-point-of-approach.
- **Measured Weapon-Employment-Zone (WEZ).** `R_max` and no-escape `R_ne` ranges
  characterised by Monte-Carlo simulation across range × aspect × altitude × speed.
- **Tacview visualisation.** Replays plus an in-world **WEZ heatmap overlay** (a
  kill-probability petal that rotates with the shooter) and **shrinking missile
  reach bubbles**.
- **Principled firing & reward.** Shots gated by the measured no-escape zone;
  policy-invariant potential-based reward shaping (no gameable bonuses).
- **Runs out of the box.** Trained model included; zero-argument `RUN_*.py`
  entry points; Windows + Linux compatible.

---

## Results

Deterministic evaluation, 40 episodes vs the standard smart Red behaviour tree:

| Model | Win rate | Notch usage | Notes |
|---|---:|---:|---|
| Old champion (no radar, 15-dim) | ~54% | — | baseline; single "run away" tactic |
| Notch retrain v1 | 18% | 65% | first sensor-aware model |
| **Notch retrain v2 (shipped)** | **28%** | **72%** | **best model; varied beam + drag evasion** |
| Notch retrain v3 | 18% | 57% | coupled-reward rebalance |
| Self-play league (vs BT) | ~32% | — | PFSP league |

**Behavioural achievement (the headline):** the original monotone *"always turn
and run"* policy is eliminated. The shipped agent **beams in ~72% of fights** and
**mixes beaming with energy-dragging**, defeating missiles by sensor *and*
kinematic means. Adding the realistic sensor layer made the task strictly harder
than the no-radar baseline; recovering the win rate while keeping this richer,
doctrine-accurate behaviour is the ongoing research direction.

> Re-measure on your machine any time: press ▶ on **`RUN_eval.py`**.

---

## Achievements

- ✅ Implemented a physically-grounded **Doppler-notch seeker** (lock-loss when a
  target beams), with a realistic **ballistic coast** after lock loss.
- ✅ Eliminated the degenerate "turn-and-run" policy; the agent now uses a
  **repertoire of evasions**.
- ✅ Added an **incoming-missile threat block** to the observation so the policy
  can sense and react to a shot (19-dim observation).
- ✅ Built a **behaviour-cloning-anchored RL pipeline (RIR)** — BC warm start +
  decaying DAPG anchor — for stable training from BT demonstrations.
- ✅ Characterised the **WEZ** (`R_max`, `R_ne`) and the **missile reach envelope**
  via Monte-Carlo, and validated `R_ne/R_max` is aspect-dependent (~0.63 head-on,
  ~0.33 off-aspect — not the textbook flat 0.6).
- ✅ Created **Tacview overlays** that render the WEZ and missile reach *in the 3-D
  world* for intuitive analysis.
- ✅ Diagnosed terminal engagements with **per-sub-step telemetry**, explaining
  close-call misses by 3-D geometry (loft/dive vertical offset + terminal beam).

---

## How it works (in one screen)

| Piece | What it does |
|---|---|
| **Agent** | PPO policy (Stable-Baselines3), 19-dim observation incl. incoming-missile geometry. |
| **Missile** | Boost → loft → dive; Proportional Navigation guidance (`N=3`); 300 m proximity (CPA) fuze. |
| **Doppler notch** | Beam (fly ⟂ to the missile's line of sight) → target hides in ground clutter → seeker loses lock → missile coasts blind. |
| **Terminal beam** | Beam at the merge → huge line-of-sight rate → the Mach-2 missile (~1.6 km turn radius) can't pull lead → it passes wide. |
| **WEZ** | `R_max` (vs non-maneuvering) and `R_ne` (vs best-evading) launch envelopes, measured by simulation. |
| **Reward** | Coupled potential `Φ = proximity × nose-on` (policy-invariant) + no-escape-zone shot bonus. |

Full details: see **[`DETAILED_README.md`](DETAILED_README.md)**.

---

## Quick start

### 1. Install
```bash
pip install -r requirements.txt
```
**Requires NumPy 2.x and torch ≥ 2.4** (the trained model was saved with NumPy 2;
it will not load on NumPy 1.x).

### 2. Run — press ▶ in VS Code
Open the folder in VS Code, open one of the entry-point scripts, and click the
**▶ Run** button (or pick it from the *Run and Debug* panel):

| Script | What it does | Output |
|---|---|---|
| `RUN_tacview.py` | Generate winning beam/notch replays | `output_tacview/*.acmi` |
| `RUN_wez_tacview.py` | Generate the WEZ-overlay replay | `output_tacview/WEZ_*.acmi` |
| `RUN_eval.py` | Measure win rate + evasion breakdown | console |
| `RUN_train.py` | Train the Blue agent | `experiments/rir/*` |

Open the `.acmi` files in **[Tacview](https://www.tacview.net/)**. Tweak behaviour
via the `CONFIG` block at the top of each script.

---

## Repository structure
```
.
├─ RUN_tacview.py / RUN_wez_tacview.py / RUN_eval.py / RUN_train.py   ▶ entry points
├─ README.md / DETAILED_README.md / requirements.txt
├─ jsb_gym/                 simulation + RL environment
│  ├─ simObjects/missiles.py        missile: boost, loft, PN, fuze, notch, coast
│  ├─ envs/BaseEnv.py               gym env: observation, reward, step
│  ├─ agents/agents.py              launch logic; bts/bts.py  Red behaviour tree
│  └─ utils/                        guidance, autopilot, Tacview logger
├─ experiments/
│  ├─ rir/                          training pipeline + trained model (rir_notch2_best.zip)
│  └─ wez/                          measured WEZ / reach envelopes (JSON)
├─ aircraft/ engine/ systems/ scripts/   JSBSim model data
└─ extra_scripts/           diagnostic / advanced tools
```

---

## Tech stack
**Python 3.11** · **PyTorch** · **Stable-Baselines3 (PPO)** · **Gymnasium** ·
**JSBSim 1.3.1** (6-DOF flight dynamics) · **NumPy 2** · **pymap3d** · **Tacview** (ACMI replays).

---

## License & acknowledgements
Released under the **GNU GPL-3.0** (see [`LICENSE`](LICENSE)). Built on the
**JSBSim** open-source flight dynamics model and the **Stable-Baselines3** RL
library. Aircraft/engine/systems data are JSBSim model files.

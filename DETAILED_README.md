# BVR Air-Combat RL Agent — Detailed Project Documentation

> A 1-v-1 Beyond-Visual-Range (BVR) air-combat reinforcement-learning project.
> A **Blue** F-16 controlled by a PPO neural-network agent fights a **Red** F-16
> driven by a behaviour tree, inside a JSBSim 6-DOF flight simulation. Blue has
> learned **sensor-aware evasion** — it beams incoming missiles into the radar
> Doppler notch and manages energy — instead of the naive "turn and run".

This is the in-depth reference. For a 30-second "how do I run it" guide, see
**`README.md`**. For the missile-physics explainer suitable for a supervisor,
see `..\..\Downloads\missile_behaviour_report.tex`.

---

## Table of contents
1. [What the project is and why](#1-what-the-project-is-and-why)
2. [Quick start (VS Code, no terminal)](#2-quick-start-vs-code-no-terminal)
3. [Environment & dependencies](#3-environment--dependencies)
4. [Folder / file map](#4-folder--file-map)
5. [The simulation in depth](#5-the-simulation-in-depth)
6. [The missile model](#6-the-missile-model)
7. [The radar Doppler notch & the coast](#7-the-radar-doppler-notch--the-coast)
8. [The RL agent: observation, reward, training](#8-the-rl-agent-observation-reward-training)
9. [Weapon-Employment-Zone (WEZ) & missile envelope](#9-weapon-employment-zone-wez--missile-envelope)
10. [Tacview visualisation](#10-tacview-visualisation)
11. [Results](#11-results)
12. [Key parameters & where to tune them](#12-key-parameters--where-to-tune-them)
13. [Diagnostics & telemetry tools](#13-diagnostics--telemetry-tools)
14. [Windows-compatibility fixes applied](#14-windows-compatibility-fixes-applied)
15. [Troubleshooting](#15-troubleshooting)
16. [Known issues & future work](#16-known-issues--future-work)
17. [Glossary](#17-glossary)

---

## 1. What the project is and why

**Goal.** Train a fighter agent that survives and wins a BVR missile duel by
using *realistic tactics* — not by exploiting the simulator. The headline
scientific point: **survival in BVR is governed by sensors and manoeuvre, not by
range geometry alone.** The agent demonstrates this by defeating missiles two
different ways (sensor defeat via the Doppler notch, and kinematic defeat via a
terminal beam) instead of simply running away.

**Why it matters.** Earlier versions of the agent learned a single degenerate
trick ("always turn and run"). Adding a radar/seeker model with a Doppler notch
forced the agent to develop *multiple* evasions and to fire principled shots,
producing behaviour that matches real BVR doctrine.

---

## 2. Quick start (VS Code, no terminal)

Open this folder in VS Code and press the ▶ **Run** button on one of the four
entry-point scripts at the project root:

| File | Purpose | Output |
|---|---|---|
| `RUN_tacview.py` | Make winning beam/notch replays | `output_tacview/*.acmi` |
| `RUN_wez_tacview.py` | Make the WEZ-overlay replay | `output_tacview/WEZ_*.acmi` |
| `RUN_eval.py` | Measure win rate + evasion breakdown | Run panel |
| `RUN_train.py` | Train the Blue agent | `experiments/rir/*` |

Settings live in the `CONFIG` block near the top of each file. The four are also
available as named configs in the **Run and Debug** panel.

---

## 3. Environment & dependencies

- **Python 3.11** recommended.
- Install once: `pip install -r requirements.txt`
- **Critical version constraints** (see §14):
  - **NumPy 2.x** (the saved models were created with NumPy 2; they will *not*
    load on NumPy 1.x).
  - **torch ≥ 2.4** (older torch is incompatible with NumPy 2).
  - `jsbsim==1.3.1`, `stable-baselines3≥2.3`, `gymnasium`, `pymap3d`,
    `cloudpickle`, `tensorboard`, `matplotlib`.
- **Tacview** (free, tacview.net) to view the `.acmi` replays.

Originally developed under **WSL Ubuntu-22.04** with a conda env (`bvrc`); this
copy is a portable snapshot with the large old checkpoints/datasets removed.

---

## 4. Folder / file map

```
bvr_project/
├─ RUN_tacview.py        ▶ winning beam/notch replays  -> output_tacview/
├─ RUN_wez_tacview.py    ▶ WEZ-overlay replay          -> output_tacview/
├─ RUN_eval.py           ▶ win rate (calls extra_scripts/eval_notch.py)
├─ RUN_train.py          ▶ training (runs experiments.rir.train_rir)
├─ README.md             quick start
├─ DETAILED_README.md    this file
├─ requirements.txt
├─ .vscode/              launch.json + settings.json (Run/Debug configs)
│
├─ jsb_gym/                       the simulation + RL environment
│  ├─ simObjects/
│  │  ├─ missiles.py              AAMBVR: boost, loft, PN, fuze, notch, coast
│  │  ├─ aircraft.py              F16BVR flight object
│  │  ├─ FDMObject.py             JSBSim wrapper base class
│  │  └─ config/AAM_config.py     ALL missile parameters
│  ├─ envs/
│  │  ├─ BaseEnv.py               BVRBase gym env: obs, reward, step, done
│  │  ├─ vec_env.py               ChunkSubprocVecEnv (parallel sims)
│  │  └─ config/baseEnv_conf.py   env config (obs shape, reward weights)
│  ├─ agents/agents.py            Blue/Red agent: missile launch logic
│  ├─ bts/bts.py                  Red behaviour tree
│  └─ utils/
│     ├─ guidance_laws.py         Proportional Navigation (PN)
│     ├─ autopilot.py             aircraft + missile PID autopilots
│     ├─ loggers.py               TacviewLogger (writes .acmi)
│     └─ geospatial.py            distance / bearing helpers
│
├─ experiments/
│  ├─ rir/
│  │  ├─ train_rir.py             main training entry (PPO + BC anchor)
│  │  ├─ behavior_clone.py        behaviour-cloning warm start
│  │  ├─ collect_bt_parallel.py   collect BT demonstrations
│  │  ├─ train_selfplay.py        PFSP self-play league
│  │  ├─ rir_notch2_best.zip      ★ the best trained model (used by defaults)
│  │  ├─ bc_model_notch2.zip      behaviour-cloning warm-start model
│  │  └─ bt_dataset_notch2.npz    BT demonstrations (19-dim)
│  ├─ wez/                        measured envelopes (JSON) — see §9
│  └─ v2/                         self-play env variant
│
├─ aircraft/ engine/ systems/ scripts/   JSBSim model data (XML) — don't move
└─ extra_scripts/                develop/diagnostic/extra-gen scripts (see its README)
```

---

## 5. The simulation in depth

- **Flight model:** JSBSim 1.3.1, full 6-DOF F-16 dynamics for both aircraft and
  for the missile airframe.
- **Engagement:** 1-v-1. Both jets start far apart at BVR range and close in.
  Each carries multiple radar-guided missiles.
- **Blue** = the PPO agent (the learner). **Red** = a behaviour tree (a scripted
  but competent opponent that flies, fires on a no-escape-range gate, and
  defends with beam manoeuvres).
- **Time stepping:** the env steps the policy at a fixed rate; inside each step
  the missile integrates several JSBSim frames (`Sim_time_step`,
  `Control_time_step` in `AAM_config.py`).
- **Episode ends** on a kill (either side) or a timeout; the env waits for
  in-flight missiles to finish so replays are clean.

---

## 6. The missile model

File: `jsb_gym/simObjects/missiles.py`, parameters in
`jsb_gym/simObjects/config/AAM_config.py`.

**Three jobs: fly, see, detonate.**

1. **Fly.** A rocket motor burns for the boost phase
   (`acceleration_stage_in_sec = 2 s`) to ~Mach 3.5–4; afterward the motor is
   **out** and the missile glides, bleeding speed to drag. Energy is everything.
2. **Loft.** For long shots the missile climbs to a high cruise
   (`alt_cruise`, dives at `dive_at`) where drag is low, then dives onto the
   target — the arched trail in Tacview.
3. **See / guide.** A nose radar tracks the target; guidance is **Proportional
   Navigation** (`guidance_laws.py`, gain `N = 3`): it steers to a lead point,
   not at the target. Autopilot is **skid-to-turn** (`autopilot.py`).
4. **Detonate — the 300 m fuze.** Real missiles use a fragmentation warhead +
   proximity fuze; a direct hit is not required. Here `effective_radius = 300 m`:
   a closest pass ≤ 300 m is a kill. It is evaluated at the true
   **closest-point-of-approach (CPA)** within each step (`is_detonated_at_cpa`),
   so a fast missile that skips past the target between samples is still scored
   correctly and never orbits.

---

## 7. The radar Doppler notch & the coast

**The notch (sensor defeat).** A look-down radar separates the target from
ground clutter by Doppler shift (along-line-of-sight closing speed). If the
target flies **perpendicular** to the missile's line of sight (a **beam**
manoeuvre), its along-LOS speed → ~0, its return falls into the rejected
ground-clutter band, and the seeker **drops lock**. Implemented in
`missiles.py:is_notched()` with:
- `notch_vmin = 75 m/s` — along-LOS speed below which the target is "in the beam"
- `notch_count = 20` — consecutive samples (~1.5 s) of beaming needed to break lock
- `notch_range = 20 km`, `notch_min_range = 1.5 km` — the active-seeker window

**The coast (what happens after lock loss).** On a notch the missile does **not**
vanish — it `_coast()`s: motor already out, it freezes the heading it had at
lock-loss and flies straight, bleeding energy until it goes subsonic or hits a
time cap, then falls away. This was added so notched missiles look physical in
Tacview (they sail past, not disappear). Two related fixes ensure correctness:
- a coasting missile is **excluded from re-launch** (`is_ready_to_launch`),
  otherwise the carrier re-fired it and it appeared to teleport back;
- the Tacview logger keeps drawing the missile during the coast.

**Terminal beam (kinematic defeat).** Beaming *at the last second* doesn't break
lock in time, but it creates a huge line-of-sight rotation rate. At ~Mach 2 the
missile's minimum turn radius is ~1.6 km, so it cannot bend onto the crossing
target and passes wide. Same "turn perpendicular" idea, different mechanism.

---

## 8. The RL agent: observation, reward, training

**Algorithm.** PPO (Stable-Baselines3), trained with the **RIR** pipeline:
behaviour-cloning (BC) warm start + a DAPG-style BC anchor that decays during RL
(`bc_start → bc_end`). This keeps the policy near sensible BT-like behaviour
early, then lets RL improve it.

**Observation (19-dim).** `observation_shape = (40, 19)` (a stacked window).
The 19 features include own/relative geometry plus an **incoming-missile threat
block**: threat bearing (sin/cos), threat range, and along-LOS closing speed —
so the policy can *sense and react to* a shot. See
`BaseEnv.py:update_observation` / `from_obs2nn`.

**Reward.** Potential-based reward shaping (policy-invariant, Ng et al.):
a coupled potential `Φ = proximity × nose-on` rewards getting close *and* nose-on
to the enemy; plus a "shot-in-WEZ" bonus for firing inside the measured
no-escape range, and a wasted-shot penalty. The old gameable `+0.05` evade bonus
(which had taught "always run") was **removed**. Weights live in
`baseEnv_conf.py` (`pbrs_coef`, `pbrs_gamma`, `shot_in_wez_bonus`, …).

**Training pipeline** (all in `experiments/rir/`):
1. `collect_bt_parallel.py --wins_only` → BT demonstrations (`bt_dataset_notch2.npz`)
2. `behavior_clone.py` → BC warm start (`bc_model_notch2.zip`)
3. `train_rir.py` (what `RUN_train.py` calls) → PPO + BC anchor → best model
4. (optional) `train_selfplay.py` → PFSP self-play league

**Firing logic** (`agents.py:launch_missile`). Both sides fire only inside a
measured no-escape range (`no_escape_range_m`), scaled per-episode for Red to
diversify its doctrine (`--red_doctrine`). This avoids marginal max-range shots.

---

## 9. Weapon-Employment-Zone (WEZ) & missile envelope

Two different "envelopes", both measured by simulation (data in `experiments/wez/`):

**WEZ — the launch envelope** (around the *shooter*): "if I launch now, where is
it lethal?" Boundaries:
- **R_max** — reach against a *non-maneuvering* target.
- **R_ne** (no-escape) — range inside which even a *best-evading* target cannot
  escape.
- Measured by `extra_scripts/wez_sweep_fine.py` (Monte-Carlo over range × aspect
  × altitude × shooter-speed). Outputs `wez_pk.json` (full Pk grid) and
  `wez_rmax_rne.json` (R_max / R_ne / ratio tables).
- Finding: **R_ne / R_max ≈ 0.63 head-on, ≈ 0.33 off-aspect** — i.e. the common
  "0.6·R_max" rule of thumb only holds head-on.

**Missile envelope — the reachable footprint** (around the *missile in flight*):
"given its current Mach, how far can it still reach?" It **shrinks** as the
missile bleeds energy. Calibrated by `extra_scripts/calibrate_missile_flyout.py`
(`flyout.json`): ~86 km reach at Mach 3.5 → ~25 km at Mach 1.5 → 0 at Mach 1.

---

## 10. Tacview visualisation

`jsb_gym/utils/loggers.py:TacviewLogger` writes `.acmi` text replays. On top of
the aircraft/missile tracks, the WEZ scripts draw the envelope **into the world**
as marker objects anchored to the shooter and re-positioned every frame (so the
zone rotates with the aircraft):
- `RUN_wez_tacview.py` — filled **Pk heatmap petal** (red lethal core → orange →
  yellow fringe), a cyan R_max ring, an animated radar-sweep spoke, and floating
  R_max / R_NE labels.
- `extra_scripts/gen_missile_envelope.py` — each missile's shrinking reach bubble.

Tacview has no polygon primitive, so zones are drawn as dense rings of small
coloured marker objects.

---

## 11. Results

Deterministic evaluation, 40 episodes vs the standard smart Red
(`extra_scripts/eval_notch.py`):

| Model | Win rate | Notch usage | Note |
|---|---|---|---|
| Old champion (15-dim, no radar) | ~54% | — | longer pipeline, no sensor layer |
| notch1_best | 18% | 65% | first notch retrain |
| **notch2_best** (shipped) | **28%** | **72%** | best notch model; varied evasion |
| notch3_best | 18% | 57% | coupled-reward rebalance |
| self-play league (vs BT) | ~32% | — | stalemated (see §16) |

**Headline behavioural result:** the original monotone "turn and run" is gone —
the agent beams in the large majority of fights and mixes beam + drag. Adding the
sensor layer made the task harder than the no-radar 54% champion; recovering that
win rate with the richer (and more realistic) behaviour is ongoing.

> Note: the CPA fuze + coast fixes (§14) changed terminal hit/miss accounting,
> so a fresh 40-episode eval is the way to get the exact current number — run
> `RUN_eval.py`.

---

## 12. Key parameters & where to tune them

**`jsb_gym/simObjects/config/AAM_config.py`** (missile):
- `acceleration_stage_in_sec` — boost duration (2 s)
- `N` — PN guidance gain (3)
- `dive_at`, `alt_cruise` — loft/dive profile (lower the loft to reduce terminal
  vertical-miss close calls)
- `effective_radius` — fuze lethal radius (300 m)
- `arm_range` — CPA-fuze arming range (1500 m)
- `notch_enable`, `notch_vmin`, `notch_count`, `notch_range`, `notch_min_range`
- `Sim_time_step`, `Control_time_step` — terminal time-step resolution

**`jsb_gym/envs/config/baseEnv_conf.py`** (environment):
- `observation_shape = (40, 19)`
- `pbrs_coef`, `pbrs_gamma`, `pbrs_w_prox`, `pbrs_w_nose`, `shot_in_wez_bonus`
- `red_doctrine_randomize`, `red_launch_scale_range`

**`RUN_train.py`** (training run): `N_ENVS`, `STEPS`, `BC_START/BC_END`,
`VALUE_WARMUP`, `OUT`.

---

## 13. Diagnostics & telemetry tools (in `extra_scripts/`)

| Script | What it does |
|---|---|
| `eval_notch.py` | Win rate + evasion breakdown (called by `RUN_eval.py`) |
| `missile_telemetry.py` | Per-sub-step telemetry of every missile in one engagement → CSVs, plots, analysis.txt |
| `missile_track.py` / `blue_missile_track.py` | Track a missile's position/bearing relative to an aircraft (terminal-behaviour diagnosis) |
| `wez_sweep_fine.py` | Re-measure the WEZ (Monte-Carlo; heavy) |
| `calibrate_missile_flyout.py` | Re-measure missile reach-vs-Mach |
| `gen_missile_envelope.py` | Replay with each missile's shrinking reach bubble |

Worked example produced by these tools: engagement **seed 7578**, Blue shot #3
passed 247 m horizontally but 358 m above Red (3-D miss 435 m > 300 m) due to a
diving loft + Red's terminal beam — a correct miss, not a bug. See
`missile_behaviour_report.tex` in Downloads.

---

## 14. Windows-compatibility fixes applied

This copy was made runnable on native Windows (the project was built on Linux):
- **Multiprocessing:** `train_rir.py` and `jsb_gym/envs/vec_env.py` now select
  `fork` on Linux and **`spawn`** on Windows, and cloudpickle the env factories
  so they survive spawn. (Original code hard-coded `fork`, which is Linux-only.)
- **NumPy/torch:** models were saved with NumPy 2 → the env must use **NumPy 2.x
  and torch ≥ 2.4** (see §3). Upgrading via `requirements.txt` resolves the
  `No module named 'numpy._core'` load error.
- **Unicode:** the `RUN_*.py` scripts force UTF-8 output (and set `PYTHONUTF8=1`
  for subprocesses) so the emoji/arrow log lines don't crash the cp1252 Windows
  console.
- **Paths:** all `RUN_*.py` use relative paths and pin the working directory, so
  the project runs from wherever the folder is placed. (Some older `gen_*.py` in
  `extra_scripts/` still have hard-coded Linux paths — prefer the `RUN_*.py`.)

---

## 15. Troubleshooting

- **`ModuleNotFoundError`** → `pip install -r requirements.txt`, then re-select
  the interpreter (`Ctrl+Shift+P → Python: Select Interpreter`).
- **`No module named 'numpy._core...'`** → NumPy too old; install NumPy 2.x +
  torch ≥ 2.4 together (`pip install -r requirements.txt`).
- **`cannot find context for 'fork'`** → already fixed here (auto-`spawn`).
- **`UnicodeEncodeError`** → already fixed (UTF-8 forced); if you run a script
  from `extra_scripts/` directly, set `PYTHONUTF8=1`.
- **`/usr/bin/env` not recognized** → that was the Code Runner extension reading
  a shebang; the shebangs were removed. Use the ▶ button / Run-and-Debug.
- **`FileNotFoundError` for a model** → open the **`bvr_project`** folder itself
  as the workspace so the working directory is correct.
- **Training slow / RAM heavy** → lower `N_ENVS` and `STEPS` in `RUN_train.py`.
- **Tacview won't open** → install Tacview; the `.acmi` is plain text.

---

## 16. Known issues & future work

- **Win-rate vs the old champion.** The sensor-aware agent (28%) trades raw win
  rate for realistic, varied behaviour; closing the gap to the 54% no-radar
  champion is ongoing.
- **Self-play stalemate.** Blue-vs-Blue self-play stalled — both copies notch so
  well that nobody dies and games time out. Deferred until kills are common.
- **Terminal vertical misses.** The loft/dive can leave a vertical offset at the
  merge (the seed-7578 close call). Lowering the loft (`dive_at`/`alt_cruise`)
  may convert some close calls to hits — worth a sweep.
- **WEZ tail-chase lobe** looked very small in the coarse sweep; verify against
  the fine Monte-Carlo grid.

---

## 17. Glossary

- **BVR** — Beyond Visual Range; missile combat at tens of km.
- **PN (Proportional Navigation)** — guidance that steers to null line-of-sight
  rotation (lead pursuit).
- **Beam / beaming** — flying ~90° to a threat missile's line of sight.
- **Notch (Doppler/main-lobe-clutter notch)** — the near-zero-closing-speed band
  where a beaming target hides in ground clutter and the seeker loses lock.
- **Drag** — turning ~180° and running to bleed a missile's energy.
- **R_max** — max launch range vs a non-maneuvering target.
- **R_ne** — no-escape range (a maneuvering target still can't escape inside it).
- **WEZ** — Weapon-Employment-Zone (the launch envelope).
- **CPA** — closest point of approach.
- **RIR** — the BC-anchored RL pipeline used here (BC warm start + decaying BC
  anchor during PPO).
- **PFSP** — Prioritised Fictitious Self-Play (the self-play league scheme).
- **ACMI** — Tacview's replay file format.

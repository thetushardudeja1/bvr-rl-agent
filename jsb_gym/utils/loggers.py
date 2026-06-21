"""
loggers.py
==========
ACMI format TacView logger for BVR Gym.
Fix: logs missile at Red's exact position on hit frame inside save_logs()
"""

import os
import shutil
from datetime import datetime
from pathlib import Path


class TacviewLogger:

    ID_BLUE      = "1000"
    ID_RED       = "2000"
    ID_BLUE_BASE = 0x3000
    ID_RED_BASE  = 0x4000

    def __init__(self, env):
        self.env                = env
        self.tacview_output_dir = env.conf.tacview_output_dir
        self.lines              = []

        self.blue_missile_launched = {}
        self.red_missile_launched  = {}
        self.blue_missile_hit      = {}
        self.red_missile_hit       = {}
        # track missiles that missed / bled energy, so they are REMOVED from
        # the replay (otherwise a missed missile freezes in place forever).
        self.blue_missile_lost     = {}
        self.red_missile_lost      = {}

        # track whether we've already written the aircraft-destruction (-ID)
        # so the jet blows up on the EXACT frame it dies, not at episode end.
        self.blue_destroyed = False
        self.red_destroyed  = False

        for k in env.blue_agent.ammo.keys():
            self.blue_missile_launched[k] = False
            self.blue_missile_hit[k]      = False
            self.blue_missile_lost[k]     = False
        for k in env.red_agent.ammo.keys():
            self.red_missile_launched[k] = False
            self.red_missile_hit[k]      = False
            self.red_missile_lost[k]     = False

        now = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        self.lines.append("FileType=text/acmi/tacview")
        self.lines.append("FileVersion=2.2")
        self.lines.append(f"0,ReferenceTime={now}")
        self.lines.append(f"0,Title=BVR Gym — RL vs Behavior Tree")
        self.lines.append(f"0,Author=BVR Gym (Scukins et al. 2024)")
        self.lines.append(f"0,Category=BVR Air Combat")
        self.lines.append(f"0,Briefing=F-16 RL agent (Blue) vs F-16 Behavior Tree (Red)")
        self.lines.append("")
        self.lines.append(f"{self.ID_BLUE},Name=F-16 (RL Agent),Color=Blue,"
                          f"Type=Air+FixedWing,Coalition=Blue,Pilot=RL Agent")
        self.lines.append(f"{self.ID_RED},Name=F-16 (Behavior Tree),Color=Red,"
                          f"Type=Air+FixedWing,Coalition=Red,Pilot=Behavior Tree")
        self.lines.append("")

    def _fmt_transform(self, lon, lat, alt, roll=0, pitch=0, yaw=0):
        return f"T={lon:.6f}|{lat:.6f}|{alt:.2f}|{roll:.2f}|{pitch:.2f}|{yaw:.2f}"

    def _log_aircraft(self, agent, obj_id):
        lon   = agent.simObj.get_long_gc_deg()
        lat   = agent.simObj.get_lat_gc_deg()
        alt   = agent.simObj.get_altitude()
        roll  = agent.simObj.get_phi()
        pitch = agent.simObj.get_theta()
        yaw   = agent.simObj.get_psi()
        self.lines.append(f"{obj_id},{self._fmt_transform(lon, lat, alt, roll, pitch, yaw)}")

    def _log_missile(self, missile, missile_id):
        lon   = missile.get_long_gc_deg()
        lat   = missile.get_lat_gc_deg()
        alt   = missile.get_altitude()
        roll  = missile.get_phi()
        pitch = missile.get_theta()
        yaw   = missile.get_psi()
        self.lines.append(f"{missile_id},{self._fmt_transform(lon, lat, alt, roll, pitch, yaw)}")

    def _log_missile_at_target(self, target_agent, missile_id):
        """Log missile at target's exact position — complete trail to impact."""
        lon = target_agent.simObj.get_long_gc_deg()
        lat = target_agent.simObj.get_lat_gc_deg()
        alt = target_agent.simObj.get_altitude()
        self.lines.append(f"{missile_id},{self._fmt_transform(lon, lat, alt)}")

    def _process_missile_hits(self, t):
        """Check and log any missile hits — called from both log_flight_data and save_logs."""

        # Blue missiles
        for k, missile in self.env.blue_agent.ammo.items():
            missile_id = f"{self.ID_BLUE_BASE + int(k):X}"

            if missile.is_active() or getattr(missile, 'coasting', False):
                if not self.blue_missile_launched[k]:
                    self.blue_missile_launched[k] = True
                    self.lines.append(
                        f"{missile_id},Name=AIM-120 AMRAAM (Blue AIM{k}),Color=Blue,"
                        f"Type=Weapon+Missile,Coalition=Blue,Parent={self.ID_BLUE}"
                    )
                self._log_missile(missile, missile_id)

            elif missile.is_target_hit() and not self.blue_missile_hit[k]:
                self.blue_missile_hit[k] = True
                # Log missile AT Red's position — complete trail to impact
                self._log_missile_at_target(self.env.red_agent, missile_id)
                self.lines.append(f"-{missile_id}")
                self.lines.append(
                    f"0,Event=Destroyed|{missile_id}|{self.ID_RED}|"
                    f"AIM-120 AMRAAM hit! RL Agent kills Behavior Tree!"
                )
                print(f"  🎯 Missile hit logged at Red's position!")
                
            elif getattr(missile, 'is_traget_lost', missile.is_target_lost)() and not self.blue_missile_lost[k]:
                self.blue_missile_lost[k] = True
                self.lines.append(f"-{missile_id}")

        # Red missiles
        for k, missile in self.env.red_agent.ammo.items():
            missile_id = f"{self.ID_RED_BASE + int(k):X}"

            if missile.is_active() or getattr(missile, 'coasting', False):
                if not self.red_missile_launched[k]:
                    self.red_missile_launched[k] = True
                    self.lines.append(
                        f"{missile_id},Name=AIM-120 AMRAAM (Red AIM{k}),Color=Red,"
                        f"Type=Weapon+Missile,Coalition=Red,Parent={self.ID_RED}"
                    )
                self._log_missile(missile, missile_id)

            elif missile.is_target_hit() and not self.red_missile_hit[k]:
                self.red_missile_hit[k] = True
                self._log_missile_at_target(self.env.blue_agent, missile_id)
                self.lines.append(f"-{missile_id}")
                self.lines.append(
                    f"0,Event=Destroyed|{missile_id}|{self.ID_BLUE}|"
                    f"AIM-120 AMRAAM hit! Behavior Tree kills RL Agent!"
                )
                
            elif getattr(missile, 'is_traget_lost', missile.is_target_lost)() and not self.red_missile_lost[k]:
                self.red_missile_lost[k] = True
                self.lines.append(f"-{missile_id}")

    def log_flight_data(self):
        t = self.env.blue_agent.simObj.get_sim_time_sec()
        self.lines.append(f"#{t:.2f}")

        # Blue — destroy on the exact frame it dies (write -ID once)
        if self.env.blue_agent.healthPoints > 0:
            self._log_aircraft(self.env.blue_agent, self.ID_BLUE)
        elif not self.blue_destroyed:
            self.blue_destroyed = True
            self.lines.append(f"-{self.ID_BLUE}")
            self.lines.append(
                f"0,Event=Destroyed|{self.ID_BLUE}||Blue F-16 (RL Agent) shot down!"
            )

        # Red — destroy on the exact frame it dies (write -ID once)
        if self.env.red_agent.healthPoints > 0:
            self._log_aircraft(self.env.red_agent, self.ID_RED)
        elif not self.red_destroyed:
            self.red_destroyed = True
            self.lines.append(f"-{self.ID_RED}")
            self.lines.append(
                f"0,Event=Destroyed|{self.ID_RED}||Red F-16 (Behavior Tree) shot down!"
            )

        self._process_missile_hits(t)

    def save_logs(self):
        """Called at episode end — log final frame including any hit events."""
        t = self.env.blue_agent.simObj.get_sim_time_sec()
        self.lines.append(f"#{t:.2f}")

        # Final aircraft positions (only if still alive)
        if self.env.blue_agent.healthPoints > 0:
            self._log_aircraft(self.env.blue_agent, self.ID_BLUE)
        elif not self.blue_destroyed:
            self.blue_destroyed = True
            self.lines.append(f"-{self.ID_BLUE}")
            self.lines.append(
                f"0,Event=Destroyed|{self.ID_BLUE}||Blue F-16 (RL Agent) shot down!"
            )
            print(f"  💥 Blue F-16 (RL) destroyed — logged in ACMI")

        if self.env.red_agent.healthPoints > 0:
            self._log_aircraft(self.env.red_agent, self.ID_RED)
        elif not self.red_destroyed:
            self.red_destroyed = True
            self.lines.append(f"-{self.ID_RED}")
            self.lines.append(
                f"0,Event=Destroyed|{self.ID_RED}||Red F-16 (Behavior Tree) shot down!"
            )
            print(f"  💥 Red F-16 (BT) destroyed — logged in ACMI")

        # process hits in final frame too (catches a hit on the done step)
        self._process_missile_hits(t)

        self.mkdir_logs()
        filepath = os.path.join(self.tacview_output_dir, "BVRGym_engagement.acmi")

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(self.lines))
            f.write('\n')

        print(f"  ✅ ACMI saved: {filepath}")

    def mkdir_logs(self):
        folder = Path(self.tacview_output_dir)
        shutil.rmtree(folder, ignore_errors=True)
        folder.mkdir(parents=True, exist_ok=True)

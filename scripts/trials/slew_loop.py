#!/usr/bin/env python3
"""
The 1.1 Hz loaded-slew loop, measured off an mcap (issue 304).

One oracle for both arms: the machine bag and a Gazebo run recorded with the
same topics. `/cranedata` is the machine's only; without it the sway section
reports what `/joint_states` alone can say.

    ./slew_loop.py <bag.mcap> --t0 75.6 --t1 81.95        # the 11-09 move
    ./slew_loop.py /tmp/repro/sim_0.mcap                  # whole capture

Reads with `mcap_ros2` off the embedded schemas: no sourced workspace, no built
`epsilon_crane_msgs`. Joints by `name[]`, never by index.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
from mcap_ros2.reader import read_ros2_messages

JS = "/joint_states"
CD = "/cranedata"
CS = "/trajectory_controllers/controller_state"

ACTUATED = {
    "slew": "theta1_slewing_joint",
    "boom": "theta2_boom_joint",
    "arm": "theta3_arm_joint",
    "tele": "q4_big_telescope",
    "rot": "theta8_rotator_joint",
}
TIP, TILT = "theta6_tip_joint", "theta7_tilt_joint"
# C3's own command lag per axis, `crane_model` ACTUATOR_FIT: dead time 0.06 s
# common + tau_v. What the MPC's model thinks the lag below is.
C3_LAG = {"slew": 0.16, "boom": 0.085, "arm": 0.06, "tele": 0.11, "rot": 0.185}


def read(path, t0, t1):
    """
    Streams on the publishers' own clock: the header stamp, never `log_time`.

    A Gazebo bag's wall clock runs at whatever real-time factor the machine
    gave it, so `log_time` would read the frequencies off by that factor.
    """
    streams = {JS: [], CD: [], CS: []}
    first = None
    for msg in read_ros2_messages(path, topics=list(streams)):
        stamp = msg.ros_msg.header.stamp
        t = stamp.sec + stamp.nanosec / 1e9
        first = t if first is None else first
        rel = t - first
        if rel < t0:
            continue
        if rel > t1:
            break
        streams[msg.channel.topic].append((rel, msg.ros_msg))
    return streams


def grid(streams):
    """Everything interpolated onto the `/joint_states` stamps."""
    rows = streams[JS]
    if len(rows) < 200:
        sys.exit(f"{JS}: {len(rows)} messages in the window, need >= 200")
    t = np.array([r[0] for r in rows])
    names = list(rows[0][1].name)
    sig = {}
    for label, joint in {**ACTUATED, "tip": TIP, "tilt": TILT}.items():
        i = names.index(joint)
        sig[f"q_{label}"] = np.array([r[1].position[i] for r in rows])
        sig[f"dq_{label}"] = np.array([r[1].velocity[i] for r in rows])
    if streams[CD]:
        tc = np.array([r[0] for r in streams[CD]])

        def imu(unit, field):
            raw = np.array(
                [getattr(getattr(r[1].imu, unit), field) for r in streams[CD]]
            )
            return np.interp(t, tc, raw)

        for unit in ("id15", "id16"):
            for ax in "xyz":
                sig[f"{unit}.w{ax}"] = imu(unit, f"omega_{ax}")
        for ax in "xyz":
            sig[f"dw_{ax}"] = sig[f"id16.w{ax}"] - sig[f"id15.w{ax}"]
        sig["id16.a_y"] = imu("id16", "a_y")
        sig["id16.a_z"] = imu("id16", "a_z")
    cmd = {}
    if streams[CS]:
        tcs = np.array([r[0] for r in streams[CS]])
        cn = list(streams[CS][0][1].joint_names)
        for label, joint in ACTUATED.items():
            i = cn.index(joint)
            raw = np.array([r[1].output.velocities[i] for r in streams[CS]])
            cmd[label] = np.interp(t, tcs, raw)
    return t, sig, cmd


class Band:
    """Single-frequency amplitude and phase over one Hann-windowed window."""

    def __init__(self, t, f0, bw):
        self.n = len(t)
        self.fs = 1.0 / float(np.median(np.diff(t)))
        self.w = np.hanning(self.n)
        self.f = np.fft.rfftfreq(self.n, 1.0 / self.fs)
        self.mask = (self.f >= f0 - bw) & (self.f <= f0 + bw)

    def spectrum(self, x):
        return np.fft.rfft((x - x.mean()) * self.w) * 2.0 / self.w.sum()

    def peak(self, x):
        """Amplitude and frequency of the strongest line in the band."""
        mag = np.abs(self.spectrum(x))[self.mask]
        return mag.max(), self.f[self.mask][mag.argmax()]

    def amp(self, x, f0, bw):
        mag = np.abs(self.spectrum(x))
        sel = (self.f >= f0 - bw) & (self.f <= f0 + bw)
        return mag[sel].max()

    def lag(self, u, y, signed=False):
        """
        How far `y` lags `u` at the band's strongest `u` line.

        A single line fixes the delay only mod 1/f: returned in [0, 1/f), or
        wrapped to (-1/2f, 1/2f] with `signed`. Phase comes back too, since
        that is the unambiguous number.
        """
        spec_u, spec_y = self.spectrum(u), self.spectrum(y)
        k = self.mask.nonzero()[0][np.abs(spec_u[self.mask]).argmax()]
        period = 1.0 / self.f[k]
        phase = -np.angle(spec_y[k] / spec_u[k])
        delay = (phase / (2 * np.pi)) % 1.0
        delay = delay - 1.0 if signed and delay > 0.5 else delay
        amp_u = abs(spec_u[k])
        return (
            delay * period,
            np.degrees(phase) % 360.0,
            abs(spec_y[k]) / max(amp_u, 1e-12),
            amp_u,
        )

    def pass_band(self, x):
        spec = np.fft.rfft(x - x.mean())
        spec[~self.mask] = 0.0
        return np.fft.irfft(spec, self.n)

    def growth(self, x, f0):
        """Exponential rate of the band envelope, fitted on per-cycle peaks."""
        envelope = np.abs(self.pass_band(x))
        step = max(int(self.fs / f0), 2)
        blocks = self.n // step
        if blocks < 4:
            return float("nan"), np.array([])
        peaks = np.array(
            [envelope[i * step : (i + 1) * step].max() for i in range(blocks)]
        )
        times = np.arange(blocks) * step / self.fs
        rate = np.polyfit(times, np.log(np.maximum(peaks, 1e-9)), 1)[0]
        return rate, peaks


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("bag")
    ap.add_argument("--t0", type=float, default=0.0, help="s after the first message")
    ap.add_argument("--t1", type=float, default=1e9)
    ap.add_argument("--freq", type=float, default=1.1, help="Hz, the loop line")
    ap.add_argument("--bw", type=float, default=0.3, help="Hz, half the search band")
    args = ap.parse_args()

    streams = read(args.bag, args.t0, args.t1)
    t, sig, cmd = grid(streams)
    band = Band(t, args.freq, args.bw)
    f0 = args.freq
    print(
        f"window {t[0]:.2f}-{t[-1]:.2f} s  N {band.n}  fs {band.fs:.1f} Hz  "
        f"line {f0} +-{args.bw} Hz  pendulum {f0 / 2:.2f} Hz"
    )
    if not streams[CD]:
        print(f"{CD}: not recorded -- sway section is /joint_states only")

    print(f"\n-- amplitudes  (A@{f0} = strongest line in the band)")
    print(f"{'signal':<12}{'rms':>9}{'A@line':>9}{'f_pk':>8}{'A@pend':>9}")
    for key, series in sig.items():
        a_line, f_pk = band.peak(series)
        print(
            f"{key:<12}{series.std():>9.4f}{a_line:>9.4f}{f_pk:>8.3f}"
            f"{band.amp(series, f0 / 2, args.bw / 2):>9.4f}"
        )

    # The angle channel is the rate's integral iff A_q * 2 pi f == A_dq. An
    # accelerometer-fed angle would carry the line without that tie.
    print("\n-- is the line real tip motion?")
    for label in ("tip", "tilt"):
        a_q, f_pk = band.peak(sig[f"q_{label}"])
        a_dq, _ = band.peak(sig[f"dq_{label}"])
        print(
            f"{label}: A_q {a_q:.4f} rad, A_dq {a_dq:.4f} rad/s, "
            f"A_dq/(2 pi f A_q) {a_dq / max(2 * np.pi * f_pk * a_q, 1e-12):.2f} (1.0 = integral)"
        )
    if streams[CD]:
        for ax, label in (("x", "tip"), ("y", "tilt")):
            a_d, _ = band.peak(sig[f"dw_{ax}"])
            a_15, _ = band.peak(sig[f"id15.w{ax}"])
            a_16, _ = band.peak(sig[f"id16.w{ax}"])
            print(
                f"dw_{ax} ({label}): A {a_d:.4f} vs id15 {a_15:.4f} + id16 {a_16:.4f} "
                f"-- differential (antiphase) if A > max of the two"
            )
            print(
                f"   corr(dw_{ax}, dq_{label}) {np.corrcoef(sig[f'dw_{ax}'], sig[f'dq_{label}'])[0, 1]:+.3f}"
            )
        print(f"   gyro quantiser 2^-9 = {2**-9:.5f} rad/s")

    print("\n-- growth of the band envelope (>0 grows)")
    verdict = []
    for key in ("dq_tip", "dq_boom", "dq_arm", "dq_tele"):
        rate, peaks = band.growth(sig[key], f0)
        _, f_pk = band.peak(sig[key])
        verdict.append(rate)
        print(
            f"{key:<8} {rate:+.2f} 1/s  x{np.exp(rate / f0):.2f} per cycle  "
            f"f_pk {f_pk:.3f}  peaks {np.round(peaks, 4)}"
        )

    if cmd:
        print(
            "\n-- command -> joint velocity at the line (lag in [0, 1/f); A_dq small = no response)"
        )
        print(
            f"{'axis':<7}{'A_u':>8}{'A_dq':>8}{'|dq/u|':>8}{'deg':>7}{'lag':>7}{'C3 lag':>8}"
        )
        for label in ACTUATED:
            delay, deg, gain, amp_u = band.lag(cmd[label], sig[f"dq_{label}"])
            a_dq, _ = band.peak(sig[f"dq_{label}"])
            print(
                f"{label:<7}{amp_u:>8.4f}{a_dq:>8.4f}{gain:>8.2f}{deg:>7.0f}"
                f"{delay:>7.3f}{C3_LAG[label]:>8.3f}"
            )
        print("\n-- sway feedback -> command at the line (signed: <0 = command leads)")
        for label in ("boom", "arm"):
            delay, deg, _, _ = band.lag(sig["q_tip"], cmd[label], signed=True)
            corr = np.corrcoef(sig["q_tip"], cmd[label])[0, 1]
            print(
                f"q_tip -> {label} u: lag {delay:+.3f} s ({deg:.0f} deg), corr {corr:+.3f}"
            )

    grew = max(verdict) > 0.05
    print(f"\nverdict: {'GROWING' if grew else 'calm'} at {f0} Hz")
    return 0


if __name__ == "__main__":
    sys.exit(main())

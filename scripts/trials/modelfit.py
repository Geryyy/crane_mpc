#!/usr/bin/env python3
"""
Model fit: how well the MPC's model predicts what the crane actually did.

One cycle ahead, exactly as `Cycle` does it: measured positions and velocities
overwrite the state, the unmeasured force and lag rows carry from the model,
integrate one `Ts` under the command that was in flight, compare the predicted
velocity against the one that arrives.

Scored against the trivial predictor -- "nothing changes" -- because over 60 ms
that is already a good guess and a model that cannot beat it is saying nothing.
`skill = 1 - rms(model)/rms(actual change)`: 1 is perfect, 0 is no better than
assuming the crane keeps doing what it is doing, negative is worse. Only cycles
where the axis actually changed speed are scored; on a still machine both errors
go to zero and the ratio means nothing.

Two sources, one number:

    modelfit.py bag '<glob of HydraulicCalib bags>' [count]   two tables
    modelfit.py capture <cap.json from trial.sh>

Positions, velocities and both passive rows are re-seeded from the measurement
every cycle, so they cannot compound. Only C3's force and command-lag rows carry
forward, because nothing measures them -- exactly what `Cycle` does.

**That carry is not a source of drift; it is the only thing holding the force
state up.** Measured on `good1`: carrying it scores sw 0.62 / ha 0.76 / ka 0.36,
re-seeding it from `static_hold_force` every cycle scores 0.03 / 0.20 / 0.14,
and every fifth cycle 0.42 / 0.25 / 0.31. During motion the real C3 force is
nowhere near the holding force, so throwing the estimate away each step is much
worse than keeping it. Skill also *rises* through a run (sw 0.54 -> 0.64 over
the second half) rather than decaying: what the carry costs is a startup
transient while the force row converges, not accumulation.

The way to remove that last error is to **measure** the force, not to re-seed
it, and the `bag` arm now scores both: `/cranedata`'s chamber pressures through
`crane_model.pressure.actuated_force_from_pressure`. On the axis a campaign bag
actually excites the measurement wins by a wide margin (169) -- one-cycle rms,
five bags each:

    axis  campaign    carried  measured
    sw    M01v10-14    0.1046    0.0293
    ha    M02v11-15    0.0985    0.0124
    ka    M03v*        0.6824    0.0818
    sa    M04v20-24    0.0237    0.0295   <- the one loss
    ro    M05v*        1.5688    0.0861

Both columns score the same cycles -- `report`'s `n` is identical arm to arm on
every campaign above, so no bag and no cycle was dropped from one and not the
other.

On an axis standing still it loses, because the measured force also carries
reactions the rigid-body model has no room for -- an end stop, and a payload
these timber bags do not share with the PZS100. The telescope is the one axis
that loses while excited, and **the reason is not established**: the formula is
circuit-independent when *both* pressures are measured, and `M04v20-24` shows
the transducers do see the regeneration (`p_a` 66 / `p_b` 80 bar extending
against 34 / 78 retracting). What is left is that the carry is already close to
the trivial predictor there (skill -0.88, against -14 to -38 elsewhere), so the
seed has almost nothing to win and its own bias -- wear-pad friction on the
smallest net area on the machine -- costs more.

In sim none of this is available: the gz actuator's `effort_state_` is the right
quantity but is not exposed, and the `effort` state interface is the transmitted
wrench, gravity and constraint reaction included, which is a different thing.

**Which command you score against decides the answer.** On a capture, use what
the JTC put on the interface (`controller_state.output`), not the MPC's `u`: the
JTC ramps between knots and adds its PI, and scoring `u` held flat instead reads
20x worse on the same run. In the bags `velocities_sp` *is* the command, Psi's
output, with no follower in between.
"""

import glob
import json
import sys

import numpy as np
from crane_model import symbolic as cs
from crane_model.conventions import canonical_joints
from crane_model.pressure import BAR_TO_PA, actuated_force_from_pressure
from crane_mpc import problem
from crane_mpc.solver import Ocp

PL = cs.K_PLANNED_DOF
POS = slice(cs.X_PLANNED_POSITION, cs.X_PLANNED_POSITION + PL)
VEL = slice(cs.X_PLANNED_VELOCITY, cs.X_PLANNED_VELOCITY + PL)
PAS = slice(cs.X_PASSIVE_POSITION, cs.X_PASSIVE_POSITION + cs.K_PASSIVE_DOF)
PAV = slice(cs.X_PASSIVE_VELOCITY, cs.X_PASSIVE_VELOCITY + cs.K_PASSIVE_DOF)
AX = ["sw", "ha", "ka", "sa", "ro"]
CANON = list(canonical_joints())
PLANNED_ROWS = [0, 1, 2, 3, 6]
PASSIVE_ROWS = [4, 5]
MOVING = 5.0e-3
CONSTANTS = cs.load_constants()


def shipped():
    import copy

    import yaml
    from ament_index_python.packages import get_package_share_directory
    from crane_model import hydraulic_limits
    from crane_mpc.config import machine_limits

    share = get_package_share_directory("crane_mpc")
    with open(f"{share}/config/crane_mpc.yaml") as stream:
        parameters = yaml.safe_load(stream)["crane_mpc"]["ros__parameters"]
    parameters.update({"hydraulics": hydraulic_limits()})
    parameters["limits"].update(machine_limits())
    return copy.deepcopy(parameters)


def score(ocp, t, q, dq, command, cycle=None, force=None):
    """
    Walk one run; return (model error, actual change) per scored cycle.

    `cycle` is one `Ts` expressed in whatever clock `t` is on -- a capture's
    controller stream is wall-stamped while `Ts` is simulated seconds, and at
    an RTF near 0.5 taking them for the same thing steps half a cycle.

    `force(k)` overwrites the force row from a measurement instead of carrying
    it. The command-lag row carries either way -- nothing measures that.
    """
    stride = max(1, int(round((cycle or ocp.Ts) / float(np.median(np.diff(t))))))
    x = np.zeros(cs.NX)
    x[POS] = q[0, :PL]
    x[PAS] = q[0, PL : PL + 2]
    x[cs.X_PROGRESS_RATE] = float(
        ocp.parameters["limits"]["progress_rate_headroom"]
    ) * problem.nominal_progress_rate(ocp.parameters)
    ocp.pin_tool(float(q[0, -1]))
    x[cs.X_ACTUATED_FORCE : cs.X_ACTUATED_FORCE + PL] = ocp.static_hold_force(x)

    model, actual = [], []
    for k in range(0, len(t) - stride, stride):
        u = command(k, t[k])
        if u is None:
            continue
        x[POS] = q[k, :PL]
        x[VEL] = dq[k, :PL]
        x[PAS] = q[k, PL : PL + 2]
        x[PAV] = dq[k, PL : PL + 2]
        if force is not None:
            x[cs.X_ACTUATED_FORCE : cs.X_ACTUATED_FORCE + PL] = force(k)
        x[cs.X_PROGRESS] = 0.0
        step = np.zeros(cs.NU_PROGRESS)
        step[:PL] = u
        try:
            predicted = ocp._integrate(ocp._stepper, x, step)
        except Exception:
            continue
        landed = dq[k + stride, :PL]
        model.append(predicted[VEL] - landed)
        actual.append(dq[k, :PL] - landed)
        x = predicted
    return np.asarray(model), np.asarray(actual)


def report(label, model, actual):
    print(f"\n{label}\n")
    print(f"{'':>4} " + " ".join(f"{a:>21}" for a in AX))
    print(f"{'':>4} " + " ".join(f"{'rms  skill     n':>21}" for _ in AX))
    cells = []
    for axis in range(PL):
        moving = np.abs(actual[:, axis]) > MOVING
        if moving.sum() < 10:
            cells.append(f"{'--':>8} {'--':>7} {0:>5}")
            continue
        rms = float(np.sqrt(np.mean(model[moving, axis] ** 2)))
        base = float(np.sqrt(np.mean(actual[moving, axis] ** 2)))
        cells.append(f"{rms:8.4f} {1 - rms / base:7.2f} {int(moving.sum()):5d}")
    print(f"{'':>4} " + " ".join(f"{c:>21}" for c in cells))
    print("\nrms is rad/s of one-cycle velocity prediction error, over cycles where")
    print(f"the axis changed speed by more than {MOVING} rad/s.")


def measured_force(q, data):
    """
    `k -> tau_a` from `/cranedata`'s chamber pressures; None if the bag has none.

    The boom and arm columns are looked up, not written as 1 and 2: `q` is
    `PLANNED_ROWS + PASSIVE_ROWS + [7]`, and reordering `PLANNED_ROWS` would
    otherwise evaluate the transmission at the wrong joints in silence.
    """
    boom, arm = PLANNED_ROWS.index(1), PLANNED_ROWS.index(2)
    p_a = np.asarray(data["pressures_a"], float) * BAR_TO_PA
    p_b = np.asarray(data["pressures_b"], float) * BAR_TO_PA
    if not (np.isfinite(p_a).all() and np.isfinite(p_b).all()):
        return None
    return lambda k: actuated_force_from_pressure(
        CONSTANTS, q[k, boom], q[k, arm], p_a[k], p_b[k]
    )


def from_bags(ocp, pattern, count, pressure=False):
    sys.path.insert(0, "/workspaces/ros2_baustelle_ws/timber_crane_mujoco_py")
    from timber_crane_mujoco_py.utils.rosbag_utils import read_bag_full

    model, actual = [], []
    for path in sorted(glob.glob(pattern))[:count]:
        try:
            data = read_bag_full(path)
        except Exception as error:
            print(f"{path.split('/')[-1]:>10} skipped: {str(error)[:50]}")
            continue
        t = np.asarray(data["timestamps_js"])
        q = np.asarray(data["positions_js"])[:, PLANNED_ROWS + PASSIVE_ROWS + [7]]
        dq = np.asarray(data["velocities_js"])[:, PLANNED_ROWS + PASSIVE_ROWS + [7]]
        u = np.asarray(data["velocities_sp"])
        force = measured_force(q, data) if pressure else None
        if pressure and force is None:
            print(f"{path.split('/')[-1]:>10} skipped: no chamber pressures")
            continue
        one, two = score(ocp, t, q, dq, lambda k, _: u[k, :PL], force=force)
        model.append(one)
        actual.append(two)
    return np.vstack(model), np.vstack(actual)


def from_capture(ocp, path):
    capture = json.load(open(path))
    js = capture["js"]
    names = PLANNED_ROWS + PASSIVE_ROWS + [7]
    wanted = [CANON[i] for i in names]

    def pick(rows, field):
        out = []
        for r in rows:
            index = {n: i for i, n in enumerate(r["names"])}
            out.append([r[field][index[n]] if n in index else np.nan for n in wanted])
        return np.asarray(out, float)

    t = np.array([r["t"] for r in js])
    # Both streams are wall-stamped; `Ts` is not.
    sim = np.array([r.get("sim", r["t"]) for r in js])
    rtf = float((sim[-1] - sim[0]) / (t[-1] - t[0]))
    ctrl = [r for r in capture.get("ctrl", []) if r.get("out")]
    if not ctrl:
        raise SystemExit(
            "this capture has no controller_state; re-record with capture.py"
        )
    ct = np.array([r["t"] for r in ctrl])
    cu = np.array(
        [
            [
                r["out"][r["names"].index(n)] if n in r["names"] else np.nan
                for n in [CANON[i] for i in PLANNED_ROWS]
            ]
            for r in ctrl
        ]
    )

    def command(k, when):
        due = np.nonzero(ct <= when)[0]
        return cu[due[-1]] if len(due) else None

    return score(ocp, t, pick(js, "pos"), pick(js, "vel"), command, cycle=ocp.Ts / rtf)


def main():
    parameters = shipped()
    ocp = Ocp(
        problem.default_description().read_text(), parameters, parameters["hydraulics"]
    )
    if sys.argv[1] == "bag":
        count = int(sys.argv[3]) if len(sys.argv) > 3 else 3
        for pressure in (False, True):
            arm = (
                "force row from measured pressure" if pressure else "force row carried"
            )
            model, actual = from_bags(ocp, sys.argv[2], count, pressure)
            report(f"{sys.argv[2]} ({count} bags) -- {arm}", model, actual)
    else:
        model, actual = from_capture(ocp, sys.argv[2])
        report(sys.argv[2], model, actual)


main()

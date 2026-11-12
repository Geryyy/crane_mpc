"""
The bench plant the loaded machine measured as: two regimes, one switch (issue 316).

313 check 2 and 314 measured the same boom in two states, and they only look
contradictory:

  spool open   `u -> dq` is a near-unity velocity source -- `|G|` 0.89-0.97,
               pure delay 167 ms (boom) / 154 ms (arm), **no mode in
               0.4-2.4 Hz** (313 check 2 1, `dq_ref` as instrument, 16 loaded
               windows over both 11-09 PID bags). Flow, not force: nothing
               integrates through the joint inertia.
  spool shut   the boom rings at 0.8-1.6 Hz -- a stiff, lightly damped spring
               through trapped oil, `k` x1.43 of the identified C3 (314 step 3,
               65 sway-free rings in 38 bags, IQR 1.23-1.60, a floor where the
               axes ring together) with `d` x0.44 (314 step 4c, zeta 0.107
               against the fit's 0.364).

So: a flow source while the spool is open, a spring through the oil while it is
shut, switched on `|u_f|`. Boom and arm only -- the other three axes keep the
C3 the staircases fitted them on, which is what `--plant c3` is everywhere.

The lowering branch carries 314 step 5's one load law: `Q ~ sqrt(p_A)` on the
boom, x1.30 loaded against empty at matched valve current, measured on two
loads with the current ratio 1.00; lifting is load-blind (meter-in with a
working compensator). `p_A` is taken as the boom's own holding force, which is
what sets the A-chamber pressure at a given pose -- 314's instrument run
backwards, with the counter-pressure it does not publish folded into one
constant.

Not here, because 314 could not measure them: a deadband (one lowering plateau
below `|u|` 0.025 in the whole set, on the one campaign with a position loop in
the way), `k(q, m)` (load, pose and campaign are one regressor), and an
arm-specific `k` (not identifiable -- every loaded arm ring is coupled to the
boom). `k` is a *range*: run the gate at `--hyd-k-scale` 1.2 and 1.8.
"""

import numpy as np
from crane_model.actuator import C3Actuator
from crane_model.mujoco_plant import PASSIVE_INDICES, PLANNED_INDICES
from crane_model.symbolic import PLANNED_DOF

PLANNED, PASSIVE = list(PLANNED_INDICES), list(PASSIVE_INDICES)
#: boom and arm in planned order sw, ha, ka, sa, ro -- the two axes 313/314 measured
HYD_AXES = (1, 2)
DEAD_TIME_S = 0.06  # the fit's own transport delay, [[hydraulics]] 8 block 1
#: per axis: 313 check 2 1's measured total lag (167 ms boom, 154 ms arm) less
#: the 60 ms dead time the convention pins. The shipped fit's own `d/k`
#: (90 / 81 ms) lands in the same place, and `d/k` is the one number of the
#: staircase identification that survives scaling `k` and `d` together.
TAU_V_S = (0.107, 0.094)
#: velocity servo an order of magnitude above the band of interest, so the
#: shape is the dead time and the PT1 (313 check 2 4); not swept there either
SERVO_RAD_S = 60.0
#: |u_f| rad/s at which the spool is open. 314 step 5's smallest lifting
#: plateau that moves the axis above the encoder noise is |u| 0.008-0.010
#: normalized, i.e. 2-3 mrad/s of commanded joint velocity; the lowering branch
#: below |u| 0.025 is unmeasured, so this is a structural choice -- sweep it.
SPOOL_OPEN_RAD_S = 0.005
K_SHUT_SCALE = 1.43  # 314 step 3 median over 65 sway-free boom rings
#: 314 step 4c, off the ring decays; reaches the plant as joint damping
D_SHUT_SCALE = 0.44
#: Per-direction flow gain, boom then arm, at the *loaded* reference below.
#: The one chain all three measurements agree on: loaded lowering 0.95 (313
#: check 2 1, whose clean half is the loaded `cmd < 0` windows), empty lowering
#: 0.95/1.30 (314 step 5's load factor), lift that over 1.17 (314 step 5's
#: empty per-direction ratio 1.36 divided by the driver's own per-direction
#: scales 0.3224/0.2782) and load-blind. 304's measured steady-state map
#: (0.71 lift / 1.41 lower) is the same asymmetry in the same direction.
GAIN_LIFT = (0.63, 0.95)
GAIN_LOWER = (0.95, 0.95)
#: N m, the boom's holding force in the loaded reference condition -- 880 kg at
#: the 11-09 slew pose, where the 0.95 above was measured. `records/analysis_316/probe.py`.
TAU_HOLD_REF = 1.264e5
#: N m, the meter-out counter-force in the `p_A` proxy. 314 step 5 publishes
#: `p_A` (91 bar empty, 166 loaded) and not `p_B`, so this is calibrated on the
#: one measured ratio: it is what makes this bench's own empty -> 880 kg
#: holding-force pair (6.11e4 -> 1.264e5 N m) give the measured x1.30. Not a
#: pressure model -- there is no second measurement to check it against.
COUNTER_FORCE = 3.3e4
#: the sqrt(p_A) factor is held inside the span 314 measured (empty .. 880 kg)
LOWER_FACTOR_CLIP = (0.70, 1.20)


def damping_scale(scale: float = D_SHUT_SCALE) -> np.ndarray:
    """
    Per-planned-axis joint-damping multiplier `--plant hyd` adds to the plant.

    `d` reaches a MuJoCo plant only as joint damping, so the shut-spool `d`
    has to be set there and not in the force law -- which also means it is the
    damping in *both* regimes. With the spool open that is a 4 % error on the
    servo (`d` against `M_ii SERVO_RAD_S`) and the holding force compensates
    the rest of it; with the spool shut it is the whole decay.
    """
    factors = np.ones(PLANNED_DOF)
    factors[list(HYD_AXES)] = scale
    return factors


class HydraulicActuator:
    """C3 on the three axes it was fitted on, 316's two-regime law on boom and arm."""

    def __init__(
        self,
        plant,
        timestep: float,
        fit=None,
        k_shut_scale: float = K_SHUT_SCALE,
        spool_open_rad_s: float = SPOOL_OPEN_RAD_S,
        dead_time_s: float | None = None,
        tau_max=None,
    ) -> None:
        self.plant = plant
        # The other three axes, their own clamp, and the fit `--k-scale` perturbs.
        # The two hydraulic rows are left *unclamped*: a flow source has no force
        # clamp, and the rated boom effort binds at 12 mrad/s of servo error,
        # which would make this a bang-bang force source instead.
        self.c3 = C3Actuator(
            timestep=timestep, fit=fit, dead_time_s=dead_time_s, tau_max=tau_max
        )
        self.timestep = float(timestep)
        self.rows = list(HYD_AXES)
        self.k_shut = float(k_shut_scale) * np.asarray(self.c3.fit.k, float)[self.rows]
        self.spool_open_rad_s = float(spool_open_rad_s)
        dead_time = DEAD_TIME_S if dead_time_s is None else float(dead_time_s)
        self._n_delay = int(round(dead_time / self.timestep))
        self._alpha = 1.0 - np.exp(-self.timestep / np.asarray(TAU_V_S, float))
        self._gain_lift = np.asarray(GAIN_LIFT, float)
        self._gain_lower = np.asarray(GAIN_LOWER, float)
        self._projection = plant.projection  # the mimic structure does not move
        self.reset()

    # --- state ---------------------------------------------------------------

    def reset(self, tau=None, u=None) -> None:
        """Clear both regimes' states; `tau` seeds the hold, as C3's does."""
        self.c3.reset(tau, u)
        self.u_f = (
            np.zeros(len(self.rows))
            if u is None
            else np.asarray(u, float).reshape(PLANNED_DOF)[self.rows].copy()
        )
        self._queue = [self.u_f.copy() for _ in range(self._n_delay)]
        self.tau = self.c3.tau.copy()
        self._trapped = np.zeros(len(self.rows))

    # --- the plant rows the law reads ----------------------------------------

    def _rows(self):
        """
        Return `(servo hold, gravity hold, reduced inertia)` at the two rows.

        `MujocoPlant.h_eff`'s own solve, repeated here because the servo gain
        needs the mass matrix that solve already builds -- calling the property
        would factor `mj_fullM` twice per plant step -- and because the two
        regimes need *different* holds. `h_eff` compensates the joint damper
        along with gravity, which is what a flow source does; handing the same
        force to the spring would cancel the damper and the shut spool would
        ring without decaying at all (it does, measurably: 1.0-2.1 per cycle
        instead of 0.6). So the shut branch gets `qfrc_bias` alone and leaves
        the description's `d` to dissipate.
        """
        plant = self.plant
        inertia = np.zeros((plant.model.nv, plant.model.nv))
        plant._mj.mj_fullM(plant.model, plant.data, inertia)
        mass = self._projection.T @ inertia @ self._projection
        passive = np.ix_(PASSIVE, PASSIVE)
        coupling = mass[np.ix_(PLANNED, PASSIVE)]
        holds = []
        for forces in (
            plant.data.qfrc_bias - plant.data.qfrc_passive,
            plant.data.qfrc_bias,
        ):
            bias = self._projection.T @ forces
            sway = np.linalg.solve(mass[passive], -bias[PASSIVE])
            holds.append((bias[PLANNED] + coupling @ sway)[self.rows])
        return (*holds, np.diag(mass)[PLANNED][self.rows])

    def lower_factor(self, hold_boom: float) -> float:
        """Scale the boom's lowering by `sqrt(p_A)`; 1 at the loaded reference."""
        ratio = (max(float(hold_boom), 0.0) + COUNTER_FORCE) / (
            TAU_HOLD_REF + COUNTER_FORCE
        )
        return float(np.clip(np.sqrt(ratio), *LOWER_FACTOR_CLIP))

    def flow_gain(self, hold) -> np.ndarray:
        """Per-direction gain on the two rows, the boom's lowering load-dependent."""
        lifting = self.u_f >= 0.0
        gain = np.where(lifting, self._gain_lift, self._gain_lower)
        if not lifting[0]:
            gain[0] *= self.lower_factor(hold[0])
        return gain

    # --- driving it ----------------------------------------------------------

    def step(self, u, dq) -> np.ndarray:
        """Advance one `timestep`: C3 on three axes, the two regimes on boom and arm."""
        command = np.asarray(u, float).reshape(PLANNED_DOF)
        rate = np.asarray(dq, float).reshape(PLANNED_DOF)
        tau = self.c3.step(
            command, rate
        )  # a copy; the rows below overwrite two of them
        delayed = command[self.rows]
        if self._n_delay:
            self._queue.append(delayed)
            delayed = self._queue.pop(0)
        self.u_f = self.u_f + self._alpha * (delayed - self.u_f)
        hold, gravity, mass = self._rows()
        # Spool open: flow. The servo holds the pose up and chases the commanded
        # velocity, so `u -> dq` is the dead time and the PT1 and nothing else.
        source = hold + mass * SERVO_RAD_S * (
            self.flow_gain(hold) * self.u_f - rate[self.rows]
        )
        # Spool shut: the load stands on a trapped column of oil. The deviation
        # is C3's own block 3 at the shut `k`, zeroed when the spool closes --
        # the valve stops the flow, it does not hand the oil the stiff servo's
        # braking force, and carrying that force in would start the ring with a
        # 5e5 N m offset.
        open_spool = np.abs(self.u_f) >= self.spool_open_rad_s
        self._trapped = np.where(
            open_spool,
            0.0,
            self._trapped + self.k_shut * (self.u_f - rate[self.rows]) * self.timestep,
        )
        tau[self.rows] = np.where(open_spool, source, gravity + self._trapped)
        self.tau = tau
        return tau.copy()

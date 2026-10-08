"""
C3's force state, estimated instead of integrated open loop.

Nothing in this stack measures `tau_a`, so every cycle used to carry it by
integrating `tau_dot = k (u_f - dq)` over the **measured** `dq`. `k Ts` runs to
6e4 per rad/s, so one cycle of `/joint_states`' velocity noise is 1e2-1e3 N m of
force error, it seeds the next solve's stage 0, and stage 0 is pinned without
slack -- which is where issue 223 found the whole stage-0 bound gap (116-3197 N,
all of it in the force rows).

What replaces it is a **steady-state Kalman filter per planned axis** on the
pair `(dq_a, tau_a)`. The prediction is not this module's: the whole state is
carried one `Ts` by the same acados integrator the horizon runs
(`Ocp.propagate`), so the model predicted through is the OCP's own. This module
owns the correction -- one gain per axis, applied to that prediction at the
measurement instant.

Four things that follow from it:

- **The force is corrected, never measured.** It is observable only through its
  effect on the acceleration, which is why the gain is derived from
  `Ocp.axis_transition` -- acados' own forward sensitivity of the step, not a
  second hand-written linearisation.
- **`x_0`'s velocity row stops being the measurement**, and that is the point:
  the noise reaches the force integrator through `dq`, so a `dq` the filter
  low-passes is what stops the force following it. The gain the old carry had is
  a special case of this one, `[[1], [0]]`.
- **The position row is *not* estimated.** `x_0`'s `q_a` stays exactly the
  measurement, because `position_box` pins the hard box on whatever `q_a` row
  `x_0` carries, and 226 measured that the QP tolerates 3e-4 rad of movement in
  that bound on the ticks where the arm already sits outside its control-safe
  limit. Half a cycle of lag there is 1e-2 rad and buys nothing: the encoder's
  own noise is 1.6e-5 rad, and its path into `dq` over one step is 3e-7 rad/s
  against the 3e-4 the process noise allows.
- **The gain is constant between resets.** The transition is pose- and
  payload-dependent and a Riccati recursion is not free, so it is solved at the
  pose that reset the filter and held until the next reset (a first cycle, a
  silence, a payload step -- `Cycle.seed_force`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from crane_model import symbolic as cs

#: The filtered pair, in this order, per axis.
ROWS = 2

#: Riccati sweeps before the gain is taken as it stands, and the absolute change
#: in the covariance that calls it converged earlier. The recursion is 2x2 and
#: converges in tens of sweeps at these numbers; the cap is a backstop.
RICCATI_SWEEPS = 2000
RICCATI_TOLERANCE = 1.0e-12


def steady_state_gain(
    transition: np.ndarray,
    observation: np.ndarray,
    process: np.ndarray,
    measurement: np.ndarray,
) -> np.ndarray:
    """
    Return the constant Kalman gain of `(A, C, Q, R)`, by the Riccati sweep.

    Returns `K` with `x+ = x- + K (z - C x-)`.

    Iterated rather than solved: there is no DARE on this image -- acados is the
    workspace's Riccati recursion and it is the one inside a QP -- and at 2x2
    the sweep costs microseconds.
    """
    prior = np.array(process, dtype=float)
    for _ in range(RICCATI_SWEEPS):
        gain = _gain(prior, observation, measurement)
        posterior = prior - gain @ observation @ prior
        swept = transition @ posterior @ transition.T + process
        converged = np.max(np.abs(swept - prior)) <= RICCATI_TOLERANCE * max(
            1.0, float(np.max(np.abs(swept)))
        )
        prior = swept
        if converged:
            break
    return _gain(prior, observation, measurement)


def _gain(prior, observation, measurement) -> np.ndarray:
    innovation = observation @ prior @ observation.T + measurement
    return prior @ observation.T @ np.linalg.inv(innovation)


@dataclass(frozen=True)
class Noise:
    """
    The filter's covariances, per planned axis, as variances of one `Ts` step.

    Process: how far the model may be over one cycle. `dq_a` carries whatever
    acceleration C3 does not describe; `tau_a` carries only what the C3 *fit*
    itself gets wrong, which is small and not the same thing as the force a
    cycle of velocity error produces -- that one the transition already carries,
    and counting it here too is what defeats the filter (issue 227's notes have
    the sweep).

    Measurement: `/joint_states`' own velocity noise, the one row observed.
    """

    dq_a: np.ndarray
    tau_a: np.ndarray
    dq_a_sensor: np.ndarray

    FIELDS = ("dq_a", "tau_a", "dq_a_sensor")

    @staticmethod
    def from_parameters(block) -> Noise:
        """Read the `force_filter` block, one array per field, planned axes only."""
        planned = cs.K_PLANNED_DOF
        return Noise(
            **{
                field: np.asarray(block[field][:planned], dtype=float)
                for field in Noise.FIELDS
            }
        )


class ForceFilter:
    """The per-axis correction, and the gain it is applied through."""

    def __init__(self, noise: Noise) -> None:
        self.noise = noise
        #: `(axes, 2, 1)` against the velocity innovation. `None` until `derive`.
        self._gain: np.ndarray | None = None

    @property
    def armed(self) -> bool:
        return self._gain is not None

    def derive(self, transition: np.ndarray) -> None:
        """Solve the per-axis Riccati at this pose. `transition` is `(axes, 2, 2)`."""
        axes = cs.K_PLANNED_DOF
        transition = np.asarray(transition, dtype=float)
        if transition.shape != (axes, ROWS, ROWS) or not np.all(
            np.isfinite(transition)
        ):
            raise ValueError(
                f"the force filter needs one finite {ROWS}x{ROWS} transition per "
                f"planned axis, and was handed {transition.shape}"
            )
        observation = np.eye(ROWS)[:1]
        self._gain = np.zeros((axes, ROWS, 1))
        for axis in range(axes):
            process = np.diag([self.noise.dq_a[axis], self.noise.tau_a[axis]])
            self._gain[axis] = steady_state_gain(
                transition[axis],
                observation,
                process,
                [[self.noise.dq_a_sensor[axis]]],
            )

    def forget(self) -> None:
        """Drop the gain: the pose it was derived at is no longer the machine's."""
        self._gain = None

    def correct(self, predicted: np.ndarray, measured_dq_a: np.ndarray) -> np.ndarray:
        """
        Return the corrected `(dq_a, tau_a)` per planned axis, `(axes, 2)`.

        `predicted` is the same pair carried to this instant by the model.
        """
        if not self.armed:
            raise RuntimeError(
                "the force filter has no gain: `derive` runs at the pose the filter "
                "was reset at, and nothing may be corrected before it"
            )
        axes = cs.K_PLANNED_DOF
        predicted = np.asarray(predicted, dtype=float).reshape(axes, ROWS)
        innovation = np.asarray(measured_dq_a, dtype=float)[:axes] - predicted[:, 0]
        return predicted + self._gain[:, :, 0] * innovation[:, None]

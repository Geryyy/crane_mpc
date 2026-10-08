"""
Where the tool hangs along a plan: `crane_model`'s equilibrium, in closed form.

One numeric solve costs ~4 ms (measured, 200 calls, empty and 800 kg), so a
60-knot horizon cannot have one per stage. It does not need them.
`passive_equilibrium` factors exactly -- to 1e-13 at every pose probed, empty
and at 800 kg on and off the rotator axis:

    tip_eq  = h_tip(q_ro) - (q_ha + q_ka)
    tilt_eq = h_tilt(q_ro)

The tool hangs vertically, so the tip angle *is* the luffing pair's travel, and
no other axis moves either row -- slew, telescope and the tool itself leave both
untouched. Only the rotator does, and only through an off-axis payload CoM, which
swings around it. `Equilibrium` therefore samples that one curve over the
rotator's turn once per payload and reads the table off it with no solve per
cycle at all.
"""

import numpy as np
from crane_model import symbolic as cs

#: Rotator samples over the full turn, per payload. The curve is flat empty and
#: with an on-axis CoM -- the table is then exact at any count -- and a single
#: harmonic at 0.5 m off axis, where 16 samples hold linear interpolation to
#: 0.007 rad, 3.5 % of the sway box. 16 solves is ~64 ms, paid on the cycle a
#: payload step already cold-starts.
TURN = 16

#: The luffing pair, 1:1 on the tip row, and the rotator, which carries the
#: payload CoM's own swing. `K_AXIS_KEYS` is the planned order.
LUFF = (cs.K_AXIS_KEYS.index("ha"), cs.K_AXIS_KEYS.index("ka"))
RO = cs.K_AXIS_KEYS.index("ro")


class Equilibrium:
    """`crane_model`'s passive equilibrium, as a table along a planned path."""

    def __init__(self, model, payload=None) -> None:
        self._model = model
        self._payload = payload
        self._angles = np.linspace(-np.pi, np.pi, TURN, endpoint=False)
        self._curve: np.ndarray | None = None

    def set_payload(self, payload) -> None:
        """Bind a new payload: a new CoM is a new curve."""
        self._payload = payload
        self._curve = None

    def at(self, q_a) -> np.ndarray:
        """Solve the hanging pose at one actuated configuration, numerically."""
        return np.asarray(
            self._model.passive_equilibrium(
                np.asarray(q_a, dtype=float), self._payload
            ),
            dtype=float,
        )

    def _sample(self, q_a: np.ndarray) -> np.ndarray:
        """`h` over `TURN` rotator angles: the hanging pose with the luff added back."""
        curve = np.empty((TURN, cs.K_PASSIVE_DOF))
        for row, angle in enumerate(self._angles):
            probe = q_a.copy()
            probe[RO] = angle
            curve[row] = self.at(probe)
            curve[row, 0] += probe[LUFF[0]] + probe[LUFF[1]]
        return curve

    def table(self, q_a, q_a_stages) -> np.ndarray:
        """
        One hanging pose per stage, read off `h`; no solve on a warm payload.

        `q_a` is the whole actuated vector the plan starts from -- it fixes
        nothing but the pose the curve is sampled at, and the curve does not
        depend on that pose. `q_a_stages` is the planned columns per knot, which
        is all the plan carries; the tool rides at its measurement in `q_a`, as
        it does in the OCP, and moves neither passive row.
        """
        q_a = np.asarray(q_a, dtype=float)
        if self._curve is None:
            self._curve = self._sample(q_a)
        stages = np.asarray(q_a_stages, dtype=float)
        hanging = np.stack(
            [
                np.interp(
                    stages[:, RO], self._angles, self._curve[:, row], period=2 * np.pi
                )
                for row in range(cs.K_PASSIVE_DOF)
            ],
            axis=1,
        )
        hanging[:, 0] -= stages[:, LUFF[0]] + stages[:, LUFF[1]]
        return hanging

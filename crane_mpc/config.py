"""
What `crane_mpc` refuses to start on.

`src/ocp_solver.cpp`'s `check_settings` (`:479-616`) and `check_payload`
(`:253-276`) in Python. They are a module of their own for the reason
`crane_planning` keeps a `config.py`: a bad configuration is a refusal at
startup, not a solve that answers NaN on the first cycle and a fault that blames
the solver.

**The reasons are the deliverable, not the predicates.** Every message names the
quantity, the value it carries and why the bound exists. "invalid weight" is
worth much less than what a negative entry does to the Gauss-Newton Hessian.

Nothing here reads a file or a parameter server: the caller has already shaped
the parameters (`node.py`'s `_parameter_dict`/`_hydraulics_dict`, or a yaml on a
driver's disk), and this refuses that shape.
"""

from __future__ import annotations

import math

import numpy as np
from crane_model import symbolic as cs

from .problem import K_PROGRESS_RATE_REFERENCE

#: Vector-valued fields and how wide each must be. In the C++ these were
#: `std::array` widths the compiler checked; out of a yaml the width is a
#: runtime question, and a short list is a bound that silently covers fewer axes
#: than the problem has rows.
WIDTHS = {
    "weights": {
        "q_a": cs.K_ACTUATED_DOF,
        "dq_a": cs.K_ACTUATED_DOF,
        "tau_a": cs.K_ACTUATED_DOF,
        "u": cs.K_ACTUATED_DOF,
        "q_u": cs.K_PASSIVE_DOF,
        "dq_u": cs.K_PASSIVE_DOF,
    },
    "limits": {
        "q_a_lower": cs.K_ACTUATED_DOF,
        "q_a_upper": cs.K_ACTUATED_DOF,
        "q_a_margin": cs.K_ACTUATED_DOF,
        "dq_a_max": cs.K_ACTUATED_DOF,
        "u_max": cs.K_ACTUATED_DOF,
        "q_u_max": cs.K_PASSIVE_DOF,
        "dq_u_max": cs.K_PASSIVE_DOF,
    },
    "slack": {
        "q_u": cs.K_PASSIVE_DOF,
        "dq_u": cs.K_PASSIVE_DOF,
        "cylinder_force": cs.K_ACTUATED_DOF,
    },
}


class MpcConfigError(ValueError):
    """Refusal. The message is what the operator gets instead of a node."""


def _positive(value: float) -> bool:
    return math.isfinite(value) and value > 0.0


def _non_negative(value: float) -> bool:
    return math.isfinite(value) and value >= 0.0


def _offender(block: dict, names, ok) -> str:
    """
    `field[row] = value` of the first entry `ok` rejects, or `""` if none is.

    One helper so that a refusal over six weight vectors still names the entry
    that caused it; the C++ named the group and the group is not the quantity.
    """
    for name in names:
        values = np.atleast_1d(np.asarray(block[name], dtype=float)).ravel()
        for row, value in enumerate(values):
            if not ok(value):
                where = f"{name}[{row}]" if values.size > 1 else name
                return f"{where} = {value:g}"
    return ""


def _refuse(offender: str, reason: str) -> None:
    if offender:
        raise MpcConfigError(f"{offender}: {reason}")


def check_settings(parameters: dict, hydraulics: dict) -> None:
    """
    Refuse a configuration the OCP cannot be posed on.

    `horizon_length >= 2` is **not** here: `problem.shooting_intervals` already
    refuses it, and on the path `scripts/export_ocp.py` takes, which never
    reaches this module.
    """
    for block, widths in WIDTHS.items():
        for name, width in widths.items():
            got = np.atleast_1d(np.asarray(parameters[block][name])).size
            if got != width:
                raise MpcConfigError(
                    f"{block}.{name} carries {got} entries and the problem is posed "
                    f"on {width}; a short list is not a shorter machine, it is a "
                    "bound that covers fewer axes than there are rows"
                )

    _refuse(
        _offender(parameters, ("Ts",), _positive),
        "T_s must be finite and positive; it is the control cycle of `mpc` §4, the "
        "knot spacing of the horizon and the integrator step, and those are one "
        "number rather than three",
    )
    _refuse(
        _offender(parameters, ("solve_budget",), _positive),
        "the solve budget is what a cycle may spend and must be finite and positive",
    )
    _refuse(
        _offender(parameters, ("levenberg_marquardt",), _non_negative),
        "the Levenberg-Marquardt regularisation must be finite and >= 0; it is added "
        "to the Hessian on every cycle and a negative one subtracts from it",
    )
    _refuse(
        _offender(parameters, ("sensor_to_valve_delay",), _non_negative),
        "the transport dead time must be finite and >= 0; zero means none has been "
        "measured and nothing is propagated, which is honest, whereas a negative one "
        "is a plan for the past",
    )

    weights = parameters["weights"]
    _refuse(
        _offender(
            weights,
            ("q_a", "dq_a", "q_u", "dq_u", "tau_a", "u", "lag", "progress_accel"),
            _non_negative,
        ),
        "every weight of `mpc` §2 must be finite and non-negative: the Gauss-Newton "
        "Hessian is J' W J and a negative entry is what turns 'positive semi-definite "
        "by construction' into a hope",
    )
    _refuse(
        _offender(weights, ("terminal_scale",), _non_negative),
        "the terminal weight scale multiplies that same Hessian's leading block and "
        "must be finite and >= 0",
    )
    _refuse(
        _offender(weights, ("progress_rate",), _positive),
        "the progress-rate weight must be finite and **positive**: it is the only "
        "price on spending time, and at zero the optimizer stops the plan for free "
        "wherever tracking is hard, which is a controller that never arrives rather "
        "than one that tracks time-indexed",
    )

    limits = parameters["limits"]
    lower = np.asarray(limits["q_a_lower"], dtype=float)
    upper = np.asarray(limits["q_a_upper"], dtype=float)
    margin = np.asarray(limits["q_a_margin"], dtype=float)
    _refuse(
        _offender(limits, ("q_a_lower", "q_a_upper"), math.isfinite),
        "constraint 1 of `mpc` §3 needs a finite control-safe range on every actuated "
        "row (`parameters.md` §2 -- not the URDF)",
    )
    _refuse(
        _offender(limits, ("q_a_margin",), _non_negative),
        "constraint 1's safety margin must be finite and >= 0 on every row; zero is "
        "legal and means the bound is the control-safe limit itself",
    )
    for row in range(cs.K_ACTUATED_DOF):
        if not lower[row] < upper[row]:
            raise MpcConfigError(
                f"q_a_lower[{row}] = {lower[row]:g} is not below q_a_upper[{row}] = "
                f"{upper[row]:g}; actuated row {row} has an empty control-safe "
                "position range"
            )
        if not lower[row] + margin[row] < upper[row] - margin[row]:
            raise MpcConfigError(
                f"q_a_margin[{row}] = {margin[row]:g} closes actuated row {row}'s own "
                f"control-safe range [{lower[row]:g}, {upper[row]:g}]; a margin is a "
                "tightening, not a replacement for the limit"
            )
    _refuse(
        _offender(limits, ("dq_a_max", "q_u_max", "dq_u_max", "u_max"), _positive),
        "constraints 2 to 5 of `mpc` §3 each need a finite positive bound; an invented "
        "or absent one is the silent stub the model API's contract 5 exists to prevent",
    )
    rate_max = float(limits["progress_rate_max"])
    if not math.isfinite(rate_max) or rate_max < K_PROGRESS_RATE_REFERENCE:
        raise MpcConfigError(
            f"progress_rate_max = {rate_max:g} is below the nominal rate "
            f"{K_PROGRESS_RATE_REFERENCE:g}; below one the reference could never be "
            "spent at its own nominal rate, which is the behaviour every other page "
            "in the wiki describes"
        )
    _refuse(
        _offender(limits, ("progress_accel_max",), _positive),
        "the progress-acceleration bound must be finite and positive; at zero the "
        "progress rate is frozen at whatever it was carried in with and the state is "
        "decoration",
    )

    _refuse(
        _offender(hydraulics, ("pump_flow_max",), _positive),
        "constraint 7 of `mpc` §3 needs a finite positive Q_P^max; `parameters.md` §4 "
        "states it once, at 1.4e-3 m^3/s measured 2023 pre-retrofit and not "
        "re-verified, and a weak number with its provenance recorded is worth more "
        "than an absent constraint",
    )
    factor = float(hydraulics["pump_flow_planning_factor"])
    if not _positive(factor) or factor > 1.0:
        raise MpcConfigError(
            f"pump_flow_planning_factor = {factor:g} is outside (0, 1]; it is "
            "`parameters.md` §4's 0.95x discount on Q_P^max, it is not a second kappa "
            "and it is not a place to buy margin back"
        )
    _refuse(
        _offender(hydraulics, ("system_pressure_pa",), _positive),
        "constraint 6 of `mpc` §3 needs a finite positive relief pressure to derive "
        "F_i^max from (`robot_model` §4.6); it is the one number that cannot come off "
        "the description",
    )

    _refuse(
        _offender(
            parameters["slack"],
            ("q_u", "dq_u", "cylinder_force", "pump_flow"),
            _positive,
        ),
        "every L1 slack price of `mpc` §3.2 must be finite and positive; a soft "
        "constraint priced at zero is not a soft constraint, it is an absent one, and "
        "it would be absent silently",
    )


def check_payload(mass_kg: float, com_m) -> None:
    """
    Refuse a payload the dynamics cannot carry.

    No `valid` flag and no inertia, unlike `check_payload` in the C++:
    `crane_msgs/Payload` carries neither, so an undeclared payload is refused
    where it is declared (`node.py`'s `_payload_from_message`) and the body
    bound into `p` here is a point mass.
    """
    mass = float(mass_kg)
    if not _non_negative(mass):
        raise MpcConfigError(
            f"the payload mass is {mass:g} kg, and a mass must be finite and >= 0: a "
            "negative one is not a light load, and a non-finite one reaches every "
            "stage of the horizon at once through `p`"
        )
    com = np.asarray(com_m, dtype=float).ravel()
    if com.size != 3 or not np.all(np.isfinite(com)):
        raise MpcConfigError(
            f"the payload's centre of mass is {com.tolist()} and must be three finite "
            "numbers; it is the moment arm the dynamics carry, and a non-finite entry "
            "is a solve that answers NaN"
        )

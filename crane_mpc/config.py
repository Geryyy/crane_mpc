"""
What `crane_mpc` refuses to start on.

A bad configuration is a refusal at startup, not a NaN blamed on the solver, and
every message names the quantity, its value and why the bound exists.

`parameter_dict` reshapes the generated `Params` object into the plain dicts
`problem`/`solver` read; a yaml on disk arrives in the same shape.
`control_safe_box`, `command_domain` and the hydraulic constants are asked of
`crane_model` rather than declared, so no deployment carries a second copy of
the machine.
"""

from __future__ import annotations

import math

import numpy as np
from crane_model import symbolic as cs
from crane_model.conventions import (
    ACTUATED_INDICES,
    CONTROL_SAFE_AXES,
    canonical_joints,
    control_safe_limits,
)
from crane_model.velocity_loop import load_velocity_loop

#: Vector-valued fields and how wide each must be: out of a yaml, a short list
#: silently covers fewer axes than the problem has rows.
WIDTHS = {
    "weights": {
        "q_a": cs.K_ACTUATED_DOF,
        "dq_a": cs.K_ACTUATED_DOF,
        "tau_a": cs.K_ACTUATED_DOF,
        "u": cs.K_ACTUATED_DOF,
        "du": cs.K_ACTUATED_DOF,
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
        "du_max": cs.K_PLANNED_DOF,
    },
    "slack": {
        "q_u": cs.K_PASSIVE_DOF,
        "dq_u": cs.K_PASSIVE_DOF,
        "cylinder_force": cs.K_ACTUATED_DOF,
    },
}


#: Scalar/weight/limit/slack names `problem`/`solver` read. One list, so a
#: yaml parameter can't be declared and quietly miss the OCP.
SCALARS = (
    "Ts",
    "levenberg_marquardt",
    "solve_budget",
    "sensor_to_valve_delay",
    # C4: the command a state, `u` its rate. Read as `bool(...)`.
    "command_state",
)
WEIGHTS = (
    "q_a",
    "dq_a",
    "q_u",
    "dq_u",
    "tau_a",
    "u",
    "du",
    "lag",
    "tool",
    "progress",
    "progress_rate",
    "progress_accel",
    "terminal_scale",
)
#: Constraint 1's box and constraint 2's bound. Not declared as parameters:
#: crane_model owns them and `crane_planning` intersects its description-read
#: limits with the same rows, so a second copy here is how the two stacks drift.
BOX = ("q_a_lower", "q_a_upper", "q_a_margin", "dq_a_max")

#: Constraint 5's bound, derived rather than written: `u` is a velocity at Psi's
#: input, so the smaller of Psi's identified domain and the control-safe speed
#: bounds it. Both tables are crane_model's, so a written product would not
#: survive a narrowing of either factor.
DERIVED = ("u_max",)

#: Declared limits, i.e. the ones a deployment still writes.
LIMITS = (
    "q_u_max",
    "dq_u_max",
    "progress_rate_headroom",
    "progress_accel_max",
    "du_max",
)
SLACK = ("q_u", "dq_u", "cylinder_force", "pump_flow")


class MpcConfigError(ValueError):
    """Refusal. The message is what the operator gets instead of a node."""


def parameter_dict(values) -> dict:
    """
    Shape the parameters as the problem reads them.

    One shape whether from the parameter server or yaml; `horizon_length` is
    the one integer, the rest floats/sequences read as-is.
    """
    shaped = {name: float(getattr(values, name)) for name in SCALARS}
    shaped["horizon_length"] = int(values.horizon_length)
    for block, names in (("weights", WEIGHTS), ("limits", LIMITS), ("slack", SLACK)):
        group = getattr(values, block)
        shaped[block] = {name: getattr(group, name) for name in names}
    shaped["limits"].update(machine_limits())
    return shaped


def machine_limits() -> dict:
    """Every limit row crane_model owns, in this problem's actuated order."""
    box = control_safe_box()
    return box | {"u_max": command_domain(box["dq_a_max"])}


def control_safe_box() -> dict:
    """
    Read the four control-safe rows, in this problem's actuated order.

    Read off `crane_model`, never declared: rows keyed by axis name there and
    by index here, and the order is derived through `canonical_joints` rather
    than written down, since only the name is stable across the two packages.
    """
    box = control_safe_limits()
    joints = canonical_joints()
    axis_of = {joint: axis for axis, joint in CONTROL_SAFE_AXES.items()}
    axes = [axis_of[joints[index]] for index in ACTUATED_INDICES]
    return {row: [box[row][axis] for axis in axes] for row in BOX}


def command_domain(dq_a_max) -> list:
    """
    Constraint 5's `u^+`: the smaller of Psi's domain and the speed bound.

    `u_clamp_min`/`u_clamp_max` are Psi's identified domain, asymmetric there and
    one magnitude here, so the smaller side is taken. The tool carries no clamp,
    so its row is the speed bound alone.

    **This bounds what the MPC asks for, not what reaches Psi.** The deployed
    clamp is on the inner loop's PI term alone and the feedforward is added
    outside it, so a stalled axis can still hand Psi more than its domain.
    """
    gains, _rate_hz = load_velocity_loop()
    joints = canonical_joints()
    return [
        min(
            abs(gains[joints[index]].u_clamp_min),
            gains[joints[index]].u_clamp_max,
            float(dq_a_max[axis]),
        )
        for axis, index in enumerate(ACTUATED_INDICES)
    ]


def _positive(value: float) -> bool:
    return math.isfinite(value) and value > 0.0


def _non_negative(value: float) -> bool:
    return math.isfinite(value) and value >= 0.0


def _offender(block: dict, names, ok) -> str:
    """
    `field[row] = value` of the first entry `ok` rejects, or `""` if none is.

    One helper so a refusal over six weight vectors names the entry, not the group.
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

    `horizon_length >= 2` isn't here: `problem.shooting_intervals` already
    refuses it on the path `scripts/export_ocp.py` takes.
    """
    for block, widths in WIDTHS.items():
        for name, width in widths.items():
            got = np.atleast_1d(np.asarray(parameters[block][name])).size
            if got != width:
                raise MpcConfigError(
                    f"{block}.{name} carries {got} entries and the problem is posed "
                    f"on {width}; a short list covers fewer axes than there are rows"
                )

    _refuse(
        _offender(parameters, ("Ts",), _positive),
        "T_s must be finite and positive: it is the control cycle of `mpc` §4, the "
        "horizon's knot spacing and the integrator step, all one number",
    )
    _refuse(
        _offender(parameters, ("solve_budget",), _positive),
        "the solve budget is what a cycle may spend and must be finite and positive",
    )
    _refuse(
        _offender(parameters, ("levenberg_marquardt",), _non_negative),
        "the Levenberg-Marquardt regularisation is added to the Hessian every cycle "
        "and must be finite and >= 0; a negative one subtracts from it",
    )
    # The one timing invariant, checked once and nowhere else: `Ocp` rolls the
    # dead time as a single `Ts` step and the next cycle measures at stage 0 of
    # this solve, and neither holds at any other delay.
    Ts = float(parameters["Ts"])
    delay = float(parameters["sensor_to_valve_delay"])
    if not math.isclose(delay, Ts, rel_tol=1e-9):
        raise MpcConfigError(
            f"sensor_to_valve_delay = {delay:g} s must equal Ts = {Ts:g} s; at any "
            "other delay the state enters the next roll off its own measurement "
            "instant -- one-signed in C3's force rows"
        )
    ticks = Ts * load_velocity_loop()[1]
    if ticks < 1.0 or not math.isclose(ticks, round(ticks), rel_tol=1e-9):
        raise MpcConfigError(
            f"Ts = {Ts:g} s is {ticks:g} JTC ticks (velocity_loop.yaml rate_hz) and "
            "must be a whole number of them; off-tick knots are sampled by "
            "interpolation rather than played"
        )

    weights = parameters["weights"]
    _refuse(
        _offender(
            weights,
            ("q_a", "dq_a", "q_u", "dq_u", "tau_a", "u", "du", "lag", "progress_accel"),
            _non_negative,
        ),
        "every weight of `mpc` §2 must be finite and non-negative: the Gauss-Newton "
        "Hessian is J' W J, which a negative entry stops being semi-definite",
    )
    _refuse(
        _offender(weights, ("terminal_scale",), _non_negative),
        "the terminal weight scale multiplies that same Hessian's leading block and "
        "must be finite and >= 0",
    )
    _refuse(
        _offender(weights, ("progress",), _positive),
        "the progress weight must be finite and **positive**: the progress state is "
        "the path parameter, so at zero the machine sits on the path rather than "
        "travelling it -- a controller that never arrives",
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
        "constraint 1's safety margin must be finite and >= 0 on every row; zero means "
        "the bound is the control-safe limit itself",
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
        _offender(
            limits, ("dq_a_max", "q_u_max", "dq_u_max", "u_max", "du_max"), _positive
        ),
        "constraints 2 to 5 of `mpc` §3 each need a finite positive bound; an invented "
        "or absent one is a silent stub",
    )
    # A multiple of the plan's own pace, so the floor is one whatever the grid is.
    # Declared as an absolute rate the floor sat at 2.34x nominal at the shipped
    # grid, which forbade every honest catch-up ceiling.
    headroom = float(limits["progress_rate_headroom"])
    if not math.isfinite(headroom) or headroom < 1.0:
        raise MpcConfigError(
            f"progress_rate_headroom = {headroom:g} is below one, so the reference "
            "could never be spent at its own pace"
        )
    _refuse(
        _offender(limits, ("progress_accel_max",), _positive),
        "the progress-acceleration bound must be finite and positive; at zero the "
        "progress rate is frozen at whatever it was carried in with",
    )

    _refuse(
        _offender(hydraulics, ("pump_flow_max",), _positive),
        "constraint 7 of `mpc` §3 needs a finite positive Q_P^max; `parameters.md` §4 "
        "states it once, weak (measured 2023 pre-retrofit) but with its provenance",
    )
    factor = float(hydraulics["pump_flow_planning_factor"])
    if not _positive(factor) or factor > 1.0:
        raise MpcConfigError(
            f"pump_flow_planning_factor = {factor:g} is outside (0, 1]; it is "
            "`parameters.md` §4's discount on Q_P^max, not a second kappa"
        )
    _refuse(
        _offender(hydraulics, ("system_pressure_pa",), _positive),
        "constraint 6 of `mpc` §3 needs a finite positive relief pressure to derive "
        "F_i^max from (`robot_model` §4.6); it cannot come off the description",
    )

    _refuse(
        _offender(
            parameters["slack"],
            ("q_u", "dq_u", "cylinder_force", "pump_flow"),
            _positive,
        ),
        "every L1 slack price of `mpc` §3.2 must be finite and positive; a soft "
        "constraint priced at zero is silently an absent one",
    )


def check_payload(mass_kg: float, com_m) -> None:
    """
    Refuse a payload the dynamics cannot carry.

    `crane_msgs/Payload` carries no `valid` flag and no inertia, so an undeclared
    payload is refused where it is declared (`reports.payload_from_message`) and
    the body bound into `p` is a point mass.
    """
    mass = float(mass_kg)
    if not _non_negative(mass):
        raise MpcConfigError(
            f"the payload mass is {mass:g} kg and must be finite and >= 0: a negative "
            "one is not a light load, and a non-finite one reaches every stage "
            "through `p`"
        )
    com = np.asarray(com_m, dtype=float).ravel()
    if com.size != 3 or not np.all(np.isfinite(com)):
        raise MpcConfigError(
            f"the payload's centre of mass is {com.tolist()} and must be three finite "
            "numbers; it is the moment arm the dynamics carry"
        )

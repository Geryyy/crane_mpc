"""
`wire_chain`'s cycle as JSON over UDP, for PlotJuggler's UDP server.

PlotJuggler: Streaming -> UDP server, the port, message protocol JSON, timestamp
field `t`. Nested keys come out as `/group/name` series. One datagram per cycle,
fire and forget: nothing listening costs nothing.
"""

import dataclasses
import functools
import json
import socket

import casadi as ca
import numpy as np
from crane_model.conventions import (
    ACTUATED_INDICES,
    PASSIVE_INDICES,
    PLANNED_INDICES,
    TOOL_AXIS,
    Tool,
    canonical_joints,
)
from crane_model.symbolic import (
    NP,
    NX_RIGID,
    P_PAYLOAD_COM,
    P_PAYLOAD_MASS,
    P_TOOL_POSITION,
    X_PASSIVE_POSITION,
    X_PLANNED_POSITION,
    CraneSymbolicModel,
)

PORT = 9870  # PlotJuggler's UDP default


def finite(group: dict) -> dict:
    """Nested series with None and non-finite values left out."""
    out = {}
    for key, value in group.items():
        if isinstance(value, dict):
            out[key] = finite(value)
        elif value is not None and np.isfinite(value):
            out[key] = float(value)
    return out


@functools.lru_cache(maxsize=1)
def coupling(description_xml: str) -> ca.Function:
    """B(x, p) = d(ddq_sway)/d(ddq_planned) = -M_uu^-1 M_ua, 2 x 5."""
    model = CraneSymbolicModel(description_xml, Tool.PZS100.value)
    return ca.Function(
        "B", [model.x, model.p], [-ca.solve(model.mass_uu, model.mass_ua)]
    )


class PlotJuggler:
    def __init__(
        self,
        description_xml,
        du_max,
        payload_mass,
        payload_com,
        port=PORT,
        host="127.0.0.1",
    ) -> None:
        self.address = (host, port)
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.names = canonical_joints()
        self.coupling = coupling(description_xml)
        self.a_max = np.asarray(du_max, float)[: len(PLANNED_INDICES)]
        self.p = np.zeros(NP)
        self.p[P_PAYLOAD_MASS] = payload_mass
        self.p[P_PAYLOAD_COM : P_PAYLOAD_COM + 3] = payload_com

    def authority(self, q) -> dict:
        """Sway accel per unit joint accel, and the most the joints can give at du_max."""
        x = np.zeros(NX_RIGID)
        x[X_PLANNED_POSITION : X_PLANNED_POSITION + len(PLANNED_INDICES)] = q[
            list(PLANNED_INDICES)
        ]
        x[X_PASSIVE_POSITION : X_PASSIVE_POSITION + 2] = q[list(PASSIVE_INDICES)]
        self.p[P_TOOL_POSITION] = q[ACTUATED_INDICES[TOOL_AXIS]]
        b = np.asarray(self.coupling(x, self.p))
        axes = [self.names[i] for i in PLANNED_INDICES]
        rows = ("tip", "tilt")
        return {
            "authority": {
                r: float(np.abs(b[k]) @ self.a_max) for k, r in enumerate(rows)
            },
            "coupling": {
                r: dict(zip(axes, map(float, b[k]))) for k, r in enumerate(rows)
            },
        }

    def joints(self, values, indices) -> dict:
        return {self.names[i]: float(v) for i, v in zip(indices, values)}

    def send(self, t: float, **groups) -> None:
        """Send one datagram; each group a `dict` of series or a scalar, non-finite dropped."""
        message = {"t": t, **finite(groups)}
        try:
            self.socket.sendto(json.dumps(message).encode(), self.address)
        except OSError:
            pass  # nothing listening, or past the UDP limit: a plot gap, not an abort

    def cycle(
        self,
        t,
        plant,
        sensor,
        sway,
        solution,
        u0,
        dq_ref,
        indices,
        terms,
        error,
        tcp_error,
        path_error,
        tcp_path_error,
    ) -> None:
        """One MPC cycle: true and measured state, command, solver, cost split."""
        planned, passive = indices
        solver = (
            {}
            if solution is None
            else {
                "solve_ms": 1e3 * solution.solve_time_s,
                "qp_iterations": float(solution.qp_iterations),
                "status": float(solution.status),
            }
        )
        cost = {} if terms is None else dataclasses.asdict(terms)
        total = sum(cost.values())
        share = {k: v / total for k, v in cost.items()} if total > 0 else {}
        self.send(
            t,
            **self.authority(plant.q),
            tracking_error=self.joints(error, planned),
            tcp_error=tcp_error,
            path_error=self.joints(path_error, planned),
            tcp_path_error=tcp_path_error,
            cost=cost,
            cost_share=share,
            sway=self.joints(sway, passive),
            dq_u=self.joints(plant.dq[passive], passive),
            q_u_measured=self.joints(sensor.q[passive], passive),
            dq_u_measured=self.joints(sensor.dq[passive], passive),
            dq_a=self.joints(plant.dq[planned], planned),
            dq_a_ref=self.joints(dq_ref, planned),
            u0=self.joints(u0, planned),
            solver=solver,
        )

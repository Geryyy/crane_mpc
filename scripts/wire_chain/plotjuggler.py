"""
`wire_chain`'s cycle as JSON over UDP, for PlotJuggler's UDP server.

PlotJuggler: Streaming -> UDP server, the port, message protocol JSON, timestamp
field `t`. Nested keys come out as `/group/name` series. One datagram per cycle,
fire and forget: nothing listening costs nothing.
"""

import dataclasses
import json
import socket

import numpy as np
from crane_model.conventions import canonical_joints

PORT = 9870  # PlotJuggler's UDP default


class PlotJuggler:
    def __init__(self, port: int = PORT, host: str = "127.0.0.1") -> None:
        self.address = (host, port)
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.names = canonical_joints()

    def joints(self, values, indices) -> dict:
        return {self.names[i]: float(v) for i, v in zip(indices, values)}

    def send(self, t: float, **groups) -> None:
        """Send one datagram; each group a `dict` of series or a scalar, non-finite dropped."""
        message = {"t": t}
        for key, value in groups.items():
            if isinstance(value, dict):
                value = {k: v for k, v in value.items() if np.isfinite(v)}
            elif value is None or not np.isfinite(value):
                continue
            message[key] = value
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
            tracking_error=self.joints(error, planned),
            tcp_error=tcp_error,
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

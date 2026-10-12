from __future__ import annotations
import csv
import json
import math
import sys
import traceback
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import kite_x_rotational_dynamics as rotational
import kite_x_6dof_plant as plant
import kite_x_four_leg_launch as launch
import kite_x_transition_flight_control as flight

OUT = Path(__file__).resolve().parent / "kite_x_validation_outputs"
OUT.mkdir(exist_ok=True)


def scenario_metrics(name: str, result: flight.SimulationResult) -> dict:
    euler_deg = np.rad2deg(result.euler_zyx)
    modes = np.asarray(result.mode)
    state = result.state
    qnorm = np.linalg.norm(state[:, 6:10], axis=1)
    return {
        "scenario": name,
        "duration_s": float(result.time[-1]),
        "final_x_m": float(state[-1, 0]),
        "final_y_m": float(state[-1, 1]),
        "final_z_m": float(state[-1, 2]),
        "max_z_m": float(np.max(state[:, 2])),
        "min_z_m": float(np.min(state[:, 2])),
        "max_abs_roll_deg": float(np.max(np.abs(euler_deg[:, 0]))),
        "max_abs_pitch_deg": float(np.max(np.abs(euler_deg[:, 1]))),
        "max_abs_yaw_deg": float(np.max(np.abs(euler_deg[:, 2]))),
        "max_body_rate_deg_s": float(np.max(np.linalg.norm(np.rad2deg(state[:, 10:13]), axis=1))),
        "flight_mode_samples": int(np.sum(modes == "FLIGHT")),
        "abort_mode_samples": int(np.sum(modes == "ABORT")),
        "first_flight_time_s": float(result.time[np.where(modes == "FLIGHT")[0][0]]) if np.any(modes == "FLIGHT") else None,
        "first_abort_time_s": float(result.time[np.where(modes == "ABORT")[0][0]]) if np.any(modes == "ABORT") else None,
        "max_total_rotor_thrust_N": float(np.max(result.total_rotor_thrust)),
        "rotor_saturation_samples": int(np.sum(np.any(result.saturated, axis=1))),
        "all_states_finite": bool(np.all(np.isfinite(state))),
        "max_quaternion_norm_error": float(np.max(np.abs(qnorm - 1.0))),
        "max_vertical_speed_m_s": float(np.max(np.abs((np.array([plant.quat_to_rotation_matrix(q) for q in state[:, 6:10]]) @ state[:, 3:6, None])[:, 2, 0]))),
    }


def save_timeseries(name: str, result: flight.SimulationResult) -> Path:
    path = OUT / f"{name}_timeseries.csv"
    euler = np.rad2deg(result.euler_zyx)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "time_s", "x_m", "y_m", "z_m", "u_body_m_s", "v_body_m_s", "w_body_m_s",
            "roll_deg", "pitch_deg", "yaw_deg", "p_rad_s", "q_rad_s", "r_rad_s",
            "mode", "T1_N", "T2_N", "T3_N", "T4_N", "total_thrust_N",
            "rotor_saturated", "transition_reason"
        ])
        for i, t in enumerate(result.time):
            s = result.state[i]
            w.writerow([float(t), *s[:6], *euler[i], *s[10:13], result.mode[i],
                        *result.rotor_thrusts[i], float(result.total_rotor_thrust[i]),
                        bool(np.any(result.saturated[i])), result.transition_reason[i]])
    return path


def plot_scenarios(results: dict[str, flight.SimulationResult]) -> None:
    fig, ax = plt.subplots()
    for name, res in results.items():
        ax.plot(res.time, res.position_i[:, 2], label=name)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Altitude z (m)")
    ax.set_title("KITE-X prototype scenarios: altitude")
    ax.grid(True); ax.legend(); fig.tight_layout()
    fig.savefig(OUT / "altitude_comparison.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots()
    for name, res in results.items():
        ax.plot(res.time, np.rad2deg(res.euler_zyx[:, 0]), label=f"{name}: roll")
        ax.plot(res.time, np.rad2deg(res.euler_zyx[:, 1]), linestyle="--", label=f"{name}: pitch")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Angle (deg)")
    ax.set_title("KITE-X prototype scenarios: attitude")
    ax.grid(True); ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(OUT / "attitude_comparison.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots()
    for name, res in results.items():
        ax.step(res.time, [1 if m == "FLIGHT" else (-1 if m == "ABORT" else 0) for m in res.mode],
                where="post", label=name)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Mode (0 launch, 1 flight, -1 abort)")
    ax.set_title("KITE-X transition state")
    ax.grid(True); ax.legend(); fig.tight_layout()
    fig.savefig(OUT / "transition_modes.png", dpi=160); plt.close(fig)


def main() -> int:
    print("KITE-X INTEGRATED VALIDATION RUNNER")
    print("All outputs are simulation results with provisional parameters.\n")
    checks = []
    for module in (rotational, plant, launch, flight):
        try:
            module.run_self_tests()
            checks.append({"module": module.__name__, "status": "PASS"})
        except Exception as exc:
            checks.append({"module": module.__name__, "status": "FAIL", "error": repr(exc)})
            print(f"SELF-TEST FAILURE in {module.__name__}: {exc}")
            (OUT / "validation_traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
            return 1

    duration, dt = 1.2, 0.0005
    cases = {
        "nominal": dict(
            launch_cfg=launch.LaunchConfig(),
            initial_position_i=(0.0, 0.0, 0.10),
            initial_omega_b=(0.0, 0.0, 0.0),
        ),
        "transition_timeout_abort": dict(
            # Forces the timeout path and verifies that abort inhibits rotor output.
            launch_cfg=launch.LaunchConfig(launch_force_total=0.0, launch_duration=0.0, contact_enabled=False),
            transition_cfg=flight.TransitionConfig(apex_timeout_s=0.03, require_apex=True),
            initial_position_i=(0.0, 0.0, 2.0),
            initial_omega_b=(math.radians(18.0), 0.0, 0.0),
        ),
        "attitude_recovery_smoke_test": dict(
            # Isolated closed-loop test: bypasses apex requirement to test feedback wiring.
            launch_cfg=launch.LaunchConfig(launch_force_total=0.0, launch_duration=0.0, contact_enabled=False),
            transition_cfg=flight.TransitionConfig(apex_timeout_s=0.5, require_apex=False),
            initial_position_i=(0.0, 0.0, 0.8),
            initial_quaternion_ib=(math.cos(math.radians(3.0)/2), 0.0, math.sin(math.radians(3.0)/2), 0.0),
            initial_omega_b=(0.0, math.radians(4.0), 0.0),
        ),
    }

    results = {}
    metrics = []
    for name, kwargs in cases.items():
        print(f"\nRunning scenario: {name} ...")
        res = flight.simulate_transition_and_flight(duration=duration, dt=dt, **kwargs)
        results[name] = res
        m = scenario_metrics(name, res)
        metrics.append(m)
        save_timeseries(name, res)
        print(f"  final altitude: {m['final_z_m']:.3f} m")
        print(f"  max |roll| / |pitch|: {m['max_abs_roll_deg']:.2f} / {m['max_abs_pitch_deg']:.2f} deg")
        print(f"  flight samples: {m['flight_mode_samples']}; abort samples: {m['abort_mode_samples']}")
        print(f"  finite states: {m['all_states_finite']}; max quaternion norm error: {m['max_quaternion_norm_error']:.2e}")
        if name == "attitude_recovery_smoke_test":
            print(f"  pitch: {np.rad2deg(res.euler_zyx[0,1]):.3f} -> {np.rad2deg(res.euler_zyx[-1,1]):.3f} deg")
            print(f"  pitch rate: {np.rad2deg(res.omega_b[0,1]):.3f} -> {np.rad2deg(res.omega_b[-1,1]):.3f} deg/s")
        if name == "transition_timeout_abort":
            abort_rows = np.asarray(res.mode) == "ABORT"
            print(f"  abort rotor thrust max: {np.max(res.total_rotor_thrust[abort_rows]) if np.any(abort_rows) else float('nan'):.3f} N")

    # Separate stiff-contact launch test at a smaller time step. The explicit RK4
    # contact model needs this smaller step for the asymmetric-force case.
    print("\nRunning asymmetric four-leg launch-only test (small time step) ...")
    asym_cfg = launch.LaunchConfig(
        leg_force_scales=np.array([1.08, 0.92, 1.00, 1.00]),
        leg_delays=np.array([0.0, 0.0015, 0.0005, 0.0020]),
    )
    asym = launch.simulate_launch(asym_cfg, duration=0.10, dt=0.00005)
    asym_path = OUT / "asymmetric_launch_only.csv"
    with asym_path.open("w", newline="", encoding="utf-8") as fcsv:
        w = csv.writer(fcsv)
        w.writerow(["time_s", "x_m", "y_m", "z_m", "roll_deg", "pitch_deg", "yaw_deg",
                    "p_rad_s", "q_rad_s", "r_rad_s", "launch_fx_N", "launch_fy_N", "launch_fz_N",
                    "moment_x_Nm", "moment_y_Nm", "moment_z_Nm"])
        eul = np.rad2deg(asym.plant.euler_zyx)
        for i, t in enumerate(asym.plant.time):
            w.writerow([float(t), *asym.plant.position_i[i], *eul[i], *asym.plant.omega_b[i],
                        *asym.launch_force_b[i], *asym.moment_total_b[i]])
    asym_metrics = {
        "scenario": "asymmetric_launch_only",
        "duration_s": float(asym.plant.time[-1]),
        "max_abs_roll_deg": float(np.max(np.abs(np.rad2deg(asym.plant.euler_zyx[:, 0])))),
        "max_abs_pitch_deg": float(np.max(np.abs(np.rad2deg(asym.plant.euler_zyx[:, 1])))),
        "max_abs_yaw_deg": float(np.max(np.abs(np.rad2deg(asym.plant.euler_zyx[:, 2])))),
        "max_body_rate_rad_s": float(np.max(np.linalg.norm(asym.plant.omega_b, axis=1))),
        "all_states_finite": bool(np.all(np.isfinite(np.column_stack([asym.plant.position_i, asym.plant.velocity_b, asym.plant.quaternion_ib, asym.plant.omega_b])))),
        "max_quaternion_norm_error": float(np.max(np.abs(np.linalg.norm(asym.plant.quaternion_ib, axis=1) - 1.0))),
        "note": "Launch-only; not coupled to rotor transition control in this scenario."
    }
    (OUT / "asymmetric_launch_metrics.json").write_text(json.dumps(asym_metrics, indent=2), encoding="utf-8")
    print(f"  max |roll| / |pitch|: {asym_metrics['max_abs_roll_deg']:.3f} / {asym_metrics['max_abs_pitch_deg']:.3f} deg")
    print(f"  finite states: {asym_metrics['all_states_finite']}; max quaternion norm error: {asym_metrics['max_quaternion_norm_error']:.2e}")

    plot_scenarios(results)
    (OUT / "scenario_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (OUT / "self_test_summary.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")

    report = [
        "KITE-X INTEGRATED VALIDATION SUMMARY",
        "====================================",
        "Status: prototype numerical validation only; not hardware or flight validation.",
        "Parameters in the current modules are provisional and must be reconciled with CAD.",
        "",
        "Subsystem self-tests:",
        *[f"- {c['module']}: {c['status']}" for c in checks],
        "",
        "Scenario metrics (see scenario_metrics.json for full values):",
    ]
    for m in metrics:
        report += [
            f"- {m['scenario']}: final z={m['final_z_m']:.4f} m; max z={m['max_z_m']:.4f} m; "
            f"max |roll|={m['max_abs_roll_deg']:.3f} deg; max |pitch|={m['max_abs_pitch_deg']:.3f} deg; "
            f"flight samples={m['flight_mode_samples']}; abort samples={m['abort_mode_samples']}; "
            f"finite={m['all_states_finite']}"
        ]
    report += [
        "",
        "Caveats:",
        "1. This runner tests software behavior and scenario sensitivity, not physical accuracy.",
        "2. Ground-contact stiffness/damping, rotor map, controller gains, geometry, mass and inertia need validation.",
        "3. A scenario entering FLIGHT does not establish stable or feasible real-world flight.",
        "4. Rotor allocation clips individual thrust; requested wrench may not be achieved under saturation.",
        "5. Sensor noise, latency, motor dynamics, aerodynamic drag and detailed leg compliance are not fully modelled.",
    ]
    (OUT / "README_validation.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"\nValidation outputs written to: {OUT}")
    print("  README_validation.txt")
    print("  scenario_metrics.json")
    print("  *_timeseries.csv and asymmetric_launch_only.csv")
    print("  altitude_comparison.png, attitude_comparison.png, transition_modes.png")
    print("IMPORTANT: do not interpret this as a hardware-ready flight model.")
    return 0 if all(m["all_states_finite"] for m in metrics) else 2

if __name__ == "__main__":
    raise SystemExit(main())

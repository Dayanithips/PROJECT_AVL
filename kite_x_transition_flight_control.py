from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from kite_x_6dof_plant import (
    GRAVITY, PROVISIONAL_INERTIA, pack_state, plant_rhs, rk4_step,
    quat_to_rotation_matrix, quat_to_euler_zyx, validate_inertia,
)
from kite_x_four_leg_launch import LaunchConfig, evaluate_forces

Array = np.ndarray


@dataclass(frozen=True)
class TransitionConfig:
    vertical_speed_threshold: float = 0.15  # m/s, absolute value
    max_tilt_deg: float = 5.0
    max_rate_deg_s: float = 10.0
    apex_timeout_s: float = 1.5
    require_apex: bool = True


@dataclass(frozen=True)
class RotorConfig:
    # X-layout rotor centres, body coordinates. Replace with CAD values.
    arm: float = 0.15
    yaw_moment_per_newton: float = 0.015  # N*m per N thrust, provisional
    max_thrust_per_rotor: float = 15.0  # N, provisional
    min_thrust_per_rotor: float = 0.0
    spin_directions: tuple = (1.0, -1.0, 1.0, -1.0)

    def __post_init__(self):
        if self.arm <= 0 or self.yaw_moment_per_newton < 0:
            raise ValueError("Rotor arm must be positive and yaw coefficient nonnegative")
        if self.max_thrust_per_rotor <= 0 or self.min_thrust_per_rotor < 0:
            raise ValueError("Rotor thrust limits are invalid")
        if self.min_thrust_per_rotor > self.max_thrust_per_rotor:
            raise ValueError("Minimum thrust cannot exceed maximum thrust")
        if len(self.spin_directions) != 4 or any(s not in (-1, 1, -1.0, 1.0) for s in self.spin_directions):
            raise ValueError("spin_directions must contain four +1/-1 values")


@dataclass(frozen=True)
class FlightControlConfig:
    target_altitude_m: float = 0.80
    kp_altitude: float = 8.0       
    kd_altitude: float = 5.0       
    kp_roll_pitch_torque: float = 0.80  
    kd_roll_pitch_rate: float = 0.12    
    kp_yaw_torque: float = 0.25         
    kd_yaw_rate_torque: float = 0.08    


@dataclass
class SimulationResult:
    time: Array
    state: Array             # N x 13, same state order as 6-DOF plant
    mode: list[str]
    rotor_thrusts: Array     # N x 4, N per rotor
    total_rotor_thrust: Array
    transition_reason: list[str]
    saturated: Array         # N x 4 boolean

    @property
    def position_i(self): return self.state[:, :3]
    @property
    def velocity_b(self): return self.state[:, 3:6]
    @property
    def quaternion_ib(self): return self.state[:, 6:10]
    @property
    def euler_zyx(self): return np.vstack([quat_to_euler_zyx(q) for q in self.quaternion_ib])
    @property
    def omega_b(self): return self.state[:, 10:13]


def transition_gate(state: Array, *, apex_detected: bool, sensor_valid: bool = True,
                    cfg: TransitionConfig = TransitionConfig()) -> tuple[bool, str]:
    """Return (accepted, reason) from current simulated state and sensor flag."""
    s = np.asarray(state, dtype=float).reshape(13)
    if not sensor_valid or not np.all(np.isfinite(s)):
        return False, "sensor/state invalid"
    euler = quat_to_euler_zyx(s[6:10])
    tilt = float(np.rad2deg(np.hypot(euler[0], euler[1])))
    rate = float(np.rad2deg(np.linalg.norm(s[10:13])))
    R = quat_to_rotation_matrix(s[6:10])
    vz_i = float((R @ s[3:6])[2])
    if cfg.require_apex and not apex_detected:
        return False, "apex not detected"
    if abs(vz_i) > cfg.vertical_speed_threshold:
        return False, f"vertical speed too high ({vz_i:.3f} m/s)"
    if tilt > cfg.max_tilt_deg:
        return False, f"tilt too high ({tilt:.2f} deg)"
    if rate > cfg.max_rate_deg_s:
        return False, f"angular rate too high ({rate:.2f} deg/s)"
    return True, "all transition gates satisfied"


def rotor_allocation_matrix(cfg: RotorConfig) -> Array:
    """Map individual rotor thrusts to [total thrust, roll, pitch, yaw] wrench."""
    a = cfg.arm / np.sqrt(2.0)
    # Rotor order: front-right, rear-right, rear-left, front-left in body x/y.
    xy = np.array([[a, -a], [-a, -a], [-a, a], [a, a]], dtype=float)
    spin = np.asarray(cfg.spin_directions, dtype=float)
    A = np.zeros((4, 4), dtype=float)
    A[0, :] = 1.0
    A[1, :] = xy[:, 1]            
    A[2, :] = -xy[:, 0]           
    A[3, :] = spin * cfg.yaw_moment_per_newton
    return A


def allocate_rotors(wrench: Array, cfg: RotorConfig) -> tuple[Array, Array]:
    """Least-squares allocation with physical thrust clipping; returns thrusts and saturation flags."""
    wrench = np.asarray(wrench, dtype=float).reshape(4)
    A = rotor_allocation_matrix(cfg)
    raw = np.linalg.solve(A, wrench) if abs(np.linalg.det(A)) > 1e-12 else np.linalg.lstsq(A, wrench, rcond=None)[0]
    clipped = np.clip(raw, cfg.min_thrust_per_rotor, cfg.max_thrust_per_rotor)
    return clipped, np.abs(clipped - raw) > 1e-10


def flight_controller(state: Array, mass: float, rotor_cfg: RotorConfig,
                      ctrl: FlightControlConfig, yaw_target: float = 0.0) -> tuple[Array, Array]:
    """Altitude PD + attitude/rate PD. Returns rotor thrusts and saturation flags."""
    s = np.asarray(state, dtype=float).reshape(13)
    z = float(s[2])
    R = quat_to_rotation_matrix(s[6:10])
    vel_i = R @ s[3:6]
    euler = quat_to_euler_zyx(s[6:10])
    p, q, r = s[10:13]

    
    az_cmd = ctrl.kp_altitude * (ctrl.target_altitude_m - z) - ctrl.kd_altitude * vel_i[2]
    total_thrust = mass * (GRAVITY + az_cmd) / max(R[2, 2], 0.25)
    total_thrust = float(np.clip(total_thrust, 0.0, 4.0 * rotor_cfg.max_thrust_per_rotor))

    
    roll_error = float(euler[0])
    pitch_error = float(euler[1])
    tau_x = -ctrl.kp_roll_pitch_torque * roll_error - ctrl.kd_roll_pitch_rate * p
    tau_y = -ctrl.kp_roll_pitch_torque * pitch_error - ctrl.kd_roll_pitch_rate * q
    yaw_error = (yaw_target - euler[2] + np.pi) % (2*np.pi) - np.pi
    tau_z = ctrl.kp_yaw_torque * yaw_error - ctrl.kd_yaw_rate_torque * r
    wrench = np.array([total_thrust, tau_x, tau_y, tau_z])
    return allocate_rotors(wrench, rotor_cfg)


def simulate_transition_and_flight(*, launch_cfg: LaunchConfig = LaunchConfig(),
                                   transition_cfg: TransitionConfig = TransitionConfig(),
                                   rotor_cfg: RotorConfig = RotorConfig(),
                                   control_cfg: FlightControlConfig = FlightControlConfig(),
                                   duration: float = 2.0, dt: float = 0.0005,
                                   initial_position_i=(0.0, 0.0, 0.10),
                                   initial_velocity_b=(0.0, 0.0, 0.0),
                                   initial_quaternion_ib=(1.0, 0.0, 0.0, 0.0),
                                   initial_omega_b=(0.0, 0.0, 0.0)) -> SimulationResult:
    """Run launch, apex-gated transition, then closed-loop rotor flight simulation."""
    if duration <= 0 or dt <= 0:
        raise ValueError("duration and dt must be positive")
    mass = launch_cfg.mass
    inertia = validate_inertia(launch_cfg.inertia_b)
    s = pack_state(initial_position_i, initial_velocity_b, initial_quaternion_ib, initial_omega_b)
    n = int(np.ceil(duration / dt))
    time = np.linspace(0.0, duration, n+1)
    states = np.zeros((n+1, 13)); states[0] = s
    modes = ["LAUNCH"] * (n+1)
    rotor_hist = np.zeros((n+1, 4)); total_hist = np.zeros(n+1)
    saturated_hist = np.zeros((n+1, 4), dtype=bool)
    reasons = ["waiting for apex"] * (n+1)
    mode = "LAUNCH"
    apex_detected = False
    previous_vz = float((quat_to_rotation_matrix(s[6:10]) @ s[3:6])[2])
    transition_time = None

    def body_rates_and_force(t, state):
        if mode == "LAUNCH":
            f, m, *_ = evaluate_forces(t, state, launch_cfg)
            return f, m
        if mode == "ABORT":
            # Explicitly inhibit rotor output on abort. The earlier prototype's
            # fallback branch accidentally kept calling the flight controller.
            return np.zeros(3), np.zeros(3)
        thrusts, _ = flight_controller(state, mass, rotor_cfg, control_cfg)
        wrench = rotor_allocation_matrix(rotor_cfg) @ thrusts
        # Rotors provide body +z thrust and reaction moments about body axes.
        return np.array([0.0, 0.0, wrench[0]]), wrench[1:4]

    for k in range(n):
        t = time[k]
        R = quat_to_rotation_matrix(s[6:10])
        vz = float((R @ s[3:6])[2])
        if previous_vz > 0.0 and vz <= 0.0:
            apex_detected = True
        if mode == "LAUNCH":
            accepted, reason = transition_gate(s, apex_detected=apex_detected, cfg=transition_cfg)
            if accepted:
                mode = "FLIGHT"
                transition_time = t
                reasons[k] = reason
            elif t > transition_cfg.apex_timeout_s:
                mode = "ABORT"
                reasons[k] = "transition timeout: " + reason
            else:
                reasons[k] = reason
        if mode == "FLIGHT":
            thrusts, sat = flight_controller(s, mass, rotor_cfg, control_cfg)
            rotor_hist[k] = thrusts
            saturated_hist[k] = sat
            total_hist[k] = np.sum(thrusts)
        elif mode == "ABORT":
            rotor_hist[k] = 0.0
            total_hist[k] = 0.0
        else:
            rotor_hist[k] = 0.0
            total_hist[k] = 0.0

        force_fn = lambda tt, ss: body_rates_and_force(tt, ss)[0]
        moment_fn = lambda tt, ss: body_rates_and_force(tt, ss)[1]
        s = rk4_step(lambda tt, ss: plant_rhs(tt, ss, force_fn, moment_fn, mass, inertia, launch_cfg.gravity), t, s, time[k+1]-time[k])
        states[k+1] = s
        modes[k] = mode
        modes[k+1] = mode
        previous_vz = vz
        if mode == "FLIGHT":
            reasons[k+1] = "flight controller active"
        elif mode == "ABORT":
            reasons[k+1] = "rotors inhibited after transition timeout"
        else:
            reasons[k+1] = reasons[k]

    # Final-sample diagnostics.
    if modes[-1] == "FLIGHT":
        last_t, sat = flight_controller(states[-1], mass, rotor_cfg, control_cfg)
        rotor_hist[-1] = last_t
        saturated_hist[-1] = sat
        total_hist[-1] = np.sum(last_t)
    return SimulationResult(time, states, modes, rotor_hist, total_hist, reasons, saturated_hist)


def run_self_tests() -> None:
    # 0) CAD-derived parameters are present and the inertia tensor is physically valid.
    cad_cfg = LaunchConfig()
    assert abs(cad_cfg.mass - 2.465) < 1e-12
    assert np.allclose(np.diag(cad_cfg.inertia_b), [0.021, 0.019, 0.020], atol=1e-12)
    assert np.all(np.linalg.eigvalsh(validate_inertia(cad_cfg.inertia_b)) > 0.0)

    # 1) Gate rejects invalid sensors and a high angular rate.
    s0 = pack_state([0, 0, 0.8], [0, 0, 0], [1, 0, 0, 0], [0, 0, 0])
    assert transition_gate(s0, apex_detected=True, sensor_valid=False)[0] is False
    high_rate = pack_state([0, 0, 0.8], [0, 0, 0], [1, 0, 0, 0], [1.0, 0, 0])
    assert not transition_gate(high_rate, apex_detected=True)[0]

    # 2) Gate accepts a level, low-rate state at apex.
    ok, _ = transition_gate(s0, apex_detected=True)
    assert ok

    # 3) Equal thrust allocation produces zero net body moments for symmetric quad layout.
    rc = RotorConfig()
    A = rotor_allocation_matrix(rc)
    w = A @ np.ones(4) * 3.0
    assert np.allclose(w, [12.0, 0.0, 0.0, 0.0], atol=1e-12)

    # 4) Allocator respects individual thrust limits.
    thr, sat = allocate_rotors([100.0, 10.0, -10.0, 1.0], rc)
    assert np.all(thr >= rc.min_thrust_per_rotor) and np.all(thr <= rc.max_thrust_per_rotor)
    assert np.any(sat)

    # 5) Integrated short simulation remains finite and keeps quaternion normalized.
    lc = LaunchConfig(launch_force_total=0.0, launch_duration=0.0, contact_enabled=False)
    out = simulate_transition_and_flight(launch_cfg=lc, duration=0.05, dt=0.001,
                                         initial_position_i=(0, 0, 2.0))
    assert np.all(np.isfinite(out.state))
    assert np.allclose(np.linalg.norm(out.state[:, 6:10], axis=1), 1.0, atol=1e-9)
    # 6) A transition timeout must enter ABORT and inhibit all rotor thrust.
    timeout_cfg = TransitionConfig(apex_timeout_s=0.03, require_apex=True)
    timeout_launch = LaunchConfig(launch_force_total=0.0, launch_duration=0.0, contact_enabled=False)
    aborted = simulate_transition_and_flight(launch_cfg=timeout_launch, transition_cfg=timeout_cfg,
        duration=0.08, dt=0.001, initial_position_i=(0.0, 0.0, 2.0),
        initial_omega_b=(np.deg2rad(18.0), 0.0, 0.0))
    assert "ABORT" in aborted.mode
    assert np.allclose(aborted.rotor_thrusts[np.asarray(aborted.mode) == "ABORT"], 0.0)

    # 7) Closed-loop attitude recovery smoke test from a small initial pitch.
    recover_cfg = TransitionConfig(apex_timeout_s=0.5, require_apex=False)
    recover_launch = LaunchConfig(launch_force_total=0.0, launch_duration=0.0, contact_enabled=False)
    q_pitch = np.array([np.cos(np.deg2rad(3.0)/2), 0.0, np.sin(np.deg2rad(3.0)/2), 0.0])
    recovered = simulate_transition_and_flight(launch_cfg=recover_launch, transition_cfg=recover_cfg,
        duration=1.0, dt=0.001, initial_position_i=(0.0, 0.0, 0.8),
        initial_quaternion_ib=q_pitch, initial_omega_b=(0.0, np.deg2rad(4.0), 0.0))
    assert "FLIGHT" in recovered.mode
    assert np.all(np.isfinite(recovered.state))
    assert np.allclose(np.linalg.norm(recovered.state[:, 6:10], axis=1), 1.0, atol=1e-9)

    print("KITE-X TRANSITION + FLIGHT CONTROL SELF-TESTS: PASS")
    print("  CAD mass/inertia configuration and positive-definite tensor: PASS")
    print("  Invalid-sensor/high-rate gate rejection: PASS")
    print("  Safe-state gate acceptance: PASS")
    print("  Symmetric rotor allocation: PASS")
    print("  Rotor thrust saturation limits: PASS")
    print("  Integrated simulation finite-state/quaternion check: PASS")
    print("  Transition timeout enters ABORT with rotors inhibited: PASS")
    print("  Closed-loop attitude recovery smoke test: PASS")


def main() -> None:
    run_self_tests()
    result = simulate_transition_and_flight(duration=1.8, dt=0.0005)
    modes = np.asarray(result.mode)
    flight_samples = int(np.sum(modes == "FLIGHT"))
    abort_samples = int(np.sum(modes == "ABORT"))
    euler_deg = np.rad2deg(result.euler_zyx)
    print("\nKITE-X TRANSITION + FLIGHT DEMO (CAD mass/inertia; other parameters provisional)")
    print(f"  final height: {result.position_i[-1, 2]:.3f} m")
    print(f"  max absolute roll: {np.max(np.abs(euler_deg[:, 0])):.2f} deg")
    print(f"  max absolute pitch: {np.max(np.abs(euler_deg[:, 1])):.2f} deg")
    print(f"  flight-mode samples: {flight_samples}")
    print(f"  abort-mode samples: {abort_samples}")
    print(f"  max rotor total thrust: {np.max(result.total_rotor_thrust):.2f} N")
    print("CAD mass used: 2.465 kg. CAD inertia diagonals: 0.021, 0.019, 0.020 kg*m^2.")
    print("NOTE: Verify CAD selection/reference frame/off-diagonal signs. Rotor geometry, thrust/torque maps, launch mechanics and control gains remain provisional. Simulation only.")


if __name__ == "__main__":
    run_self_tests()


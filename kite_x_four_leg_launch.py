from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import numpy as np

from kite_x_6dof_plant import (
    GRAVITY, PROVISIONAL_INERTIA, PlantResult, pack_state,
    quat_to_rotation_matrix, simulate_6dof, validate_inertia,
)

Array = np.ndarray


@dataclass(frozen=True)
class LaunchConfig:
    mass: float = 2.465
    inertia_b: Array = None  # defaults to CAD-reported tensor; see plant module caveat
    gravity: float = GRAVITY
    leg_positions_b: Array = None
    launch_force_total: float = 900.0
    launch_start: float = 0.0
    launch_duration: float = 0.0125
    leg_force_scales: Array = None
    leg_delays: Array = None
    ground_stiffness: float = 5.0e5
    ground_damping: float = 5.0e4
    contact_enabled: bool = True

    def __post_init__(self):
        if self.inertia_b is None:
            object.__setattr__(self, "inertia_b", PROVISIONAL_INERTIA.copy())
        if self.leg_positions_b is None:
            # Placeholder square footprint, 0.20 m half-span. Replace with CAD values.
            object.__setattr__(self, "leg_positions_b", np.array([
                [ 0.20,  0.20, -0.10],
                [ 0.20, -0.20, -0.10],
                [-0.20,  0.20, -0.10],
                [-0.20, -0.20, -0.10],
            ], dtype=float))
        I = validate_inertia(self.inertia_b)
        legs = np.asarray(self.leg_positions_b, dtype=float)
        scales = np.ones(4) if self.leg_force_scales is None else np.asarray(self.leg_force_scales, dtype=float)
        delays = np.zeros(4) if self.leg_delays is None else np.asarray(self.leg_delays, dtype=float)
        if self.mass <= 0 or self.gravity <= 0:
            raise ValueError("mass and gravity must be positive")
        if legs.shape != (4, 3) or not np.all(np.isfinite(legs)):
            raise ValueError("leg_positions_b must be a finite 4x3 array")
        if scales.shape != (4,) or not np.all(np.isfinite(scales)) or np.any(scales < 0):
            raise ValueError("leg_force_scales must contain four finite nonnegative values")
        if delays.shape != (4,) or not np.all(np.isfinite(delays)) or np.any(delays < 0):
            raise ValueError("leg_delays must contain four finite nonnegative values")
        if self.launch_force_total < 0 or self.launch_duration < 0:
            raise ValueError("launch force and duration cannot be negative")
        if self.ground_stiffness < 0 or self.ground_damping < 0:
            raise ValueError("ground stiffness and damping cannot be negative")
        object.__setattr__(self, "inertia_b", I)
        object.__setattr__(self, "leg_positions_b", legs)
        object.__setattr__(self, "leg_force_scales", scales)
        object.__setattr__(self, "leg_delays", delays)


@dataclass
class LaunchResult:
    plant: PlantResult
    launch_force_b: Array       # N x 3, launch actuator resultant in body axes
    contact_force_i: Array      # N x 3, total ground reaction in inertial axes
    contact_per_leg_i: Array   # N x 4 x 3, per-foot ground reaction
    moment_total_b: Array       # N x 3, resultant moment about CG in body axes
    penetration_m: Array        # N x 4, positive downward penetration


def pulse_active(t: float, cfg: LaunchConfig) -> bool:
    return cfg.launch_start <= t < cfg.launch_start + cfg.launch_duration


def leg_launch_forces_b(t: float, cfg: LaunchConfig) -> Array:
    """Equal-share prescribed actuator force on body, one vector per leg."""
    forces = np.zeros((4, 3))
    for i in range(4):
        leg_start = cfg.launch_start + cfg.leg_delays[i]
        if leg_start <= t < leg_start + cfg.launch_duration:
            # Equal nominal share, modified by each leg's scale and timing delay.
            # Upward is body +z only when the body is level.
            forces[i, 2] = (cfg.launch_force_total / 4.0) * cfg.leg_force_scales[i]
    return forces


def evaluate_forces(t: float, state: Array, cfg: LaunchConfig):
    """Return net body force, net body moment, and contact diagnostics."""
    s = np.asarray(state, dtype=float).reshape(13)
    pos_i, vel_b, q_ib, omega_b = s[:3], s[3:6], s[6:10], s[10:13]
    R_ib = quat_to_rotation_matrix(q_ib)
    legs_b = cfg.leg_positions_b

    actuator_forces_b = leg_launch_forces_b(t, cfg)
    force_launch_b = actuator_forces_b.sum(axis=0)
    moment_launch_b = np.sum(np.cross(legs_b, actuator_forces_b), axis=0)

    contacts_i = np.zeros((4, 3))
    penetration = np.zeros(4)
    if cfg.contact_enabled:
        for j, r_b in enumerate(legs_b):
            foot_i = pos_i + R_ib @ r_b
            foot_vel_b = vel_b + np.cross(omega_b, r_b)
            foot_vel_i = R_ib @ foot_vel_b
            delta = max(0.0, -foot_i[2])
            penetration[j] = delta
            if delta > 0.0:
                # Positive upward reaction. Damping acts only while the foot moves downward.
                normal = cfg.ground_stiffness * delta + cfg.ground_damping * max(0.0, -foot_vel_i[2])
                contacts_i[j, 2] = max(0.0, normal)

    force_contact_i = contacts_i.sum(axis=0)
    force_contact_b = R_ib.T @ force_contact_i
    moment_contact_b = np.zeros(3)
    for j, r_b in enumerate(legs_b):
        force_contact_b_j = R_ib.T @ contacts_i[j]
        moment_contact_b += np.cross(r_b, force_contact_b_j)

    force_net_b = force_launch_b + force_contact_b
    moment_net_b = moment_launch_b + moment_contact_b
    return force_net_b, moment_net_b, force_launch_b, force_contact_i, contacts_i, penetration


def simulate_launch(cfg: LaunchConfig = LaunchConfig(), *, duration: float = 0.8,
                    dt: float = 0.0002, initial_position_i=(0.0, 0.0, 0.10),
                    initial_velocity_b=(0.0, 0.0, 0.0), initial_quaternion_ib=(1.0, 0.0, 0.0, 0.0),
                    initial_omega_b=(0.0, 0.0, 0.0)) -> LaunchResult:
    """Simulate four-leg launch and ground contact through the 6-DOF plant."""
    s0 = pack_state(initial_position_i, initial_velocity_b, initial_quaternion_ib, initial_omega_b)

    def force_fn(t, s):
        return evaluate_forces(t, s, cfg)[0]

    def moment_fn(t, s):
        return evaluate_forces(t, s, cfg)[1]

    plant = simulate_6dof(cfg.mass, cfg.inertia_b, s0, force_fn, moment_fn,
                          duration=duration, dt=dt, gravity=cfg.gravity)
    n = len(plant.time)
    launch = np.zeros((n, 3)); contact_i = np.zeros((n, 3))
    contact_each = np.zeros((n, 4, 3)); moments = np.zeros((n, 3)); penetration = np.zeros((n, 4))
    for k, (t, p, v, q, w) in enumerate(zip(plant.time, plant.position_i, plant.velocity_b,
                                             plant.quaternion_ib, plant.omega_b)):
        s = pack_state(p, v, q, w)
        _, moments[k], launch[k], contact_i[k], contact_each[k], penetration[k] = evaluate_forces(t, s, cfg)
    return LaunchResult(plant, launch, contact_i, contact_each, moments, penetration)


def run_self_tests():
    #IMPORTANT
    # Test 1: symmetric four-leg geometry and equal force produce no launch moment.
    cfg = LaunchConfig(launch_force_total=900.0, launch_duration=0.02,
                       contact_enabled=False)
    s = pack_state([0, 0, 1], [0, 0, 0], [1, 0, 0, 0], [0, 0, 0])
    f, m, *_ = evaluate_forces(0.005, s, cfg)
    assert np.allclose(f, [0, 0, 900], atol=1e-12)
    assert np.allclose(m, [0, 0, 0], atol=1e-12)

    # Test 2: launch pulse is off at and after its end time.
    assert np.allclose(leg_launch_forces_b(0.02, cfg), 0.0)

    # Test 3: asymmetric leg force creates the expected nonzero body moment.
    asymmetric_cfg = LaunchConfig(launch_force_total=900.0, launch_duration=0.02,
                                  leg_force_scales=[0.0, 0.0, 0.0, 1.0], contact_enabled=False)
    _, asym_m, asym_force, *_ = evaluate_forces(0.005, s, asymmetric_cfg)
    expected = np.cross(asymmetric_cfg.leg_positions_b[3], np.array([0.0, 0.0, 225.0]))
    assert np.allclose(asym_force, [0.0, 0.0, 225.0])
    assert np.allclose(asym_m, expected)
    assert np.linalg.norm(asym_m[:2]) > 0.0

    # Test 4: delayed leg pulse starts at its own configured time.
    delayed_cfg = LaunchConfig(launch_force_total=900.0, launch_duration=0.02,
                               leg_delays=[0.0, 0.0, 0.003, 0.003], contact_enabled=False)
    delayed_forces = leg_launch_forces_b(0.001, delayed_cfg)
    assert delayed_forces[0, 2] > 0 and delayed_forces[2, 2] == 0

    # Test 5: penetrated foot at rest gets an upward contact force.

    contact_cfg = LaunchConfig(launch_force_total=0.0, contact_enabled=True,
                               ground_stiffness=1000.0, ground_damping=0.0)
    sc = pack_state([0, 0, 0.05], [0, 0, 0], [1, 0, 0, 0], [0, 0, 0])
    *_, contacts, penetration = evaluate_forces(0.0, sc, contact_cfg)
    assert np.all(contacts[:, 2] >= 0.0)
    assert np.allclose(contacts[:, :2], 0.0)
    assert np.allclose(penetration, 0.05)  # feet are at z=-0.05, penetration 0.05 m

    # Test 6: short symmetric launch integrates without NaNs and keeps quaternion normalized.
    short_cfg = LaunchConfig(launch_force_total=900.0, launch_duration=0.0125,
                             contact_enabled=False)
    out = simulate_launch(short_cfg, duration=0.05, dt=0.0005, initial_position_i=(0, 0, 1.0))
    assert np.all(np.isfinite(out.plant.position_i))
    assert np.allclose(np.linalg.norm(out.plant.quaternion_ib, axis=1), 1.0, atol=1e-10)
    print("KITE-X FOUR-LEG LAUNCH SELF-TESTS: PASS")
    print("  Symmetric launch force/moment balance: PASS")
    print("  Pulse timing boundary: PASS")
    print("  Asymmetric leg force produces body moment: PASS")
    print("  Per-leg timing delay: PASS")
    print("  Unilateral ground contact response: PASS")
    print("  Coupled 6-DOF numerical integration: PASS")


def main():
    run_self_tests()
    cfg = LaunchConfig()
    # Start at z=0.10 m with feet extending 0.10 m down to ground.
    result = simulate_launch(cfg, duration=0.8, dt=0.0002, initial_position_i=(0, 0, 0.10))
    z = result.plant.position_i[:, 2]
    vz_i = np.array([
        (quat_to_rotation_matrix(q) @ v)[2]
        for q, v in zip(result.plant.quaternion_ib, result.plant.velocity_b)
    ])
    apex_idx = np.where((vz_i[:-1] > 0.0) & (vz_i[1:] <= 0.0))[0]
    print("\nKITE-X FOUR-LEG LAUNCH DEMO")
    print(f"  mass: {cfg.mass:.3f} kg (CAD-reported; confirm selected components)")
    print(f"  prescribed launch force: {cfg.launch_force_total:.1f} N total")
    print(f"  launch pulse: {cfg.launch_duration*1000:.2f} ms")
    print(f"  maximum body height: {np.max(z):.4f} m")
    print(f"  maximum foot penetration: {np.max(result.penetration_m):.6f} m")
    print(f"  maximum absolute roll: {np.rad2deg(np.max(np.abs(result.plant.euler_zyx[:,0]))):.4f} deg")
    print(f"  maximum absolute pitch: {np.rad2deg(np.max(np.abs(result.plant.euler_zyx[:,1]))):.4f} deg")
    print(f"  maximum absolute yaw: {np.rad2deg(np.max(np.abs(result.plant.euler_zyx[:,2]))):.4f} deg")
    print(f"  apex crossing detected: {'yes' if len(apex_idx) else 'no'}")
    print("NOTE: demo parameters and leg footprint are placeholders; this is not hardware validation.")

if __name__ == '__main__':
    main()

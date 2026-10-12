from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import numpy as np

Array = np.ndarray
GRAVITY = 9.81

CAD_REPORTED_INERTIA = np.array([
    [0.021000,  0.0001989,  0.0001222],
    [0.0001989, 0.019000,  -0.0001897],
    [0.0001222, -0.0001897, 0.020000],
], dtype=float)


PROVISIONAL_INERTIA = CAD_REPORTED_INERTIA.copy()


def validate_inertia(inertia: Array) -> Array:
    I = np.asarray(inertia, dtype=float)
    if I.shape != (3, 3) or not np.all(np.isfinite(I)):
        raise ValueError("Inertia must be a finite 3x3 matrix.")
    if not np.allclose(I, I.T, atol=1e-10, rtol=0.0):
        raise ValueError("Inertia tensor must be symmetric.")
    if np.any(np.linalg.eigvalsh(I) <= 0):
        raise ValueError("Inertia tensor must be positive definite.")
    return I


def quat_normalize(q: Array) -> Array:
    q = np.asarray(q, dtype=float).reshape(4)
    n = np.linalg.norm(q)
    if not np.isfinite(n) or n < 1e-14:
        raise ValueError("Quaternion must be finite and non-zero.")
    return q / n


def quat_multiply(a: Array, b: Array) -> Array:
    """Hamilton product for [w,x,y,z] quaternions."""
    aw, ax, ay, az = np.asarray(a, dtype=float).reshape(4)
    bw, bx, by, bz = np.asarray(b, dtype=float).reshape(4)
    return np.array([
        aw*bw - ax*bx - ay*by - az*bz,
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
    ])


def quat_to_rotation_matrix(q_ib: Array) -> Array:
    """Return R_IB, mapping body vectors to inertial coordinates."""
    w, x, y, z = quat_normalize(q_ib)
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ])


def euler_zyx_to_quat(roll: float, pitch: float, yaw: float) -> Array:
    """Convert ZYX Euler angles (rad) to body-to-inertial quaternion."""
    cr, sr = np.cos(roll/2), np.sin(roll/2)
    cp, sp = np.cos(pitch/2), np.sin(pitch/2)
    cy, sy = np.cos(yaw/2), np.sin(yaw/2)
    return quat_normalize(np.array([
        cy*cp*cr + sy*sp*sr,
        cy*cp*sr - sy*sp*cr,
        cy*sp*cr + sy*cp*sr,
        sy*cp*cr - cy*sp*sr,
    ]))


def quat_to_euler_zyx(q: Array) -> Array:
    """Return [roll,pitch,yaw] in radians, using ZYX convention."""
    R = quat_to_rotation_matrix(q)
    roll = np.arctan2(R[2, 1], R[2, 2])
    pitch = np.arcsin(np.clip(-R[2, 0], -1.0, 1.0))
    yaw = np.arctan2(R[1, 0], R[0, 0])
    return np.array([roll, pitch, yaw])


def angular_acceleration(omega_b: Array, moment_b: Array, inertia_b: Array) -> Array:
    I = validate_inertia(inertia_b)
    omega = np.asarray(omega_b, dtype=float).reshape(3)
    moment = np.asarray(moment_b, dtype=float).reshape(3)
    return np.linalg.solve(I, moment - np.cross(omega, I @ omega))


@dataclass
class PlantResult:
    time: Array
    position_i: Array       # N x 3, m
    velocity_b: Array       # N x 3, m/s
    quaternion_ib: Array    # N x 4, [w,x,y,z]
    euler_zyx: Array        # N x 3, rad
    omega_b: Array          # N x 3, rad/s
    force_b: Array          # N x 3, N
    moment_b: Array         # N x 3, N*m


def pack_state(position_i: Array, velocity_b: Array, quaternion_ib: Array, omega_b: Array) -> Array:
    return np.concatenate([
        np.asarray(position_i, dtype=float).reshape(3),
        np.asarray(velocity_b, dtype=float).reshape(3),
        quat_normalize(quaternion_ib),
        np.asarray(omega_b, dtype=float).reshape(3),
    ])


def plant_rhs(_t: float, state: Array, force_fn: Callable, moment_fn: Callable,
              mass: float, inertia_b: Array, gravity: float = GRAVITY) -> Array:
    """Derivative for the 13-state 6-DOF rigid body."""
    if mass <= 0 or not np.isfinite(mass):
        raise ValueError("Mass must be finite and positive.")
    s = np.asarray(state, dtype=float).reshape(13)
    pos_i, vel_b, q_ib, omega_b = s[:3], s[3:6], quat_normalize(s[6:10]), s[10:13]
    force_b = np.asarray(force_fn(_t, s), dtype=float).reshape(3)
    moment_b = np.asarray(moment_fn(_t, s), dtype=float).reshape(3)
    R_ib = quat_to_rotation_matrix(q_ib)
    gravity_i = np.array([0.0, 0.0, -gravity])

    pos_dot_i = R_ib @ vel_b
    # Newton's law expressed in a rotating body frame.
    vel_dot_b = force_b / mass + R_ib.T @ gravity_i - np.cross(omega_b, vel_b)
    # q_IB_dot = 1/2 * q_IB (*) [0, omega_B]
    q_dot = 0.5 * quat_multiply(q_ib, np.array([0.0, *omega_b]))
    omega_dot_b = angular_acceleration(omega_b, moment_b, inertia_b)
    return np.concatenate([pos_dot_i, vel_dot_b, q_dot, omega_dot_b])


def rk4_step(rhs: Callable, t: float, state: Array, h: float) -> Array:
    k1 = rhs(t, state)
    k2 = rhs(t+h/2, state+h*k1/2)
    k3 = rhs(t+h/2, state+h*k2/2)
    k4 = rhs(t+h, state+h*k3)
    out = state + h*(k1+2*k2+2*k3+k4)/6
    out[6:10] = quat_normalize(out[6:10])
    return out


def simulate_6dof(mass: float, inertia_b: Array, initial_state: Array,
                  force_fn: Callable, moment_fn: Callable, *,
                  duration: float = 2.0, dt: float = 0.001,
                  gravity: float = GRAVITY) -> PlantResult:
    """Integrate the 6-DOF plant. Inputs are force/moment callbacks in body axes."""
    I = validate_inertia(inertia_b)
    if mass <= 0 or duration <= 0 or dt <= 0:
        raise ValueError("mass, duration and dt must be positive.")
    initial = np.asarray(initial_state, dtype=float).reshape(13).copy()
    initial[6:10] = quat_normalize(initial[6:10])
    n = int(np.ceil(duration/dt))
    time = np.linspace(0.0, duration, n+1)
    states = np.zeros((n+1, 13))
    forces = np.zeros((n+1, 3)); moments = np.zeros((n+1, 3))
    states[0] = initial
    rhs = lambda t, s: plant_rhs(t, s, force_fn, moment_fn, mass, I, gravity)
    for k in range(n):
        h = time[k+1]-time[k]
        forces[k] = np.asarray(force_fn(time[k], states[k]), dtype=float).reshape(3)
        moments[k] = np.asarray(moment_fn(time[k], states[k]), dtype=float).reshape(3)
        states[k+1] = rk4_step(rhs, time[k], states[k], h)
    forces[-1] = np.asarray(force_fn(time[-1], states[-1]), dtype=float).reshape(3)
    moments[-1] = np.asarray(moment_fn(time[-1], states[-1]), dtype=float).reshape(3)
    euler = np.vstack([quat_to_euler_zyx(q) for q in states[:, 6:10]])
    return PlantResult(time, states[:, :3], states[:, 3:6], states[:, 6:10], euler,
                       states[:, 10:13], forces, moments)


def run_self_tests() -> None:
    I = validate_inertia(PROVISIONAL_INERTIA)
    ident = np.array([1.0, 0.0, 0.0, 0.0])
    zero3 = lambda _t, _s: np.zeros(3)
    # IMPORTANT
    # 1) Zero force and zero initial velocity: free fall matches z=-0.5*g*t^2.
    s0 = pack_state([0, 0, 0], [0, 0, 0], ident, [0, 0, 0])
    r = simulate_6dof(2.5, I, s0, zero3, zero3, duration=0.8, dt=0.001)
    assert abs(r.position_i[-1, 2] + 0.5*GRAVITY*0.8**2) < 2e-8
    assert np.max(np.abs(r.position_i[-1, :2])) < 1e-12

    # 2) Hover force equals mg along body +z when level: no vertical acceleration.
    hover_force = lambda _t, _s: np.array([0.0, 0.0, 2.5*GRAVITY])
    h = simulate_6dof(2.5, I, s0, hover_force, zero3, duration=0.5, dt=0.001)
    assert np.max(np.abs(h.position_i)) < 2e-8
    assert np.max(np.abs(h.velocity_b)) < 2e-8

    # 3) Constant yaw moment from rest gives correct initial angular acceleration.
    M = np.array([0.0, 0.0, 0.01])
    m_fn = lambda _t, _s: M
    d0 = plant_rhs(0.0, s0, zero3, m_fn, 2.5, I)
    assert np.allclose(d0[10:13], np.linalg.solve(I, M), rtol=1e-12, atol=1e-12)

    # 4) Constant body-x velocity at zero attitude advances inertial x.
    sx = pack_state([0, 0, 0], [1, 0, 0], ident, [0, 0, 0])
    xrun = simulate_6dof(2.5, I, sx, zero3, lambda _t, _s: np.zeros(3), duration=0.25, dt=0.001)
    assert abs(xrun.position_i[-1, 0] - 0.25) < 2e-8
    assert abs(xrun.position_i[-1, 2] + 0.5*GRAVITY*0.25**2) < 2e-8

    # 5) Quaternion remains normalized and Euler/quaternion conversion round-trips.
    angles = np.deg2rad([12.0, -7.0, 33.0])
    q = euler_zyx_to_quat(*angles)
    assert np.allclose(quat_to_euler_zyx(q), angles, atol=1e-12)
    assert abs(np.linalg.norm(r.quaternion_ib, axis=1).min()-1.0) < 1e-12
    assert abs(np.linalg.norm(r.quaternion_ib, axis=1).max()-1.0) < 1e-12

    print("KITE-X 6-DOF PLANT SELF-TESTS: PASS")
    print("  Free-fall analytic position check: PASS")
    print("  Level hover force-balance check: PASS")
    print("  Rotational equation coupling check: PASS")
    print("  Inertial/body translation mapping check: PASS")
    print("  Quaternion conversion/normalization check: PASS")


def main() -> None:
    run_self_tests()
    print("\n6-DOF STATE ORDER:")
    print("  [x_I,y_I,z_I, u_B,v_B,w_B, qw,qx,qy,qz, p,q,r]")
    print("Convention: inertial z-up; gravity is -z; quaternion maps body to inertial.")
    print("Force and moment callbacks currently represent applied net force/moment at the CG.")
    print("NOT YET INCLUDED: leg contact/springs, launch pulse, rotor actuator dynamics, aerodynamics, controller.")
    print("Inertia tensor uses the provided CAD panel values; verify full-vehicle selection, centre-of-mass frame, and off-diagonal convention before validation.")

if __name__ == '__main__':
    main()

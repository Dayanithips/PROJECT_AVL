from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import numpy as np

Array = np.ndarray


def validate_inertia(inertia: Array, *, symmetry_tol: float = 1e-10) -> Array:
    """Validate and return a 3x3 symmetric positive-definite inertia tensor."""
    I = np.asarray(inertia, dtype=float)
    if I.shape != (3, 3):
        raise ValueError(f"Inertia tensor must have shape (3, 3), got {I.shape}.")
    if not np.all(np.isfinite(I)):
        raise ValueError("Inertia tensor contains non-finite values.")
    if not np.allclose(I, I.T, atol=symmetry_tol, rtol=0.0):
        raise ValueError("Inertia tensor must be symmetric. Check CAD axis/sign convention.")
    eig = np.linalg.eigvalsh(I)
    if np.any(eig <= 0.0):
        raise ValueError(f"Inertia tensor must be positive definite; eigenvalues={eig}.")
    return I


def angular_acceleration(omega_body: Array, torque_body: Array, inertia_body: Array) -> Array:
    
    I = validate_inertia(inertia_body)
    omega = np.asarray(omega_body, dtype=float).reshape(3)
    torque = np.asarray(torque_body, dtype=float).reshape(3)
    if not np.all(np.isfinite(omega)) or not np.all(np.isfinite(torque)):
        raise ValueError("Angular velocity and torque must be finite.")
    # I * omega_dot = torque - omega x (I * omega)
    return np.linalg.solve(I, torque - np.cross(omega, I @ omega))


def rotational_rhs(_time: float, state: Array, torque_fn: Callable[[float, Array], Array], inertia: Array) -> Array:
    """State derivative for state=[p,q,r] in body coordinates."""
    omega = np.asarray(state, dtype=float).reshape(3)
    torque = np.asarray(torque_fn(_time, omega), dtype=float).reshape(3)
    return angular_acceleration(omega, torque, inertia)


def rk4_step(rhs: Callable[[float, Array], Array], time: float, state: Array, dt: float) -> Array:
    
    if dt <= 0:
        raise ValueError("dt must be positive.")
    x = np.asarray(state, dtype=float)
    k1 = rhs(time, x)
    k2 = rhs(time + dt / 2, x + dt * k1 / 2)
    k3 = rhs(time + dt / 2, x + dt * k2 / 2)
    k4 = rhs(time + dt, x + dt * k3)
    return x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


@dataclass
class RotationalResult:
    time: Array
    omega_body: Array  # N x 3, rad/s
    torque_body: Array  # N x 3, N*m
    angular_acceleration_body: Array  # N x 3, rad/s^2
    kinetic_energy: Array  # J
    angular_momentum_norm: Array  # kg*m^2/s


def simulate_rotation(
    inertia_body: Array,
    omega0_body: Array,
    torque_fn: Callable[[float, Array], Array],
    *, duration: float = 2.0, dt: float = 0.001,
) -> RotationalResult:
    """Integrate body rates with RK4 and calculate diagnostic quantities."""
    I = validate_inertia(inertia_body)
    omega0 = np.asarray(omega0_body, dtype=float).reshape(3)
    if duration <= 0 or dt <= 0:
        raise ValueError("duration and dt must be positive.")
    n = int(np.ceil(duration / dt))
    time = np.linspace(0.0, duration, n + 1)
    omega = np.zeros((n + 1, 3), dtype=float)
    torques = np.zeros_like(omega)
    alphas = np.zeros_like(omega)
    omega[0] = omega0

    def rhs(t: float, w: Array) -> Array:
        return rotational_rhs(t, w, torque_fn, I)

    for k in range(n):
        h = time[k + 1] - time[k]
        torques[k] = np.asarray(torque_fn(time[k], omega[k]), dtype=float).reshape(3)
        alphas[k] = angular_acceleration(omega[k], torques[k], I)
        omega[k + 1] = rk4_step(rhs, time[k], omega[k], h)
    torques[-1] = np.asarray(torque_fn(time[-1], omega[-1]), dtype=float).reshape(3)
    alphas[-1] = angular_acceleration(omega[-1], torques[-1], I)

    # For body-frame omega, these are instantaneous kinetic energy and the
    # magnitude of angular momentum expressed in body coordinates.
    energy = 0.5 * np.einsum("ni,ij,nj->n", omega, I, omega)
    momentum = omega @ I.T
    momentum_norm = np.linalg.norm(momentum, axis=1)
    return RotationalResult(time, omega, torques, alphas, energy, momentum_norm)


def run_self_tests() -> None:
    #IMPORTANT
    # Use a full tensor with non-zero products of inertia.
    I = np.array([
        [0.02013, 0.00019, 0.00012],
        [0.00019, 0.01854, -0.00018],
        [0.00012, -0.00018, 0.01956],
    ])
    validate_inertia(I)

    # Test 1: zero rates and zero torque must remain at rest.
    zero = simulate_rotation(I, [0, 0, 0], lambda _t, _w: np.zeros(3), duration=0.1, dt=0.001)
    assert np.max(np.abs(zero.omega_body)) < 1e-14

    # Test 2: with zero initial rate and constant torque, initial alpha=I^-1*tau.
    torque = np.array([0.01, -0.02, 0.015])
    alpha = angular_acceleration(np.zeros(3), torque, I)
    assert np.allclose(alpha, np.linalg.solve(I, torque), rtol=1e-12, atol=1e-12)

    # Test 3: torque-free motion conserves inertial kinetic energy and angular
    # momentum magnitude. Body components vary, but these invariants should not.
    free = simulate_rotation(I, [1.1, -0.7, 0.9], lambda _t, _w: np.zeros(3), duration=2.0, dt=0.0005)
    energy_rel_drift = np.ptp(free.kinetic_energy) / free.kinetic_energy[0]
    momentum_rel_drift = np.ptp(free.angular_momentum_norm) / free.angular_momentum_norm[0]
    assert energy_rel_drift < 2e-7, f"Energy drift too high: {energy_rel_drift:.3e}"
    assert momentum_rel_drift < 2e-7, f"Angular-momentum drift too high: {momentum_rel_drift:.3e}"

    # Test 4: the zero-rate angular acceleration equals I^-1 torque even with
    # products of inertia, guarding against diagonal-only simplifications.
    assert np.allclose(angular_acceleration([0, 0, 0], torque, I), np.linalg.solve(I, torque))

    print("ROTATIONAL DYNAMICS SELF-TESTS: PASS")
    print(f"  Torque-free relative kinetic-energy drift: {energy_rel_drift:.3e}")
    print(f"  Torque-free relative angular-momentum drift: {momentum_rel_drift:.3e}")


def main() -> None:
    run_self_tests()
    # Provisional example tensor from a previously discussed CAD estimate.
    # Replace with the final verified CAD inertia tensor before project claims.
    inertia = np.array([
        [0.02013, 0.00019, 0.00012],
        [0.00019, 0.01854, -0.00018],
        [0.00012, -0.00018, 0.01956],
    ])
    omega0 = np.deg2rad([8.0, -5.0, 12.0])
    result = simulate_rotation(
        inertia, omega0, lambda _t, _w: np.zeros(3), duration=2.0, dt=0.001
    )
    print("\nPROVISIONAL TORQUE-FREE DEMO")
    print("Initial body rates [deg/s]:", np.round(np.rad2deg(result.omega_body[0]), 4))
    print("Final body rates   [deg/s]:", np.round(np.rad2deg(result.omega_body[-1]), 4))
    print("Initial kinetic energy [J]:", f"{result.kinetic_energy[0]:.8f}")
    print("Final kinetic energy   [J]:", f"{result.kinetic_energy[-1]:.8f}")
    print("NOTE: this validates the rotational subsystem numerically, not the full KITE-X vehicle.")


if __name__ == "__main__":
    main()

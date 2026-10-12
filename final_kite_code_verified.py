#KITE-X apex detection and transition-intelligence simulation.
#Research prototype only; not hardware-ready flight-control software. 
import numpy as np
import matplotlib.pyplot as plt

m = 2.465  # CAD-reported mass; verify this represents the complete vehicle
F_launch = 900.0
c_leg = 100.0
dt_launch = 0.0125
g = 9.81
target_height = 0.6

eps_v = 0.15
eps_theta = np.deg2rad(5.0)
eps_omega = np.deg2rad(10.0)

dt = 0.0005
t_end = 0.8
t = np.arange(0.0, t_end + dt, dt)

tau = m / c_leg
F_net = F_launch - m * g

F = np.zeros_like(t)
z = np.zeros_like(t)
Vz = np.zeros_like(t)

idx_launch = t <= dt_launch
tL = t[idx_launch]

F[idx_launch] = F_launch

Vz[idx_launch] = (F_net / c_leg) * (
    1.0 - np.exp(-tL / tau)
)

z[idx_launch] = (F_net / c_leg) * (
    tL - tau * (1.0 - np.exp(-tL / tau))
)

launch_idx = np.where(idx_launch)[0][-1]

Vz_launch = Vz[launch_idx]
z_launch = z[launch_idx]

idx_ballistic = t > dt_launch
tb = t[idx_ballistic] - dt_launch

Vz[idx_ballistic] = Vz_launch - g * tb

z[idx_ballistic] = (
    z_launch
    + Vz_launch * tb
    - 0.5 * g * tb**2
)

ground_indices = np.where(
    (z <= 0.0) & (t > dt_launch)
)[0]

if ground_indices.size:
    ground_idx = ground_indices[0]
    z[ground_idx:] = 0.0
    Vz[ground_idx:] = 0.0
    t_sim_end = t[ground_idx]
else:
    t_sim_end = t[-1]

valid_idx = t <= t_sim_end

t = t[valid_idx]
F = F[valid_idx]
z = z[valid_idx]
Vz = Vz[valid_idx]

apex_crossings = np.where(
    (Vz[:-1] > 0.0) &
    (Vz[1:] <= 0.0)
)[0]

if apex_crossings.size:
    apex_idx = apex_crossings[0]

    t1 = t[apex_idx]
    t2 = t[apex_idx + 1]

    v1 = Vz[apex_idx]
    v2 = Vz[apex_idx + 1]

    t_apex_true = (
        t1
        + (-v1) * (t2 - t1) / (v2 - v1)
    )
else:
    apex_idx = int(np.argmin(np.abs(Vz)))
    t_apex_true = t[apex_idx]

z_apex_true = (
    z_launch
    + Vz_launch * (t_apex_true - dt_launch)
    - 0.5 * g * (t_apex_true - dt_launch)**2
)

# Synthetic attitude/rate traces for software testing only. These are NOT sensor data
# and must be replaced by signals from the coupled 6-DOF vehicle model.
theta0_test = np.deg2rad(1.5)
theta = theta0_test * np.exp(-t / 0.08)
omega = -(theta0_test / 0.08) * np.exp(-t / 0.08)

rng = np.random.default_rng(10)

sigma_z = 0.005
sigma_v = 0.08
sigma_theta = np.deg2rad(0.5)
sigma_omega = np.deg2rad(1.5)

z_measured = (
    z
    + sigma_z * rng.standard_normal(t.size)
)

v_measured = (
    Vz
    + sigma_v * rng.standard_normal(t.size)
)

filter_window = 11

z_est = np.convolve(
    z_measured,
    np.ones(filter_window) / filter_window,
    mode="same"
)

v_est = np.convolve(
    v_measured,
    np.ones(filter_window) / filter_window,
    mode="same"
)

search_start = int(
    np.searchsorted(t, dt_launch)
)

estimated_crossings = np.where(
    (v_est[search_start:-1] > 0.0) &
    (v_est[search_start + 1:] <= 0.0)
)[0]

if estimated_crossings.size:

    estimated_idx = (
        search_start
        + estimated_crossings[0]
    )

    t1 = t[estimated_idx]
    t2 = t[estimated_idx + 1]

    v1 = v_est[estimated_idx]
    v2 = v_est[estimated_idx + 1]

    if v2 != v1:
        t_apex_est = (
            t1
            + (-v1) * (t2 - t1) / (v2 - v1)
        )
    else:
        t_apex_est = t1

else:

    estimated_idx = (
        search_start
        + int(
            np.argmin(
                np.abs(v_est[search_start:])
            )
        )
    )

    t_apex_est = t[estimated_idx]

z_apex_est = np.interp(
    t_apex_est,
    t,
    z_est
)

v_apex_est = np.interp(
    t_apex_est,
    t,
    v_est
)

theta_apex_est = np.interp(
    t_apex_est,
    t,
    theta
)

omega_apex_est = np.interp(
    t_apex_est,
    t,
    omega
)

vz_check = (
    abs(v_apex_est) <= eps_v
)

theta_check = (
    abs(theta_apex_est) <= eps_theta
)

omega_check = (
    abs(omega_apex_est) <= eps_omega
)

transition_ok = (
    vz_check
    and theta_check
    and omega_check
)

rotor_command = np.zeros_like(t)
hybrid_state = np.zeros_like(t, dtype=int)

if transition_ok:

    rotor_command[t >= t_apex_est] = 1

    hybrid_state[t < dt_launch] = 0

    hybrid_state[
        (t >= dt_launch) &
        (t < t_apex_est)
    ] = 1

    hybrid_state[
        t >= t_apex_est
    ] = 2

else:

    hybrid_state[t < dt_launch] = 0

    hybrid_state[
        t >= dt_launch
    ] = 1

launch_impulse = (
    F_launch * dt_launch
)

launch_acceleration_initial = (
    F_net / m
)

Vz_launch_analytical = (
    F_net / c_leg
) * (
    1.0
    - np.exp(
        -c_leg * dt_launch / m
    )
)

z_launch_analytical = (
    F_net / c_leg
) * (
    dt_launch
    - (m / c_leg)
    * (
        1.0
        - np.exp(
            -c_leg * dt_launch / m
        )
    )
)

t_ballistic_apex = (
    Vz_launch_analytical / g
)

z_ballistic_gain = (
    Vz_launch_analytical**2
    / (2.0 * g)
)

z_apex_analytical = (
    z_launch_analytical
    + z_ballistic_gain
)

t_apex_analytical = (
    dt_launch
    + t_ballistic_apex
)

velocity_error = abs(
    Vz_launch
    - Vz_launch_analytical
)

height_error = abs(
    z_apex_true
    - z_apex_analytical
)

time_error = abs(
    t_apex_true
    - t_apex_analytical
)

velocity_error_percent = (
    100.0
    * velocity_error
    / abs(Vz_launch_analytical)
)

height_error_percent = (
    100.0
    * height_error
    / abs(z_apex_analytical)
)

time_error_percent = (
    100.0
    * time_error
    / abs(t_apex_analytical)
)

validation_pass = (
    velocity_error_percent < 0.1
    and height_error_percent < 0.1
    and time_error_percent < 0.1
)

target_difference = (
    z_apex_true
    - target_height
)

target_percentage_difference = (
    100.0
    * target_difference
    / target_height
)

F_values = np.arange(
    600.0,
    1200.0 + 1.0,
    50.0
)

c_values = np.arange(
    0.0,
    200.0 + 1.0,
    20.0
)

mass_values = np.arange(
    1.5,
    4.0 + 0.001,
    0.25
)

apex_heights_F = np.zeros_like(F_values)
apex_heights_c = np.zeros_like(c_values)
apex_heights_m = np.zeros_like(mass_values)

for i, F_test in enumerate(F_values):

    Fnet_test = (
        F_test - m * g
    )

    if Fnet_test > 0.0:

        v_test = (
            Fnet_test / c_leg
        ) * (
            1.0
            - np.exp(
                -c_leg * dt_launch / m
            )
        )

        z_test = (
            Fnet_test / c_leg
        ) * (
            dt_launch
            - (m / c_leg)
            * (
                1.0
                - np.exp(
                    -c_leg * dt_launch / m
                )
            )
        )

        apex_heights_F[i] = (
            z_test
            + v_test**2 / (2.0 * g)
        )

for i, c_test in enumerate(c_values):

    if c_test == 0.0:

        v_test = (
            (F_launch / m) - g
        ) * dt_launch

        z_test = (
            0.5
            * ((F_launch / m) - g)
            * dt_launch**2
        )

    else:

        tau_test = m / c_test

        Fnet_test = (
            F_launch - m * g
        )

        v_test = (
            Fnet_test / c_test
        ) * (
            1.0
            - np.exp(
                -dt_launch / tau_test
            )
        )

        z_test = (
            Fnet_test / c_test
        ) * (
            dt_launch
            - tau_test
            * (
                1.0
                - np.exp(
                    -dt_launch / tau_test
                )
            )
        )

    apex_heights_c[i] = (
        z_test
        + v_test**2 / (2.0 * g)
    )

for i, m_test in enumerate(mass_values):

    Fnet_test = (
        F_launch - m_test * g
    )

    if Fnet_test > 0.0:

        tau_test = (
            m_test / c_leg
        )

        v_test = (
            Fnet_test / c_leg
        ) * (
            1.0
            - np.exp(
                -dt_launch / tau_test
            )
        )

        z_test = (
            Fnet_test / c_leg
        ) * (
            dt_launch
            - tau_test
            * (
                1.0
                - np.exp(
                    -dt_launch / tau_test
                )
            )
        )

        apex_heights_m[i] = (
            z_test
            + v_test**2 / (2.0 * g)
        )

test_cases = np.array([
    [0.08, 2.0, 4.0],
    [0.35, 2.0, 4.0],
    [0.08, 8.0, 4.0],
    [0.08, 2.0, 15.0],
    [0.20, 6.0, 12.0],
    [0.01, 0.0, 0.0]
])

transition_results = np.zeros(
    len(test_cases),
    dtype=bool
)

for i, case in enumerate(test_cases):

    vz_test = case[0]

    theta_test = np.deg2rad(
        case[1]
    )

    omega_test = np.deg2rad(
        case[2]
    )

    transition_results[i] = (
        abs(vz_test) <= eps_v
        and abs(theta_test) <= eps_theta
        and abs(omega_test) <= eps_omega
    )

num_trials = 500

mc_window = 11

rng_mc = np.random.default_rng(42)

apex_times_mc = np.full(
    num_trials,
    np.nan
)

apex_heights_mc = np.full(
    num_trials,
    np.nan
)

time_errors_mc = np.full(
    num_trials,
    np.nan
)

height_errors_mc = np.full(
    num_trials,
    np.nan
)

transition_success_mc = np.zeros(
    num_trials,
    dtype=bool
)

for trial in range(num_trials):

    z_measured_mc = (
        z
        + sigma_z
        * rng_mc.standard_normal(t.size)
    )

    v_measured_mc = (
        Vz
        + sigma_v
        * rng_mc.standard_normal(t.size)
    )

    z_est_mc = np.convolve(
        z_measured_mc,
        np.ones(mc_window) / mc_window,
        mode="same"
    )

    v_est_mc = np.convolve(
        v_measured_mc,
        np.ones(mc_window) / mc_window,
        mode="same"
    )

    crossing_mc = np.where(
        (v_est_mc[search_start:-1] > 0.0)
        &
        (v_est_mc[search_start + 1:] <= 0.0)
    )[0]

    if crossing_mc.size:

        idx_mc = (
            search_start
            + crossing_mc[0]
        )

        t1 = t[idx_mc]
        t2 = t[idx_mc + 1]

        v1 = v_est_mc[idx_mc]
        v2 = v_est_mc[idx_mc + 1]

        if v2 != v1:

            t_apex_est_mc = (
                t1
                + (-v1)
                * (t2 - t1)
                / (v2 - v1)
            )

        else:

            t_apex_est_mc = t1

        z_apex_est_mc = np.interp(
            t_apex_est_mc,
            t,
            z_est_mc
        )

        v_apex_est_mc = np.interp(
            t_apex_est_mc,
            t,
            v_est_mc
        )

        apex_times_mc[trial] = (
            t_apex_est_mc
        )

        apex_heights_mc[trial] = (
            z_apex_est_mc
        )

        time_errors_mc[trial] = abs(
            t_apex_est_mc
            - t_apex_true
        )

        height_errors_mc[trial] = abs(
            z_apex_est_mc
            - z_apex_true
        )

        theta_measured_mc = (
            theta
            + sigma_theta * rng_mc.standard_normal(t.size)
        )
        omega_measured_mc = (
            omega
            + sigma_omega * rng_mc.standard_normal(t.size)
        )
        theta_apex_mc = np.interp(t_apex_est_mc, t, theta_measured_mc)
        omega_apex_mc = np.interp(t_apex_est_mc, t, omega_measured_mc)
        transition_success_mc[trial] = (
            abs(v_apex_est_mc) <= eps_v
            and abs(theta_apex_mc) <= eps_theta
            and abs(omega_apex_mc) <= eps_omega
        )

successful_detections = np.sum(
    ~np.isnan(apex_times_mc)
)

detection_rate = (
    100.0
    * successful_detections
    / num_trials
)

successful_transitions = np.sum(
    transition_success_mc
)

transition_rate = (
    100.0
    * successful_transitions
    / num_trials
)

valid_time_errors = (
    time_errors_mc[
        ~np.isnan(time_errors_mc)
    ]
)

valid_height_errors = (
    height_errors_mc[
        ~np.isnan(height_errors_mc)
    ]
)

mean_time_error = np.mean(
    valid_time_errors
)

std_time_error = np.std(
    valid_time_errors
)

max_time_error = np.max(
    valid_time_errors
)

mean_height_error = np.mean(
    valid_height_errors
)

std_height_error = np.std(
    valid_height_errors
)

max_height_error = np.max(
    valid_height_errors
)

# Simplified single-axis attitude-control demonstration only; not the full vehicle.
Kp = 8.0
Ki = 0.0
Kd = 1.0
J = 0.08
torque_limit = 0.5

theta_ref = 0.0
pid_duration = 2.0
pid_time = np.arange(0.0, pid_duration + dt, dt)
pid_idx = 0
theta_pid = np.zeros_like(pid_time)
omega_pid = np.zeros_like(pid_time)
pid_command = np.zeros_like(pid_time)
theta_pid[0] = np.deg2rad(3.0)
omega_pid[0] = np.deg2rad(4.0)
pid_integral = 0.0
integral_limit = 0.25

for i in range(1, len(pid_time)):
    error = theta_ref - theta_pid[i - 1]
    candidate_integral = np.clip(pid_integral + error * dt,
                                 -integral_limit, integral_limit)
    # Derivative-on-measurement avoids derivative kick; torque is saturated.
    raw_torque = Kp * error + Ki * candidate_integral - Kd * omega_pid[i - 1]
    torque = np.clip(raw_torque, -torque_limit, torque_limit)
    # Basic anti-windup: accept integral update when unsaturated or when it
    # would drive the actuator out of saturation.
    if abs(raw_torque) <= torque_limit or np.sign(error) != np.sign(raw_torque):
        pid_integral = candidate_integral
        raw_torque = Kp * error + Ki * pid_integral - Kd * omega_pid[i - 1]
        torque = np.clip(raw_torque, -torque_limit, torque_limit)
    angular_acceleration = torque / J
    omega_pid[i] = omega_pid[i - 1] + angular_acceleration * dt
    theta_pid[i] = theta_pid[i - 1] + omega_pid[i] * dt
    pid_command[i] = torque

pid_final_angle_deg = np.rad2deg(theta_pid[-1])
pid_final_rate_deg_s = np.rad2deg(omega_pid[-1])
pid_recovery_pass = (
    abs(pid_final_angle_deg) < 0.25
    and abs(pid_final_rate_deg_s) < 1.0
)

print("\n" + "=" * 68)
print("KITE-X PYTHON TRANSITION INTELLIGENCE AND VALIDATION")
print("=" * 68)

print("\nSYSTEM")
print(f"Mass                       : {m:.3f} kg")
print(f"Launch force               : {F_launch:.3f} N")
print(f"Launch damping             : {c_leg:.3f} N.s/m")
print(f"Launch pulse               : {dt_launch:.4f} s")
print(f"Target height              : {target_height:.3f} m")
print("NOTE: attitude/rate inputs are synthetic test traces, not sensor data.")

print("\nTRANSITION")
print(f"True apex time             : {t_apex_true:.6f} s")
print(f"True apex height           : {z_apex_true:.6f} m")
print(f"Estimated apex time        : {t_apex_est:.6f} s")
print(f"Estimated apex height      : {z_apex_est:.6f} m")
print(
    f"Transition decision        : "
    f"{'ACCEPTED' if transition_ok else 'REJECTED'}"
)
print(
    f"Rotor command              : "
    f"{1 if transition_ok else 0}"
)

print("\nVALIDATION")
print(
    f"Analytical apex time       : "
    f"{t_apex_analytical:.6f} s"
)
print(
    f"Analytical apex height     : "
    f"{z_apex_analytical:.6f} m"
)
print(
    f"Apex time error            : "
    f"{time_error:.8f} s"
)
print(
    f"Apex height error          : "
    f"{height_error:.8f} m"
)
print(
    f"Validation status          : "
    f"{'PASS' if validation_pass else 'CHECK'}"
)

print("\nSENSITIVITY")
print(
    f"Force range                : "
    f"{F_values[0]:.0f} to {F_values[-1]:.0f} N"
)
print(
    f"Damping range              : "
    f"{c_values[0]:.0f} to {c_values[-1]:.0f} N.s/m"
)
print(
    f"Mass range                 : "
    f"{mass_values[0]:.2f} to {mass_values[-1]:.2f} kg"
)

print("\nMONTE CARLO")
print(f"Trials                     : {num_trials}")
print(
    f"Altitude noise sigma       : "
    f"{sigma_z:.4f} m"
)
print(
    f"Velocity noise sigma       : "
    f"{sigma_v:.4f} m/s"
)
print(
    f"Apex detection rate        : "
    f"{detection_rate:.2f} %"
)
print(
    f"Mean apex-time error       : "
    f"{mean_time_error:.6f} s"
)
print(
    f"Maximum apex-time error    : "
    f"{max_time_error:.6f} s"
)
print(
    f"Mean apex-height error     : "
    f"{mean_height_error:.6f} m"
)
print(
    f"Transition success rate    : "
    f"{transition_rate:.2f} %"
)

print("\nPID FLIGHT STABILIZATION")
print(f"Kp                         : {Kp:.3f}")
print(f"Ki                         : {Ki:.3f}")
print(f"Kd                         : {Kd:.3f}")
print(f"PID simulation duration    : {pid_duration:.2f} s")
print(
    f"Final attitude             : "
    f"{pid_final_angle_deg:.6f} deg"
)
print(
    f"Final angular rate         : "
    f"{pid_final_rate_deg_s:.6f} deg/s"
)
print(f"PID recovery check          : {'PASS' if pid_recovery_pass else 'CHECK'}")

print("\n" + "=" * 68)

with open(
    "KITE_X_PYTHON_OUTPUTS.txt",
    "w",
    encoding="utf-8"
) as f:

    f.write(
        f"KITE-X PYTHON TRANSITION INTELLIGENCE AND VALIDATION\n\n"
    )

    f.write(
        f"True apex time: {t_apex_true:.6f} s\n"
    )

    f.write(
        f"True apex height: {z_apex_true:.6f} m\n"
    )

    f.write(
        f"Estimated apex time: {t_apex_est:.6f} s\n"
    )

    f.write(
        f"Estimated apex height: {z_apex_est:.6f} m\n"
    )

    f.write(
        f"Transition: "
        f"{'ACCEPTED' if transition_ok else 'REJECTED'}\n"
    )

    f.write(
        f"Rotor command: "
        f"{1 if transition_ok else 0}\n"
    )

    f.write(
        f"Validation: "
        f"{'PASS' if validation_pass else 'CHECK'}\n"
    )

    f.write(
        f"Detection rate: {detection_rate:.4f} %\n"
    )

    f.write(
        f"Mean apex-time error: "
        f"{mean_time_error:.6f} s\n"
    )

    f.write(
        f"Mean apex-height error: "
        f"{mean_height_error:.6f} m\n"
    )

    f.write(
        f"Transition success rate: "
        f"{transition_rate:.4f} %\n"
    )

    f.write(
        f"PID final attitude: "
        f"{pid_final_angle_deg:.6f} deg\n"
    )

    f.write(
        f"PID final angular rate: "
        f"{pid_final_rate_deg_s:.6f} deg/s\n"
    )

np.savetxt(
    "KITE_X_MONTE_CARLO_RESULTS.csv",
    np.column_stack(
        (
            np.arange(
                1,
                num_trials + 1
            ),
            apex_times_mc,
            apex_heights_mc,
            time_errors_mc,
            height_errors_mc,
            transition_success_mc.astype(int)
        )
    ),
    delimiter=",",
    header=(
        "trial,apex_time,apex_height,"
        "time_error,height_error,"
        "transition_success"
    ),
    comments=""
)

plt.figure(figsize=(9, 5))
plt.plot(t, z, linewidth=1.8)
plt.plot(
    t_apex_true,
    z_apex_true,
    "o",
    markersize=7
)
plt.xlabel("Time [s]")
plt.ylabel("Altitude z [m]")
plt.title("KITE-X Ballistic Trajectory")
plt.grid(True)
plt.tight_layout()
plt.savefig(
    "PYTHON_TRAJECTORY.png",
    dpi=300
)
plt.close()

plt.figure(figsize=(9, 5))
plt.plot(
    t,
    Vz,
    linewidth=1.8
)
plt.axhline(
    0.0,
    linestyle="--"
)
plt.axvline(
    t_apex_true,
    linestyle="--"
)
plt.xlabel("Time [s]")
plt.ylabel("Vertical Velocity [m/s]")
plt.title("KITE-X Vertical Velocity")
plt.grid(True)
plt.tight_layout()
plt.savefig(
    "PYTHON_VERTICAL_VELOCITY.png",
    dpi=300
)
plt.close()

plt.figure(figsize=(9, 5))
plt.plot(
    t,
    z,
    linewidth=1.5,
    label="True"
)
plt.plot(
    t,
    z_measured,
    ".",
    markersize=2,
    label="Measured"
)
plt.plot(
    t,
    z_est,
    linewidth=1.3,
    label="Estimated"
)
plt.xlabel("Time [s]")
plt.ylabel("Altitude [m]")
plt.title("KITE-X Altitude State Estimation")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(
    "PYTHON_ALTITUDE_ESTIMATION.png",
    dpi=300
)
plt.close()

plt.figure(figsize=(9, 5))
plt.plot(
    t,
    Vz,
    linewidth=1.5,
    label="True"
)
plt.plot(
    t,
    v_measured,
    ".",
    markersize=2,
    label="Measured"
)
plt.plot(
    t,
    v_est,
    linewidth=1.3,
    label="Estimated"
)
plt.xlabel("Time [s]")
plt.ylabel("Vertical Velocity [m/s]")
plt.title("KITE-X Vertical Velocity State Estimation")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(
    "PYTHON_VELOCITY_ESTIMATION.png",
    dpi=300
)
plt.close()

plt.figure(figsize=(9, 5))
plt.step(
    t,
    rotor_command,
    where="post",
    linewidth=1.8
)
plt.yticks(
    [0, 1],
    ["OFF", "ON"]
)
plt.xlabel("Time [s]")
plt.ylabel("Rotor Command")
plt.title("KITE-X Intelligent Rotor Activation")
plt.grid(True)
plt.tight_layout()
plt.savefig(
    "PYTHON_ROTOR_COMMAND.png",
    dpi=300
)
plt.close()

plt.figure(figsize=(9, 5))
plt.plot(
    F_values,
    apex_heights_F,
    linewidth=1.8
)
plt.xlabel("Launch Force [N]")
plt.ylabel("Predicted Apex Height [m]")
plt.title(
    "KITE-X Apex Height Sensitivity to Launch Force"
)
plt.grid(True)
plt.tight_layout()
plt.savefig(
    "PYTHON_SENSITIVITY_FORCE.png",
    dpi=300
)
plt.close()

plt.figure(figsize=(9, 5))
plt.plot(
    c_values,
    apex_heights_c,
    linewidth=1.8
)
plt.xlabel("Launch Damping [N.s/m]")
plt.ylabel("Predicted Apex Height [m]")
plt.title(
    "KITE-X Apex Height Sensitivity to Launch Damping"
)
plt.grid(True)
plt.tight_layout()
plt.savefig(
    "PYTHON_SENSITIVITY_DAMPING.png",
    dpi=300
)
plt.close()

plt.figure(figsize=(9, 5))
plt.plot(
    mass_values,
    apex_heights_m,
    linewidth=1.8
)
plt.xlabel("System Mass [kg]")
plt.ylabel("Predicted Apex Height [m]")
plt.title(
    "KITE-X Apex Height Sensitivity to Mass"
)
plt.grid(True)
plt.tight_layout()
plt.savefig(
    "PYTHON_SENSITIVITY_MASS.png",
    dpi=300
)
plt.close()

plt.figure(figsize=(9, 5))
plt.hist(
    valid_time_errors,
    bins=30
)
plt.xlabel(
    "Absolute Apex-Time Error [s]"
)
plt.ylabel("Number of Trials")
plt.title(
    "KITE-X Monte Carlo Apex-Time Error"
)
plt.grid(True)
plt.tight_layout()
plt.savefig(
    "PYTHON_MC_TIME_ERROR.png",
    dpi=300
)
plt.close()

plt.figure(figsize=(9, 5))
plt.hist(
    valid_height_errors,
    bins=30
)
plt.xlabel(
    "Absolute Apex-Height Error [m]"
)
plt.ylabel("Number of Trials")
plt.title(
    "KITE-X Monte Carlo Apex-Height Error"
)
plt.grid(True)
plt.tight_layout()
plt.savefig(
    "PYTHON_MC_HEIGHT_ERROR.png",
    dpi=300
)
plt.close()

plt.figure(figsize=(9, 5))
plt.plot(
    np.arange(1, num_trials + 1),
    apex_times_mc,
    ".",
    markersize=4
)
plt.axhline(
    t_apex_true,
    linestyle="--"
)
plt.xlabel("Monte Carlo Trial")
plt.ylabel("Detected Apex Time [s]")
plt.title(
    "KITE-X Apex Detection Across Monte Carlo Trials"
)
plt.grid(True)
plt.tight_layout()
plt.savefig(
    "PYTHON_MC_APEX_TRIALS.png",
    dpi=300
)
plt.close()

plt.figure(figsize=(9, 5))
plt.plot(
    pid_time,
    np.rad2deg(
        theta_pid
    ),
    linewidth=1.8
)
plt.axhline(
    0.0,
    linestyle="--"
)
plt.xlabel("Time [s]")
plt.ylabel("Pitch Error [deg]")
plt.title(
    "KITE-X PID Flight Stabilization"
)
plt.grid(True)
plt.tight_layout()
plt.savefig(
    "PYTHON_PID_STABILIZATION.png",
    dpi=300
)
plt.close()


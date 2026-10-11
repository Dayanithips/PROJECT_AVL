clear; clc; close all;
% System Parameters
m = 2.5;            % Total mass (kg)
g = 9.81;           % Acceleration due to gravity (m/s^2)
F_pulse = 900.0;    % Total peak force across 4 legs (N)
dt = 0.0125;        % Calibrated pulse duration for 0.60 m apex (s)
k_leg = 2000;       % Leg spring stiffness (N/m)
c_leg = 20.0;       % Leg damping coefficient (N*s/m)
c_air = 5.0;        % In-flight air drag / damping (N*s/m)
% 1. Thrust/Push Phase (0 <= t <= dt)
thrust_ode = @(t, y) [y(2); (F_pulse - k_leg*y(1) - c_leg*y(2) - m*g)/m];
[t1, y1] = ode45(thrust_ode, [0, dt], [0; 0]);
z_launch = y1(end, 1);
v_launch = y1(end, 2);
% 2. Ballistic Flight Phase (t > dt up to Apex)
ballistic_ode = @(t, y) [y(2); -g - (c_air/m)*y(2)];
t_apex_est = dt + (v_launch / g);
[t2, y2] = ode45(ballistic_ode, [dt, t_apex_est], [z_launch; v_launch]);
% Combine Trajectory Data
t_all = [t1; t2(2:end)];
z_all = [y1(:,1); y2(2:end,1)];
% Plot Results
figure('Color', 'w', 'Position', [100, 100, 750, 450]);
plot(t_all, z_all, 'b', 'LineWidth', 2.2);
grid on;
xlabel('Time (s)', 'FontSize', 11, 'FontWeight', 'bold');
ylabel('Vertical Displacement (m)', 'FontSize', 11, 'FontWeight', 'bold');
title('UAV-UGV Hop Trajectory (Target Apex = 0.60 m)', 'FontSize', 12, 'FontWeight', 'bold');
xlim([0, max(t_all)*1.1]);
ylim([0, 0.75]);
% Console Output
fprintf('===========================================\n');
fprintf('     FINAL VALIDATED HOP SIMULATION        \n');
fprintf('===========================================\n');
fprintf('System Mass (m)       : %.2f kg\n', m);
fprintf('Peak Force (F_pulse)  : %.1f N\n', F_pulse);
fprintf('Impulse Duration (dt) : %.4f s (%.1f ms)\n', dt, dt*1000);
fprintf('Liftoff Velocity (v0) : %.4f m/s\n', v_launch);
fprintf('Achieved Apex Height  : %.4f m (%.2f ft)\n', max(z_all), max(z_all)*3.28084);
fprintf('===========================================\n');
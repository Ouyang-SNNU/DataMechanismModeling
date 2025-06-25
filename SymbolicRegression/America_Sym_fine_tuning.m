clear all
clc
data = readtable('../Data/Compartment_US.csv');

S = data.S;
E = data.E;
I = data.I;
R = data.R;
p_values = data.p;
I_new_nn = data.I_new_nn;
I_cum_nn = data.I_cum_nn;
I_new_real = data.I_new_real;
I_cum_real = data.I_cum_real;

a=270;
b=440;
S = S(a:b);
I = I(a:b);
E = E(a:b);
R = R(a:b);
p_values = p_values(a:b);
I_new_nn = I_new_nn(a:b);
I_cum_nn = I_cum_nn(a:b);
I_new_real = I_new_real(a:b);
I_cum_real = I_cum_real(a:b);

start_date = datetime(2020, 10, 14);
time = start_date + days(0:length(S)-1);
tspan = 1:length(S);

S0 = S(1);
E0 = E(1);
I0 = I(1);
R0 = 1 - S0 - E0 - I0;
p0 = p_values(1);
N = 331002647;
initial_conditions = [S0 * N; E0 * N; I0 * N; R0 * N; p0];

beta_n = 0.4944;
beta_a = 0.1;
sigma = 0.2;
gamma = 0.162 + 4.816e-3;

initial_params = [-9.738531/0.017726593; 0.017726593];
ub = [-500, 0.4]; 
lb = [-1000, 0.001];

options = optimoptions('fmincon', 'Algorithm', 'interior-point', 'Display', 'iter', 'ConstraintTolerance', 1e-3, 'TolX', 1e-15);
result = fmincon(@(params) objective(params, initial_conditions, tspan, beta_n, beta_a, sigma, gamma, N, I_new_real), initial_params, [], [], [], [], lb, ub, [], options);

a_opt = result(1)
b_opt = result(2)






function dydt = SEIR(y, a, b, beta_n, beta_a, sigma, gamma, N)
    S = y(1);
    E = y(2);
    I = y(3);
    R = y(4);
    p = y(5);
    dS = -(p * beta_n + (1 - p) * beta_a) * S * I / N;
    dE = (p * beta_n + (1 - p) * beta_a) * S * I / N - sigma * E;
    dI = sigma * E - gamma * I;
    dR = gamma * I;
    dp = b * p * (1 - p) * (1+a*I/N);
    dydt = [dS; dE; dI; dR; dp];
end

function I_new_ODE = solve_seir_and_get_I_new(params, initial_conditions, tspan, beta_n, beta_a, sigma, gamma, N)
    a = params(1);
    b = params(2);
    [~, sol] = ode45(@(t, y) SEIR(y, a, b, beta_n, beta_a, sigma, gamma, N), tspan, initial_conditions);
    I_new_ODE = gamma * sol(:, 3);
end

function mse = objective(params, initial_conditions, tspan, beta_n, beta_a, sigma, gamma, N, I_new_real)
    I_new_ODE = solve_seir_and_get_I_new(params, initial_conditions, tspan, beta_n, beta_a, sigma, gamma, N);
    mse = mean((I_new_ODE - I_new_real).^2);
end
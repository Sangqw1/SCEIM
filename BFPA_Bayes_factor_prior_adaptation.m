function [x_post, P_post, out] = BFPA_Bayes_factor_prior_adaptation(x_pred, P_pred, H, z, R_nominal, varargin)
% Inputs:
%     x_pred    - Predicted state, n_x x 1.
%     P_pred    - Predicted covariance, n_x x n_x.
%     H         - Measurement matrix, n_z x n_x.
%     z         - Measurement vector, n_z x 1.
%     R_nominal - Nominal measurement covariance, n_z x n_z.
%
% Outputs:
%     x_post - Posterior state.
%     P_post - Posterior covariance.
%     out    - Diagnostics: R_eff, E_t, E_lambda, kappa, K,
%              u_kj, BF_kj, Beta parameters, iterations, converged,
%              residual, B_kj, B_over_R.

opts = local_parse_options(varargin{:});

x_pred = x_pred(:);
z = z(:);
[n_z, n_x] = size(H);

if numel(x_pred) ~= n_x
    error('BFPA_Bayes_factor_prior_adaptation:InvalidStateSize', ...
        'x_pred length must match size(H,2).');
end
if numel(z) ~= n_z
    error('BFPA_Bayes_factor_prior_adaptation:InvalidMeasurementSize', ...
        'z length must match size(H,1).');
end
if ~isequal(size(P_pred), [n_x, n_x])
    error('BFPA_Bayes_factor_prior_adaptation:InvalidCovarianceSize', ...
        'P_pred must be size(H,2)-by-size(H,2).');
end
if ~isequal(size(R_nominal), [n_z, n_z])
    error('BFPA_Bayes_factor_prior_adaptation:InvalidNoiseSize', ...
        'R_nominal must be size(H,1)-by-size(H,1).');
end

P_pred = local_symmetrize(P_pred);
R_nominal = local_symmetrize(R_nominal);
R_diag = max(abs(diag(R_nominal)), opts.regularization);

S0 = local_symmetrize(H * P_pred * H' + R_nominal);
S0 = S0 + opts.regularization * eye(n_z);
innov0 = z - H * x_pred;
u_kj = innov0 ./ sqrt(max(abs(diag(S0)), opts.regularization));

[a0, b0, BF_kj, log_BF_kj] = local_initialize_bfpa_prior( ...
    u_kj, opts.nu_measurement, opts.prior_strength);

E_t = ones(n_z, 1);
E_lambda = ones(n_z, 1);
E_log_lambda = zeros(n_z, 1);
E_log_pi = local_digamma(a0) - local_digamma(a0 + b0);
E_log_1_pi = local_digamma(b0) - local_digamma(a0 + b0);

x_post = x_pred;
P_post = P_pred;
K = zeros(n_x, n_z);
B_kj = zeros(n_z, 1);
B_over_R = zeros(n_z, 1);
converged = false;

for iter = 1:opts.max_iterations
    x_prev = x_post;

    kappa = E_t + (1 - E_t) .* E_lambda;
    kappa = min(max(kappa, opts.lambda_bounds(1)), opts.lambda_bounds(2));

    R_eff = diag(R_diag ./ kappa);
    R_eff = R_eff + opts.regularization * eye(n_z);

    S = local_symmetrize(H * P_pred * H' + R_eff);
    S = S + opts.regularization * eye(n_z);
    K = (P_pred * H') / S;

    innovation = z - H * x_pred;
    x_post = x_pred + K * innovation;
    P_post = local_symmetrize(P_pred - K * H * P_pred);

    post_residual = z - H * x_post;
    residual_second_moment = post_residual * post_residual' + H * P_post * H';
    B_kj = max(diag(residual_second_moment), 0);
    B_over_R = B_kj ./ R_diag;

    log_pt1 = E_log_pi - 0.5 * B_over_R;
    log_pt0 = E_log_1_pi + 0.5 * E_log_lambda - 0.5 * E_lambda .* B_over_R;
    E_t = local_probability_from_two_logs(log_pt1, log_pt0);

    alpha_lambda = 0.5 * (1 - E_t) + 0.5 * opts.nu_measurement;
    beta_lambda = 0.5 * B_over_R .* (1 - E_t) + 0.5 * opts.nu_measurement;
    beta_lambda = max(beta_lambda, opts.regularization);

    E_lambda = alpha_lambda ./ beta_lambda;
    E_lambda = min(max(E_lambda, opts.lambda_bounds(1)), opts.lambda_bounds(2));
    E_log_lambda = local_digamma(alpha_lambda) - log(beta_lambda);

    a_post = a0 + E_t;
    b_post = b0 + 1 - E_t;
    E_log_pi = local_digamma(a_post) - local_digamma(a_post + b_post);
    E_log_1_pi = local_digamma(b_post) - local_digamma(a_post + b_post);

    rel_change = norm(x_post - x_prev) / max(norm(x_post), opts.regularization);
    if opts.verbose
        fprintf('GSTM VB iteration %d: mean(E_t)=%.4f, mean(E_lambda)=%.4f, rel_change=%.3e\n', ...
            iter, mean(E_t), mean(E_lambda), rel_change);
    end
    if rel_change < opts.tolerance
        converged = true;
        break;
    end
end

kappa = E_t + (1 - E_t) .* E_lambda;
kappa = min(max(kappa, opts.lambda_bounds(1)), opts.lambda_bounds(2));
R_eff = diag(R_diag ./ kappa);

out = struct();
out.R_eff = R_eff;
out.E_t = E_t;
out.E_lambda = E_lambda;
out.kappa = kappa;
out.K = K;
out.u_kj = u_kj;
out.normalizedInnovation = u_kj;
out.BF_kj = BF_kj;
out.log_BF_kj = log_BF_kj;
out.beta_a0 = a0;
out.beta_b0 = b0;
out.beta_a = a0 + E_t;
out.beta_b = b0 + 1 - E_t;
out.iterations = iter;
out.converged = converged;
out.residual = z - H * x_post;
out.B_kj = B_kj;
out.B_over_R = B_over_R;
out.gamma2 = B_over_R;
end

function opts = local_parse_options(varargin)
opts = struct();
opts.nu_measurement = 5;
opts.max_iterations = 50;
opts.tolerance = 1e-7;
opts.prior_strength = 1.0;
opts.lambda_bounds = [0.05, 1.0];
opts.regularization = 1e-10;
opts.verbose = false;

if mod(numel(varargin), 2) ~= 0
    error('BFPA_Bayes_factor_prior_adaptation:InvalidOptions', ...
        'Optional inputs must be name-value pairs.');
end

for i = 1:2:numel(varargin)
    name = lower(string(varargin{i}));
    value = varargin{i + 1};
    switch name
        case "numeasurement"
            opts.nu_measurement = value;
        case "maxiterations"
            opts.max_iterations = value;
        case "tolerance"
            opts.tolerance = value;
        case "priorstrength"
            opts.prior_strength = value;
        case "lambdabounds"
            opts.lambda_bounds = value;
        case "regularization"
            opts.regularization = value;
        case "verbose"
            opts.verbose = logical(value);
        otherwise
            error('BFPA_Bayes_factor_prior_adaptation:UnknownOption', ...
                'Unknown option "%s".', name);
    end
end

opts.nu_measurement = max(double(opts.nu_measurement), eps);
opts.max_iterations = max(1, round(double(opts.max_iterations)));
opts.tolerance = max(double(opts.tolerance), eps);
opts.prior_strength = max(double(opts.prior_strength), eps);
opts.lambda_bounds = double(opts.lambda_bounds(:)');
if numel(opts.lambda_bounds) ~= 2 || opts.lambda_bounds(1) <= 0 || opts.lambda_bounds(2) <= opts.lambda_bounds(1)
    error('BFPA_Bayes_factor_prior_adaptation:InvalidLambdaBounds', ...
        'LambdaBounds must be [lower upper] with 0 < lower < upper.');
end
opts.regularization = max(double(opts.regularization), eps);
end

function [a0, b0, BF_kj, log_BF_kj] = local_initialize_bfpa_prior(u_kj, nu_t, prior_strength)
u_kj = u_kj(:);
log_BF_kj = 0.5 * log(nu_t / 2) + gammaln(nu_t / 2) - gammaln((nu_t + 1) / 2) ...
    - 0.5 * (u_kj .^ 2) ...
    + 0.5 * (nu_t + 1) * log1p((u_kj .^ 2) / nu_t);

rho_bfpa = 1 ./ (1 + exp(-min(max(log_BF_kj, -50), 50)));
BF_kj = exp(min(max(log_BF_kj, -700), 700));

a0 = prior_strength * rho_bfpa;
b0 = prior_strength * (1 - rho_bfpa);
a0 = max(a0, eps);
b0 = max(b0, eps);
end

function p = local_probability_from_two_logs(log_a, log_b)
max_log = max([log_a(:), log_b(:)], [], 2);
a = exp(log_a(:) - max_log);
b = exp(log_b(:) - max_log);
p = a ./ max(a + b, realmin);
p = min(max(p, 1e-6), 1 - 1e-6);
end

function y = local_digamma(x)
% Numeric digamma approximation, included to keep this file self-contained.
x = double(x);
y = zeros(size(x));
small = x < 6;
while any(small(:))
    y(small) = y(small) - 1 ./ x(small);
    x(small) = x(small) + 1;
    small = x < 6;
end
inv_x = 1 ./ x;
inv_x2 = inv_x .^ 2;
y = y + log(x) - 0.5 * inv_x ...
    - inv_x2 .* (1/12 - inv_x2 .* (1/120 - inv_x2 .* (1/252)));
end

function A = local_symmetrize(A)
A = 0.5 * (A + A');
end

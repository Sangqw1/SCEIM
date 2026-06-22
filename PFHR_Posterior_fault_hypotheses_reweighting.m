function [p_fault, posterior_hypothesis, fault_relative_posterior, diagnostics] = ...
    PFHR_Posterior_fault_hypotheses_reweighting(pap_subset, subset_log_evidence, use_posterior, varargin)
%
% Inputs:
%     pap_subset          - Prior hypothesis weights. Element 1 is H_0
%                           no-fault; elements 2:end are monitored faults.
%     subset_log_evidence - Same-size log predictive evidence vector.

%     use_posterior       - 1 enables redistribution; 0 keeps prior weights.


% Outputs:
%     p_fault              - PL risk weights.
%     posterior_hypothesis - Normalized posterior over all valid hypotheses.
%     fault_relative_posterior
%                          - Conditional posterior over monitored faults.
%     diagnostics          - Valid masks, risk-mass checks, status reason.
%

opts = local_parse_options(varargin{:});

if nargin < 3 || isempty(use_posterior)
    use_posterior = 0;
end

original_size = size(pap_subset);
pap_col = pap_subset(:);
log_like_col = subset_log_evidence(:);

p_fault_col = pap_col;
posterior_col = local_normalize_prior(pap_col);
fault_relative_col = zeros(size(pap_col));

diagnostics = struct();
diagnostics.validHypothesis = false(size(pap_col));
diagnostics.validFaultHypothesis = false(size(pap_col));
diagnostics.priorFaultMass = 0;
diagnostics.redistributedFaultMass = 0;
diagnostics.posteriorEnabled = logical(use_posterior);
diagnostics.reason = "posterior disabled";

if ~isempty(p_fault_col)
    p_fault_col(1) = opts.no_fault_pl_coefficient;
end

if ~logical(use_posterior)
    p_fault = reshape(p_fault_col, original_size);
    posterior_hypothesis = reshape(posterior_col, original_size);
    fault_relative_posterior = reshape(fault_relative_col, original_size);
    return;
end

if isempty(pap_col) || numel(pap_col) <= 1
    diagnostics.reason = "not enough hypotheses";
    p_fault = reshape(p_fault_col, original_size);
    posterior_hypothesis = reshape(posterior_col, original_size);
    fault_relative_posterior = reshape(fault_relative_col, original_size);
    return;
end

if isempty(log_like_col) || numel(log_like_col) ~= numel(pap_col)
    diagnostics.reason = "prior and evidence size mismatch";
    p_fault = reshape(p_fault_col, original_size);
    posterior_hypothesis = reshape(posterior_col, original_size);
    fault_relative_posterior = reshape(fault_relative_col, original_size);
    return;
end

prior_for_diagnostics = pap_col;
prior_for_diagnostics(1) = max(prior_for_diagnostics(1), realmin);

valid = isfinite(prior_for_diagnostics) & prior_for_diagnostics > 0 & isfinite(log_like_col);
diagnostics.validHypothesis = valid;
if ~any(valid)
    diagnostics.reason = "no valid hypotheses";
    p_fault = reshape(p_fault_col, original_size);
    posterior_hypothesis = reshape(posterior_col, original_size);
    fault_relative_posterior = reshape(fault_relative_col, original_size);
    return;
end

valid_idx = find(valid);
log_weight_all = log(prior_for_diagnostics(valid_idx)) + log_like_col(valid_idx);
posterior_valid = local_softmax_from_log_weights(log_weight_all);
posterior_col = zeros(size(pap_col));
posterior_col(valid_idx) = posterior_valid;

fault_idx = (2:numel(pap_col))';
valid_fault_idx = fault_idx(isfinite(pap_col(fault_idx)) & pap_col(fault_idx) > 0 & isfinite(log_like_col(fault_idx)));
diagnostics.validFaultHypothesis(valid_fault_idx) = true;

if isempty(valid_fault_idx)
    diagnostics.reason = "no valid monitored fault hypotheses";
    p_fault = reshape(p_fault_col, original_size);
    posterior_hypothesis = reshape(posterior_col, original_size);
    fault_relative_posterior = reshape(fault_relative_col, original_size);
    return;
end

% Eq. (36): posterior weight inside monitored fault event M.
log_weight_fault = log(pap_col(valid_fault_idx)) + log_like_col(valid_fault_idx);
alpha_fault = local_softmax_from_log_weights(log_weight_fault);
if ~any(alpha_fault > 0)
    diagnostics.reason = "no valid posterior fault mass";
    p_fault = reshape(p_fault_col, original_size);
    posterior_hypothesis = reshape(posterior_col, original_size);
    fault_relative_posterior = reshape(fault_relative_col, original_size);
    return;
end

prior_fault_mass = sum(pap_col(valid_fault_idx));
fault_relative_col(valid_fault_idx) = alpha_fault;

% Eq. (40): posterior fault hypothesis weight with total monitored risk fixed.
p_fault_col(valid_fault_idx) = prior_fault_mass .* alpha_fault;
p_fault_col(1) = opts.no_fault_pl_coefficient;

diagnostics.priorFaultMass = prior_fault_mass;
diagnostics.redistributedFaultMass = sum(p_fault_col(valid_fault_idx));
diagnostics.reason = "posterior redistribution applied";

p_fault = reshape(p_fault_col, original_size);
posterior_hypothesis = reshape(posterior_col, original_size);
fault_relative_posterior = reshape(fault_relative_col, original_size);
end

function opts = local_parse_options(varargin)
opts = struct();
opts.no_fault_pl_coefficient = 2;

if mod(numel(varargin), 2) ~= 0
    error('PFHR_Posterior_fault_hypotheses_reweighting:InvalidOptions', ...
        'Optional inputs must be name-value pairs.');
end

for i = 1:2:numel(varargin)
    name = lower(string(varargin{i}));
    value = varargin{i + 1};
    switch name
        case "nofaultplcoefficient"
            opts.no_fault_pl_coefficient = value;
        otherwise
            error('PFHR_Posterior_fault_hypotheses_reweighting:UnknownOption', ...
                'Unknown option "%s".', name);
    end
end
end

function posterior = local_normalize_prior(prior)
posterior = zeros(size(prior));
if isempty(prior)
    return;
end

prior = prior(:);
prior(1) = max(prior(1), realmin);
valid = isfinite(prior) & prior > 0;
prior_sum = sum(prior(valid));
if prior_sum > 0 && isfinite(prior_sum)
    posterior(valid) = prior(valid) ./ prior_sum;
end
end

function weight = local_softmax_from_log_weights(log_weight)
max_log_weight = max(log_weight);
weight = exp(log_weight - max_log_weight);
weight_sum = sum(weight);
if ~isfinite(weight_sum) || weight_sum <= 0
    weight = zeros(size(log_weight));
else
    weight = weight ./ weight_sum;
end
end

#defaults for computing log potential for bayesian models
function log_potential(state::AbstractState, model::AbstractModel, cv::AbstractLogProbEstimator)
    state.lp_evals += length(cv.inds)
    return (log_prior(state.θ, model) + 
            log_likelihood_cv!(state, model, cv) + 
            log_likelihood_diff!(state, model, cv))
end
function grad_log_potential(state::AbstractState, model::AbstractModel, cv::AbstractLogProbEstimator)
    state.grad_lp_evals += length(cv.inds)
    return (grad_log_prior(state.θ, model) .+ 
            grad_log_likelihood_cv!(state, model, cv) .+ 
            grad_log_likelihood_diff!(state, model, cv))
end
function hess_log_potential(state::AbstractState, model::AbstractModel, cv::AbstractLogProbEstimator)
    state.hess_lp_evals += length(cv.inds)
    return (hess_log_prior(state.θ, model) + 
            hess_log_likelihood_cv!(state, model, cv) + 
            hess_log_likelihood_diff!(state, model, cv))
end
function log_likelihood_sum(state::AbstractState, model::AbstractModel, cv::AbstractLogProbEstimator)
    state.lp_evals += length(cv.inds)
    return mapreduce(x->log_likelihood(x, state.θ, model), +, @view(model.dataset[cv.inds]))
end
function log_likelihood_array(state::AbstractState, model::AbstractModel, cv::AbstractLogProbEstimator)
    state.lp_evals += length(cv.inds)
    λ = x -> log_likelihood(x, state.θ, model)
    return λ.(@view(model.dataset[cv.inds]))
end
function grad_log_likelihood_array(state::AbstractState, model::AbstractModel, cv::AbstractLogProbEstimator)
    state.grad_lp_evals += length(cv.inds)
    λ = x -> grad_log_likelihood(x, state.θ, model)
    return λ.(@view(model.dataset[cv.inds]))
end
function hess_log_likelihood_array(state::AbstractState, model::AbstractModel, cv::AbstractLogProbEstimator)
    state.hess_lp_evals += length(cv.inds)
    λ = x -> hess_log_likelihood(x, state.θ, model)
    return λ.(@view(model.dataset[cv.inds]))
end
function debiased_log_potential(state::AbstractState, model::AbstractModel, cv::AbstractLogProbEstimator)
    n = model.N
    m = length(cv.inds)
    liks = log_likelihood_array(state, model, cv)
    ds = liks .- log_likelihood_cv_array!(state, model, cv)
    lp = log_likelihood_cv!(state, model, cv) + (n ./ m) * sum(ds)
    corrected_lp = lp - ((n^2)/(2*m))*var(ds)
    return log_prior(state.θ, model) + corrected_lp
end
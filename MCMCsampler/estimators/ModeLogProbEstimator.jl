@with_kw mutable struct ModeLogProbEstimator <: SizeBasedLogProbEstimator
    N::Int64
    inds::Union{AbstractArray, Nothing} = nothing
    initialized::Bool = false
    mode::Union{AbstractArray, Nothing} = nothing
    sum_potential_mode::Union{Float64, Nothing} = nothing
    sum_grad_potential_mode::Union{AbstractArray, Nothing} = nothing
    sum_hess_potential_mode::Union{AbstractArray, UniformScaling, Nothing} = nothing
    # ADAM parameters
    α::Float64 = 0.001
    β1::Float64 = 0.9
    β2::Float64 = 0.999
    ϵ::Float64 = 10^-8
    tol::Float64 = 1e-4
    mode_n::Int64 = 1 # mode finding sample size
end

function update_estimator!(state::AbstractState, model::AbstractModel, cv::ModeLogProbEstimator, 
                            θ′::Union{AbstractVector, Nothing}, μ0::Union{Float64, Nothing}, inds::Union{AbstractArray, Nothing})
    if !cv.initialized
        θ0 = copy(state.θ)
        cv.mode = find_map!(state, model, cv)
        @info cv.mode
        state.θ = cv.mode
        cv_zero = ZeroLogProbEstimator(N = model.N)
        update_estimator!(state, model, cv_zero, nothing, nothing, nothing)
        cv.sum_potential_mode = mapreduce(identity, +, log_likelihood_array(state, model, cv_zero))
        cv.sum_grad_potential_mode = mapreduce(identity, +, grad_log_likelihood_array(state, model, cv_zero))
        # cv.sum_hess_potential_mode = mapreduce(identity, +, hess_log_likelihood_array(state, model, cv_zero))
        cv.initialized = true
        state.θ = θ0
    end
    cv.inds = isnothing(inds) ? sample(state.rng, [1:model.N;], cv.N, replace = false) : inds
end

function update_inds!(cv::ModeLogProbEstimator, inds::Array{Int64})
    if length(inds) != cv.N
        error("inds size does not match that specified by the estimator")
    end
    cv.inds = inds
end

function log_likelihood_cv!(state::AbstractState, model::AbstractModel, cv::ModeLogProbEstimator)
    return cv.sum_potential_mode
end
function grad_log_likelihood_cv!(state::AbstractState, model::AbstractModel, cv::ModeLogProbEstimator)
    return cv.sum_grad_potential_mode
end
function hess_log_likelihood_cv!(state::AbstractState, model::AbstractModel, cv::ModeLogProbEstimator) 
    return cv.sum_hess_potential_mode
end
function log_likelihood_cv_array!(state::AbstractState, model::AbstractModel, cv::ModeLogProbEstimator)
    state.lp_evals += length(cv.inds)
    λ = x -> log_likelihood(x, cv.mode, model)
    return λ.(@view(model.dataset[cv.inds]))
end

function log_likelihood_diff!(state::AbstractState, model::AbstractModel, cv::ModeLogProbEstimator)
    state.lp_evals += 2*length(cv.inds)
    return (model.N ./ cv.N) * 
            (mapreduce(x->log_likelihood(x, state.θ, model), +, @view(model.dataset[cv.inds])) -
            mapreduce(x->log_likelihood(x, cv.mode, model), +, @view(model.dataset[cv.inds])))
end
function grad_log_likelihood_diff!(state::AbstractState, model::AbstractModel, cv::ModeLogProbEstimator)
    state.grad_lp_evals += 2*length(cv.inds)
    return (model.N ./ cv.N) * 
            (mapreduce(x->grad_log_likelihood(x, state.θ, model), +, @view(model.dataset[cv.inds])) -
            mapreduce(x->grad_log_likelihood(x, cv.mode, model), +, @view(model.dataset[cv.inds])))
end
function hess_log_likelihood_diff!(state::AbstractState, model::AbstractModel, cv::ModeLogProbEstimator)
    state.hess_lp_evals += 2*length(cv.inds)
    return (model.N ./ cv.N) * 
            (mapreduce(x->hess_log_likelihood(x, state.θ, model), +, @view(model.dataset[cv.inds])) -
            mapreduce(x->hess_log_likelihood(x, cv.mode, model), +, @view(model.dataset[cv.inds])))
end
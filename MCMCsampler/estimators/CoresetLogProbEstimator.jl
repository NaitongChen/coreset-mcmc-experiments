@with_kw mutable struct CoresetLogProbEstimator <: SizeBasedLogProbEstimator
    N::Int64
    inds::Union{AbstractArray, Nothing} = nothing
    weights::Union{AbstractArray, Nothing} = nothing
    sub_dataset::Union{AbstractArray, Nothing} = nothing
    sub_xp::Union{AbstractArray, Nothing} = nothing
end

function init_estimator!(state::AbstractState, model::AbstractModel, cv::CoresetLogProbEstimator)
    if isnothing(cv.inds)
        if isnothing(model.sampler)
            cv.inds = sample(state.rng, [1:model.N;], cv.N; replace = false)
        else
            cv.inds = model.sampler(model, cv.N, state.rng)
        end
    end
    cv.sub_dataset = @view(model.datamat[cv.inds,:])
    cv.sub_xp = Matrix(cv.sub_dataset')
    cv.weights = (model.N / cv.N) * ones(cv.N)
end

function update_estimator!(state::AbstractState, model::AbstractModel, cv::CoresetLogProbEstimator, 
                            θ′::Union{AbstractVector, Nothing}, μ0::Union{Float64, Nothing}, inds::Union{AbstractArray, Nothing})
    return 0.0
end

function update_inds!(cv::CoresetLogProbEstimator, inds::Array{Int64})
    return 0.0
end

function log_likelihood_cv!(state::AbstractState, model::AbstractModel, cv::CoresetLogProbEstimator)
    return 0.0
end
function grad_log_likelihood_cv!(state::AbstractState, model::AbstractModel, cv::CoresetLogProbEstimator)
    return 0.0
end
function hess_log_likelihood_cv!(state::AbstractState, model::AbstractModel, cv::CoresetLogProbEstimator) 
    return 0.0 * I
end
function log_likelihood_cv_array!(state::AbstractState, model::AbstractModel, cv::CoresetLogProbEstimator)
    return zeros(cv.N)
end

function log_likelihood_diff!(state::AbstractState, model::AbstractModel, cv::CoresetLogProbEstimator)
    logliks = log_likelihood_array(state, model, cv)
    return sum(logliks .* cv.weights)
end
function grad_log_likelihood_diff!(state::AbstractState, model::AbstractModel, cv::CoresetLogProbEstimator)
    grad_logliks = grad_log_likelihood_array(state, model, cv)
    return sum(grad_logliks .* cv.weights)
end
function hess_log_likelihood_diff!(state::AbstractState, model::AbstractModel, cv::CoresetLogProbEstimator)
    hess_logliks = hess_log_likelihood_array(state, model, cv)
    return sum(hess_logliks .* cv.weights)
end
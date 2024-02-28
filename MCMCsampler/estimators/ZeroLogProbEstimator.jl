@with_kw mutable struct ZeroLogProbEstimator <: SizeBasedLogProbEstimator
    N::Int64
    inds::Union{AbstractArray, Nothing} = nothing
    sub_dataset::Union{AbstractArray, Nothing} = nothing
    sub_xp::Union{AbstractArray, Nothing} = nothing
    weights::Union{AbstractArray, Nothing} = nothing
end

function update_estimator!(state::AbstractState, model::AbstractModel, cv::ZeroLogProbEstimator, 
                            θ′::Union{AbstractVector, Nothing}, μ0::Union{Float64, Nothing}, inds::Union{AbstractArray, Nothing})
    if isnothing(cv.inds) || (cv.N != model.N)
        Zygote.ignore() do
            cv.inds = isnothing(inds) ? sample(state.rng, [1:model.N;], cv.N, replace = false, ordered = true) : inds
            cv.sub_dataset = @view(model.datamat[cv.inds,:])
            cv.sub_xp = Matrix(cv.sub_dataset')
            cv.weights = ones(cv.N)
        end
    end
end

function update_inds!(cv::ZeroLogProbEstimator, inds::Array{Int64})
    if length(inds) != cv.N
        error("inds size does not match that specified by the estimator")
    end
    cv.inds = inds
end

function log_likelihood_cv!(state::AbstractState, model::AbstractModel, cv::ZeroLogProbEstimator)
    return 0.0
end
function grad_log_likelihood_cv!(state::AbstractState, model::AbstractModel, cv::ZeroLogProbEstimator)
    return 0.0
end
function hess_log_likelihood_cv!(state::AbstractState, model::AbstractModel, cv::ZeroLogProbEstimator) 
    return 0.0 * I
end
function log_likelihood_cv_array!(state::AbstractState, model::AbstractModel, cv::ZeroLogProbEstimator)
    return zeros(cv.N)
end

function log_likelihood_diff!(state::AbstractState, model::AbstractModel, cv::ZeroLogProbEstimator)
    state.lp_evals += length(cv.inds)
    return (model.N ./ cv.N) * mapreduce(x->log_likelihood(x, state.θ, model), 
                                                        +, @view(model.dataset[cv.inds]))
end
function grad_log_likelihood_diff!(state::AbstractState, model::AbstractModel, cv::ZeroLogProbEstimator)
    state.grad_lp_evals += length(cv.inds)
    return (model.N ./ cv.N) * mapreduce(x->grad_log_likelihood(x, state.θ, model), 
                                                        +, @view(model.dataset[cv.inds]))
end
function hess_log_likelihood_diff!(state::AbstractState, model::AbstractModel, cv::ZeroLogProbEstimator)
    state.hess_lp_evals += length(cv.inds)
    return (model.N ./ cv.N) * mapreduce(x->hess_log_likelihood(x, state.θ, model), 
                                                        +, @view(model.dataset[cv.inds]))
end
@with_kw mutable struct AusterityLogProbEstimator <: QualityBasedLogProbEstimator
    inds::Union{AbstractArray, Nothing} = nothing
    ϵ::Float64 = 0.05
    batch_size::UInt64 = 500
end

function update_estimator!(state::AbstractState, model::AbstractModel, cv::AusterityLogProbEstimator,
                            θ′::AbstractVector, μ0::Float64, inds::Union{AbstractArray, Nothing})
    lsum, l2sum, n, done, θ0, N = 0., 0., 0, false, copy(state.θ), model.N
    cv.inds = []
    remaining_inds = [1:N;]
    while !done
        b = min(cv.batch_size, length(remaining_inds))
        new_batch = sample(state.rng, remaining_inds, b, replace = false)
        cv.inds = vcat(cv.inds, new_batch)
        setdiff!(remaining_inds, new_batch)
        n += b
        copy_inds = copy(cv.inds)
        cv.inds = new_batch
        lp = log_likelihood_array(state, model, cv)
        state.θ = θ′
        lp′ = log_likelihood_array(state, model, cv)
        state.θ = θ0
        cv.inds = copy_inds
        lsum += sum(lp′ .- lp)
        l2sum += sum((lp′ .- lp).^2)
        sl = sqrt(((l2sum/n)-(lsum/n)^2)*n/(n-1))
        s = sl/sqrt(n)*sqrt(1-((n-1)/(N-1)))
        t = ((lsum/n)-μ0)/s
        δ = 1 - cdf(TDist(n-1), abs(t))
        if δ < cv.ϵ
            done = true
        end
    end
end

function log_likelihood_cv!(state::AbstractState, model::AbstractModel, cv::AusterityLogProbEstimator)
    return 0.0
end
function grad_log_likelihood_cv!(state::AbstractState, model::AbstractModel, cv::AusterityLogProbEstimator)
    return 0.0
end
function hess_log_likelihood_cv!(state::AbstractState, model::AbstractModel, cv::AusterityLogProbEstimator) 
    return 0.0 * I
end
function log_likelihood_cv_array!(state::AbstractState, model::AbstractModel, cv::AusterityLogProbEstimator)
    return zeros(length(cv.inds))
end

function log_likelihood_diff!(state::AbstractState, model::AbstractModel, cv::AusterityLogProbEstimator)
    state.lp_evals += length(cv.inds)
    return (model.N ./ length(cv.inds)) * mapreduce(x->log_likelihood(x, state.θ, model), 
                                                        +, @view(model.dataset[cv.inds]))
end
function grad_log_likelihood_diff!(state::AbstractState, model::AbstractModel, cv::AusterityLogProbEstimator)
    state.grad_lp_evals += length(cv.inds)
    return (model.N ./ length(cv.inds)) * mapreduce(x->grad_log_likelihood(x, state.θ, model), 
                                                        +, @view(model.dataset[cv.inds]))
end
function hess_log_likelihood_diff!(state::AbstractState, model::AbstractModel, cv::AusterityLogProbEstimator)
    state.hess_lp_evals += length(cv.inds)
    return (model.N ./ length(cv.inds)) * mapreduce(x->hess_log_likelihood(x, state.θ, model), 
                                                        +, @view(model.dataset[cv.inds]))
end
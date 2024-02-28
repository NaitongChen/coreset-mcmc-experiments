@with_kw mutable struct ConfidenceLogProbEstimator <: QualityBasedLogProbEstimator
    inds::Union{AbstractArray, Nothing} = nothing
    δ::Float64 = 0.05
    p::Float64 = 2. # p>1
    γ::Float64 = 2.
    init_batch_size::UInt64 = 1 # less than dataset size
end

function update_estimator!(state::AbstractState, model::AbstractModel, cv::ConfidenceLogProbEstimator,
                            θ′::AbstractVector, μ0::Float64, inds::Union{AbstractArray, Nothing})
    done, θ0, N, b, n, Λ, iter = false, copy(state.θ), model.N, cv.init_batch_size, 0, 0., 0
    cv.inds = []
    remaining_inds = [1:N;]
    while !done
        iter += 1
        new_batch = sample(state.rng, remaining_inds, b-length(cv.inds), replace = false)
        cv.inds = vcat(cv.inds, new_batch)
        setdiff!(remaining_inds, new_batch)
        copy_inds = copy(cv.inds)
        cv.inds = new_batch
        lp = log_likelihood_array(state, model, cv)
        state.θ = θ′
        lp′ = log_likelihood_array(state, model, cv)
        state.θ = θ0
        cv.inds = copy_inds
        @infiltrate length(abs.(lp′ .- lp)) == 0
        Λ = (n*Λ + sum(lp′ .- lp))/b
        n = b
        Cn = maximum(abs.(lp′ .- lp))
        δn = (cv.p-1)/(cv.p*(iter^cv.p))*cv.δ
        σn = std(lp′ .- lp)
        c = σn*sqrt(2*log(3/δn)/n) + (6*Cn*log(3/δn))/n
        b = min(N, Int(ceil(cv.γ*n)))
        if abs(Λ - μ0) >= c || b >= N
            done = true
        end
    end
end

function log_likelihood_cv!(state::AbstractState, model::AbstractModel, cv::ConfidenceLogProbEstimator)
    return 0.0
end
function grad_log_likelihood_cv!(state::AbstractState, model::AbstractModel, cv::ConfidenceLogProbEstimator)
    return 0.0
end
function hess_log_likelihood_cv!(state::AbstractState, model::AbstractModel, cv::ConfidenceLogProbEstimator) 
    return 0.0 * I
end
function log_likelihood_cv_array!(state::AbstractState, model::AbstractModel, cv::ConfidenceLogProbEstimator)
    return zeros(length(cv.inds))
end

function log_likelihood_diff!(state::AbstractState, model::AbstractModel, cv::ConfidenceLogProbEstimator)
    state.lp_evals += length(cv.inds)
    return (model.N ./ length(cv.inds)) * mapreduce(x->log_likelihood(x, state.θ, model), 
                                                        +, @view(model.dataset[cv.inds]))
end
function grad_log_likelihood_diff!(state::AbstractState, model::AbstractModel, cv::ConfidenceLogProbEstimator)
    state.grad_lp_evals += length(cv.inds)
    return (model.N ./ length(cv.inds)) * mapreduce(x->grad_log_likelihood(x, state.θ, model), 
                                                        +, @view(model.dataset[cv.inds]))
end
function hess_log_likelihood_diff!(state::AbstractState, model::AbstractModel, cv::ConfidenceLogProbEstimator)
    state.hess_lp_evals += length(cv.inds)
    return (model.N ./ length(cv.inds)) * mapreduce(x->hess_log_likelihood(x, state.θ, model), 
                                                        +, @view(model.dataset[cv.inds]))
end
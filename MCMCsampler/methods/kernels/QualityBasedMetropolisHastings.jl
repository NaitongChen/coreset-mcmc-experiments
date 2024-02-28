@with_kw mutable struct QualityBasedMetropolisHastings <: AbstractKernel
    σ::Float64 = 1.
    propose::Function = (rng, θ) -> rand(rng, MvNormal(θ, σ * I))
    proposal_logpdf::Function = (θ, θ_given) -> logpdf(MvNormal(θ_given, σ * I), θ)
end

function step!(kernel::QualityBasedMetropolisHastings, state::AbstractState, model::AbstractModel, 
                cv::QualityBasedLogProbEstimator, iter::Int64)
    θ0 = copy(state.θ)
    θ′ = kernel.propose(state.rng, state.θ)
    u = rand(state.rng)
    μ0 = 1. / model.N * (log(u) + log_prior(state.θ, model) + kernel.proposal_logpdf(θ′, state.θ) 
                                        - log_prior(θ′, model) - kernel.proposal_logpdf(state.θ, θ′))
    update_estimator!(state, model, cv, θ′, μ0, nothing)
    lp = log_likelihood_sum(state, model, cv)
    state.θ = θ′
    lp′ = log_likelihood_sum(state, model, cv)
    state.θ = θ0
    lbar = (lp′ - lp)/length(cv.inds)
    if lbar > μ0
        state.θ = θ′
    end
end
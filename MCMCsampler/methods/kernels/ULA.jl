@with_kw mutable struct ULA <: AbstractKernel
    a::Float64 = 0.02
    b::Float64 = 0.
    γ::Float64 = 1. # (0.5, 1]
    adapt::Bool = 1 # if adapt == 0, step size is constant
end

function step!(kernel::ULA, state::AbstractState, model::AbstractModel, cv::AbstractLogProbEstimator, iter::Int64)
    update_estimator!(state, model, cv, nothing, nothing, nothing)
    ϵ = kernel.a*((kernel.b+iter)^(-kernel.γ))^kernel.adapt
    η = rand(state.rng, MvNormal(zeros(length(state.θ)), ϵ*I))
    state.θ = state.θ + ϵ/2*grad_log_potential(state, model, cv) + η
end
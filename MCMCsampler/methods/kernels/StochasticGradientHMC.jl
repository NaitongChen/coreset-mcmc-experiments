@with_kw mutable struct SGHMC <: AbstractKernel
    L::Int = 20
    C::Union{UniformScaling, Matrix{Float64}} = I
    B::Union{UniformScaling, Matrix{Float64}} = 0. * I
    # learning rate param
    a::Float64 = 0.01
    b::Float64 = 0.
    γ::Float64 = 1. # (0.5, 1]
    adapt::Bool = 1
end

function step!(kernel::SGHMC, state::AbstractState, model::AbstractModel, cv::AbstractLogProbEstimator, iter::Int64)
    r = randn(state.rng, length(state.θ))
    ϵ = kernel.a*(((kernel.b+iter)^(-kernel.γ))^kernel.adapt)
    for i=1:kernel.L
        update_estimator!(state, model, cv, nothing, nothing, nothing)
        state.θ += ϵ .* r
        r = (r + ϵ .* grad_log_potential(state, model, cv) - ϵ .* (kernel.C*r) + 
                rand(state.rng, MvNormal(zeros(length(state.θ)), 2*ϵ*(kernel.C-kernel.B))))
    end
end

# @with_kw mutable struct SGHMC <: AbstractKernel
#     L::Int = 5
#     η::Float64 = 0.01 # learning rate
#     α::Float64 = 0.01 # α = ϵ, assuming M = C = I
#     β::Float64 = 0.
# end

# function step!(kernel::SGHMC, state::AbstractState, model::AbstractModel, cv::AbstractLogProbEstimator, iter::Int64)
#     r = randn(state.rng, length(state.θ))
#     for i=1:kernel.L
#         update_estimator!(state, model, cv, nothing, nothing)
#         state.θ += kernel.α .* r
#         r = (r + kernel.η .* grad_log_potential(state, model, cv) - kernel.α * (kernel.α .* r) + 
#             sqrt(2 * (kernel.α - kernel.β) * kernel.η) .* randn(state.rng, length(state.θ)))
#     end
# end
# Laplace approximation
abstract type LaplaceApprox <: AbstractKernel end
@with_kw mutable struct LaplaceApproxDeterministic <: LaplaceApprox
    μ::Union{Nothing, Vector{Float64}} = nothing
    Σ::Union{Nothing, Matrix{Float64}, UniformScaling{Float64}} = nothing
    initialized::Bool = false
    # backtracking line search parameters
    c1::Float64 = 10^-4 
    c2::Float64 = 0.9 
    τ::Float64 = 0.7 
    tol::Float64 = 1e-3
end
@with_kw mutable struct LaplaceApproxStochastic <: LaplaceApprox
    μ::Union{Nothing, Vector{Float64}} = nothing
    Σ::Union{Nothing, Matrix{Float64}, UniformScaling{Float64}} = nothing
    initialized::Bool = false
    # ADAM parameters
    α::Float64 = 0.001
    β1::Float64 = 0.9
    β2::Float64 = 0.999
    ϵ::Float64 = 10^-8
    tol::Float64 = 1e-4
end

function step!(kernel::LaplaceApprox, state::AbstractState, model::AbstractModel, cv::AbstractLogProbEstimator, iter::Int64)
    if !kernel.initialized
        θ_init = copy(state.θ)
        find_map!(kernel, state, model, cv) # optimizes state.θ
        kernel.μ = copy(state.θ)
        hess = hess_log_potential(state, model, cv)
        kernel.Σ = -inv(hess)
        kernel.initialized = true
        state.θ = θ_init # resets state.θ.z to init value
    end
    state.θ = rand(state.rng, MvNormal(kernel.μ, kernel.Σ))
    return state.θ
end
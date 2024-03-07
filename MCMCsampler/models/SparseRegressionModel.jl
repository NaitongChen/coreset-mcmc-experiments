@with_kw struct SparseRegressionModel <: AbstractModel
    N::Int64
    d::Int64
    dataset::AbstractArray
    datamat::AbstractArray
    p::Float64
    ν::Float64
    λ::Float64
    c::Float64
    τ::Float64
    sampler::Union{Nothing, Function} = nothing
end
function init!(rng::AbstractRNG, model::SparseRegressionModel; init_val::Any = nothing)
    if isnothing(init_val)
        θ0 = zeros(2*model.d + 1)
        # [β1, ..., βd, γ1, ..., γd, σ2]
        # βs = θ[1:model.d]
        # γs = θ[(model.d+1):(2*model.d)]
        # σ2 = θ[end]
        X = model.datamat[:,1:end-1]
        y = model.datamat[:,end]
        θ0[1:model.d] = (X' * X) \ (X' * y)
        θ0[(model.d+1):(2*model.d)] .= 1.
        θ0[end] = var(X * θ0[1:model.d] - y)
    else
        θ0 = init_val
    end
    return State(θ = θ0, rng = rng)
end

function log_prior(θ, model::SparseRegressionModel)
    ret = logpdf(InverseGamma(model.ν/2, model.ν*model.λ/2), θ[end]) + sum(logpdf.(Bernoulli(model.p), θ[(model.d+1):(2*model.d)]))
    iszerovec = (θ[(model.d+1):(2*model.d)] .== 0)
    diag_entry = (iszerovec .* 1. + (1 .- iszerovec) .* model.c) .* model.τ
    Vβ = diagm(diag_entry.^2)
    ret += logpdf(MvNormal(zeros(model.d), Vβ), θ[1:model.d])
end

grad_log_prior(θ, model::SparseRegressionModel) = error("not implemented")
hess_log_prior(θ, model::SparseRegressionModel) = error("not implemented")

function log_likelihood(x, θ, model::SparseRegressionModel)
    return -0.5 * (log(θ[end]) + (x[1:end-1]' * θ[1:model.d] - x[end])^2 / θ[end])
end

grad_log_likelihood(x, θ, model::SparseRegressionModel) = error("not implemented")
hess_log_likelihood(x, θ, model::SparseRegressionModel) = error("not implemented")
data_grad_log_likelihood(x, θ, model::SparseRegressionModel) = error("not implemented")
data_hess_log_likelihood(x, θ, model::SparseRegressionModel) = error("not implemented")
grad_data_grad_log_likelihood(x, θ, model::SparseRegressionModel) = error("not implemented")
hess_data_grad_log_likelihood(x, θ, model::SparseRegressionModel) = error("not implemented")
grad_data_hess_log_likelihood(x, θ, model::SparseRegressionModel) = error("not implemented")
hess_data_hess_log_likelihood(x, θ, model::SparseRegressionModel) = error("not implemented")
grad_log_potential(state::AbstractState, model::SparseRegressionModel, cv::CoresetLogProbEstimator) = error("not implemented")
log_potential(state::AbstractState, model::SparseRegressionModel, cv::ZeroLogProbEstimator) = error("not implemented")

function log_likelihood_array(state::AbstractState, model::SparseRegressionModel, cv::SizeBasedLogProbEstimator)
    state.lp_evals += cv.N
    xs = @view(cv.sub_dataset[:,1:end-1])
    ys = @view(cv.sub_dataset[:,end])
    return -0.5 .* (log(state.θ[end]) .+ (xs * state.θ[1:model.d] .- ys).^2 ./ state.θ[end])
end
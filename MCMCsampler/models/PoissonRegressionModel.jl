@with_kw struct PoissonRegressionModel <: AbstractModel
    N::Int64
    dataset::AbstractArray # [1, x1, ..., xd, y]
    datamat::AbstractArray
    d::Int64 # number of covariates
    σ_prior::Float64
    sampler::Union{Nothing, Function} = nothing
end

function init!(rng::AbstractRNG, model::PoissonRegressionModel)
    θ0 = zeros(length(model.dataset[1])-1)
    # [β0, β1, ..., βd]
    return State(θ = θ0, rng = rng)
end

log_prior(θ, model::PoissonRegressionModel) = -0.5dot(θ, θ)/model.σ_prior^2
grad_log_prior(θ, model::PoissonRegressionModel) = -θ/model.σ_prior^2
log_likelihood(x, θ, model::PoissonRegressionModel) = x[end] * log(log1pexp(x[1:end-1]' * θ)) - log1pexp(x[1:end-1]' * θ)

function grad_log_likelihood(x, θ, model::PoissonRegressionModel) 
    lp = x[1:end-1]' * θ
    return ((x[end] / log1pexp(lp)) - 1) * logistic(lp) .* x[1:end-1]
end

hess_log_prior(θ, model::PoissonRegressionModel) = error("not implemented")
hess_log_likelihood(x, θ, model::PoissonRegressionModel) = error("not implemented")
data_grad_log_likelihood(x, θ, model::PoissonRegressionModel) = error("not implemented")
data_hess_log_likelihood(x, θ, model::PoissonRegressionModel) = error("not implemented")
grad_data_grad_log_likelihood(x, θ, model::PoissonRegressionModel) = error("not implemented")
hess_data_grad_log_likelihood(x, θ, model::PoissonRegressionModel) = error("not implemented")
grad_data_hess_log_likelihood(x, θ, model::PoissonRegressionModel) = error("not implemented")
hess_data_hess_log_likelihood(x, θ, model::PoissonRegressionModel) = error("not implemented")

function grad_log_potential(state::AbstractState, model::PoissonRegressionModel, cv::CoresetLogProbEstimator)
    state.grad_lp_evals += cv.N
    xs = @view(cv.sub_dataset[:,1:end-1])
    ys = @view(cv.sub_dataset[:,end])
    lps = xs * state.θ
    grad = vec(sum(((ys ./ log1pexp.(lps) .- 1) .* logistic.(lps) .* cv.weights) .* xs, dims=1))
    return grad_log_prior(state.θ, model) + grad
end

function log_potential(state::AbstractState, model::PoissonRegressionModel, cv::ZeroLogProbEstimator)
    state.lp_evals += cv.N
    xs = @view(cv.sub_dataset[:,1:end-1])
    ys = @view(cv.sub_dataset[:,end])
    prods = xs * state.θ
    temp = Zygote.Buffer(zeros(length(prods)))
    temp[:] = log.(log1pexp.(prods))
    temp[prods .< -700] = prods[prods .< -700]
    return (log_prior(state.θ, model) + (model.N ./ cv.N)*sum(ys .* copy(temp) .- log1pexp.(prods)))
end

function log_likelihood_array(state::AbstractState, model::PoissonRegressionModel, cv::SizeBasedLogProbEstimator)
    state.lp_evals += cv.N
    xs = @view(cv.sub_dataset[:,1:end-1])
    ys = @view(cv.sub_dataset[:,end])
    prods = xs * state.θ
    temp = log.(log1pexp.(prods))
    temp[prods .< -700] = prods[prods .< -700]
    return ys .* copy(temp) .- log1pexp.(prods)
end
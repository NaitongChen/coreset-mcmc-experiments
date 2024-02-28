@with_kw struct LogisticRegressionModel <: AbstractModel
    N::Int64
    dataset::AbstractArray # [1, x1, ..., xd, y]
    datamat::AbstractArray
    d::Int64 # number of covariates
    sampler::Union{Nothing, Function} = nothing
end
function init!(rng::AbstractRNG, model::LogisticRegressionModel; init_val::Any = nothing)
    if isnothing(init_val)
        θ0 = zeros(model.d+1)
    else
        θ0 = init_val
    end
    # [β0, β1, ..., βd]
    # βs = θ[1:model.d+1]
    return State(θ = θ0, rng = rng)
end
log_prior(θ, model::LogisticRegressionModel) = -sum(log1p.(θ.^2))
grad_log_prior(θ, model::LogisticRegressionModel) = -2*θ ./ (1 .+ θ.^2)
log_likelihood(x, θ, model::LogisticRegressionModel) = x[end]*log_logistic(θ'*x[1:end-1])+(1-x[end])*log_logistic(-θ'*x[1:end-1])
grad_log_likelihood(x, θ, model::LogisticRegressionModel) = (x[end]-logistic(x[1:end-1]'*θ)).*x[1:end-1]

hess_log_prior(θ, model::LogisticRegressionModel) = error("not implemented")
hess_log_likelihood(x, θ, model::LogisticRegressionModel) = error("not implemented")
data_grad_log_likelihood(x, θ, model::LogisticRegressionModel) = error("not implemented")
data_hess_log_likelihood(x, θ, model::LogisticRegressionModel) = error("not implemented")
grad_data_grad_log_likelihood(x, θ, model::LogisticRegressionModel) = error("not implemented")
hess_data_grad_log_likelihood(x, θ, model::LogisticRegressionModel) = error("not implemented")
grad_data_hess_log_likelihood(x, θ, model::LogisticRegressionModel) = error("not implemented")
hess_data_hess_log_likelihood(x, θ, model::LogisticRegressionModel) = error("not implemented")

function grad_log_potential(state::AbstractState, model::LogisticRegressionModel, cv::CoresetLogProbEstimator)
    state.grad_lp_evals += cv.N
    xs = @view(cv.sub_dataset[:,1:end-1])
    ys = @view(cv.sub_dataset[:,end])
    Pfit = -logistic.(xs*state.θ) 
    grad = vec(sum(((Pfit .+ ys) .* cv.weights) .* xs, dims=1))
    return grad_log_prior(state.θ, model) + grad
end

function log_potential(state::AbstractState, model::LogisticRegressionModel, cv::ZeroLogProbEstimator)
    state.lp_evals += cv.N
    # datamat = reduce(hcat, model.dataset[cv.inds])'
    xs = @view(cv.sub_dataset[:,1:end-1])
    ys = @view(cv.sub_dataset[:,end])
    return (log_prior(state.θ, model) + 
            (model.N ./ cv.N) * sum(ys .* log_logistic.(xs * state.θ) .+ (1 .- ys) .* log_logistic.(-xs*state.θ)))

end

function log_likelihood_array(state::AbstractState, model::LogisticRegressionModel, cv::SizeBasedLogProbEstimator)
    state.lp_evals += cv.N
    # datamat = reduce(hcat, model.dataset[cv.inds])'
    xs = @view(cv.sub_dataset[:,1:end-1])
    ys = @view(cv.sub_dataset[:,end])
    return ys .* log_logistic.(xs * state.θ) .+ (1 .- ys) .* log_logistic.(-xs*state.θ)
end
@with_kw struct LinearRegressionModel <: AbstractModel
    N::Int64
    dataset::AbstractArray # [1, x1, ..., xd, y]
    datamat::AbstractArray
    d::Int64 # number of covariates
    σ_prior::Float64
    μ_prior::Vector{Float64}
    sampler::Union{Nothing, Function} = nothing
end
function init!(rng::AbstractRNG, model::LinearRegressionModel; init_val::Any = nothing)
    if isnothing(init_val)
        θ0 = 10 * ones(length(model.dataset[1]))
        θ0[1:end-1] .= 0.
        # [β0, β1, ..., βd, logσ2]
        # βs = θ[1:model.d+1]
        # logσ2 = θ[end]
    else
        θ0 = init_val
    end
    return State(θ = θ0, rng = rng)
end
log_prior(θ, model::LinearRegressionModel) = -0.5dot(θ-model.μ_prior, θ-model.μ_prior)/model.σ_prior^2
grad_log_prior(θ, model::LinearRegressionModel) = -(θ-model.μ_prior)/model.σ_prior^2
log_likelihood(x, θ, model::LinearRegressionModel) = -0.5*θ[end]-(1/(2*exp(θ[end])))*((θ[1:end-1]'*x[1:end-1])-x[end])^2
function grad_log_likelihood(x, θ, model::LinearRegressionModel) 
    diff = x[end]-θ[1:end-1]'*x[1:end-1]
    return vcat(diff.*x[1:end-1]./exp(θ[end]), 0.5*exp(-θ[end])*(diff)^2-0.5)
end

hess_log_prior(θ, model::LinearRegressionModel) = error("not implemented")
hess_log_likelihood(x, θ, model::LinearRegressionModel) = error("not implemented")
data_grad_log_likelihood(x, θ, model::LinearRegressionModel) = error("not implemented")
data_hess_log_likelihood(x, θ, model::LinearRegressionModel) = error("not implemented")
grad_data_grad_log_likelihood(x, θ, model::LinearRegressionModel) = error("not implemented")
hess_data_grad_log_likelihood(x, θ, model::LinearRegressionModel) = error("not implemented")
grad_data_hess_log_likelihood(x, θ, model::LinearRegressionModel) = error("not implemented")
hess_data_hess_log_likelihood(x, θ, model::LinearRegressionModel) = error("not implemented")

function grad_log_potential(state::AbstractState, model::LinearRegressionModel, cv::CoresetLogProbEstimator)
    state.grad_lp_evals += cv.N
    xs = @view(cv.sub_dataset[:,1:end-1])
    ys = @view(cv.sub_dataset[:,end])
    grads_p = Zygote.Buffer(zeros(length(state.θ)))
    diffs = ys .- xs*state.θ[1:end-1]
    s = vec(sum((diffs .* cv.weights) .* xs, dims=1))
    grads_p[1:length(state.θ)-1] = s ./ exp(state.θ[end])
    t = sum(diffs.^2 .* cv.weights)
    grads_p[length(state.θ)] = 0.5 * exp(-state.θ[end]) * t - 0.5 * sum(cv.weights)
    return grad_log_prior(state.θ, model) + copy(grads_p)
end

function log_potential(state::AbstractState, model::LinearRegressionModel, cv::ZeroLogProbEstimator)
    state.lp_evals += cv.N
    # datamat = reduce(hcat, model.dataset[cv.inds])'
    xs = @view(cv.sub_dataset[:,1:end-1])
    ys = @view(cv.sub_dataset[:,end])
    return (log_prior(state.θ, model)
            -0.5*(model.N ./ cv.N)*sum((state.θ[end] .+ 
                                        (xs * state.θ[1:model.d+1] .- ys).^2 ./exp(state.θ[end]))))
end

function log_likelihood_array(state::AbstractState, model::LinearRegressionModel, cv::SizeBasedLogProbEstimator)
    state.lp_evals += cv.N
    # datamat = reduce(hcat, model.dataset[cv.inds])'
    xs = @view(cv.sub_dataset[:,1:end-1])
    ys = @view(cv.sub_dataset[:,end])
    return -0.5 .* (state.θ[end] .+ (xs * state.θ[1:model.d+1] .- ys).^2 ./ exp(state.θ[end]))
end
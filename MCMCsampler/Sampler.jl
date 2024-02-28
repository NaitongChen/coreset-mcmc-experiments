# general API for adaptive sampling algorithms

function sample!(kernel::AbstractKernel, model::AbstractModel, cv::AbstractLogProbEstimator, 
                    n_samples::Int64, rng::AbstractRNG; init_val::Any = nothing)
    state = init!(rng, model; init_val)
    return sample!(kernel, state, model, cv, n_samples)
end

function sample!(alg::AbstractAlgorithm, model::AbstractModel, cv::CoresetLogProbEstimator, 
                    n_samples::Int64, rng::AbstractRNG; init_val::Any = nothing)
    metaState = MetaState()
    for i in [1:alg.replicas;]
        push!(metaState.states, init!(Xoshiro(abs(rand(rng, Int))), model; init_val))
    end
    return sample!(alg, metaState, model, cv, n_samples)
end

function sample!(kernel::AbstractKernel, state::AbstractState, model::AbstractModel, 
                    cv::AbstractLogProbEstimator, n_samples::Int64)
    θs = Array{typeof(state.θ)}(undef, n_samples)
    cumulative_lp_evals = Array{Int64}(undef, n_samples)
    cumulative_grad_lp_evals = Array{Int64}(undef, n_samples)
    cumulative_hess_lp_evals = Array{Int64}(undef, n_samples)
    cumulative_time = Array{Float64}(undef, n_samples)
    @showprogress for i=1:n_samples
        t = @elapsed step!(kernel, state, model, cv, i)
        θs[i] = copy(state.θ)
        cumulative_time[i] = (i == 1 ? t : cumulative_time[i-1]+t)
        cumulative_lp_evals[i] = state.lp_evals
        cumulative_grad_lp_evals[i] = state.grad_lp_evals
        cumulative_hess_lp_evals[i] = state.hess_lp_evals
    end
    return θs, cumulative_lp_evals, cumulative_grad_lp_evals, cumulative_hess_lp_evals, cumulative_time
end

function sample!(alg::QuasiNewtonCoreset, metaState::AbstractMetaState, model::AbstractModel, 
                    cv::CoresetLogProbEstimator, n_samples::Int64)
    θs = Vector{typeof(metaState.states[1].θ)}(undef, 0)
    weights = Vector{Vector{Float64}}(undef, 0)
    cumulative_lp_evals = Array{Int64}(undef, n_samples)
    cumulative_grad_lp_evals = Array{Int64}(undef, n_samples)
    cumulative_hess_lp_evals = Array{Int64}(undef, n_samples)
    cumulative_time = Array{Float64}(undef, n_samples)
    @showprogress for i=1:n_samples
        t = @elapsed step!(alg, metaState, model, cv, i)
        cumulative_time[i] = (i == 1 ? t : cumulative_time[i-1]+t)
        cumulative_lp_evals[i] = mapreduce(x -> x.lp_evals, +, metaState.states)
        cumulative_grad_lp_evals[i] = mapreduce(x -> x.grad_lp_evals, +, metaState.states)
        cumulative_hess_lp_evals[i] = mapreduce(x -> x.hess_lp_evals, +, metaState.states)
        for j in [1:alg.replicas;]
            push!(θs, copy(metaState.states[j].θ))
        end
        push!(weights, copy(cv.weights))
    end
    return θs, cumulative_lp_evals, cumulative_grad_lp_evals, cumulative_hess_lp_evals, cumulative_time, weights
end

function sample!(alg::CoresetMCMC, metaState::AbstractMetaState, model::AbstractModel, 
    cv::CoresetLogProbEstimator, n_samples::Int64)
    θs = Vector{typeof(metaState.states[1].θ)}(undef, 0)
    weights = Vector{Vector{Float64}}(undef, 0)
    overall_steps = alg.train_iter * alg.delay + Int(ceil(n_samples / alg.replicas))
    cumulative_lp_evals = Array{Int64}(undef, overall_steps)
    cumulative_grad_lp_evals = Array{Int64}(undef, overall_steps)
    cumulative_hess_lp_evals = Array{Int64}(undef, overall_steps)
    cumulative_time = Array{Float64}(undef, overall_steps)
    @showprogress for i=1:overall_steps
        t = @elapsed step!(alg, metaState, model, cv, i)
        cumulative_time[i] = (i == 1 ? t : cumulative_time[i-1]+t)
        cumulative_lp_evals[i] = mapreduce(x -> x.lp_evals, +, metaState.states)
        cumulative_grad_lp_evals[i] = mapreduce(x -> x.grad_lp_evals, +, metaState.states)
        cumulative_hess_lp_evals[i] = mapreduce(x -> x.hess_lp_evals, +, metaState.states)
        for j in [1:alg.replicas;]
            push!(θs, copy(metaState.states[j].θ))
        end
        push!(weights, copy(cv.weights))
    end
    return θs, cumulative_lp_evals, cumulative_grad_lp_evals, cumulative_hess_lp_evals, cumulative_time, weights
end

function sample!(kernel::AbstractKernel, state::AbstractState, model::AbstractModel, 
                    cv::CoresetLogProbEstimator, n_samples::Int64)
    θs = Array{typeof(state.θ)}(undef, n_samples)
    weights = Vector{Vector{Float64}}(undef, 0)
    cumulative_lp_evals = Array{Int64}(undef, n_samples)
    cumulative_grad_lp_evals = Array{Int64}(undef, n_samples)
    cumulative_hess_lp_evals = Array{Int64}(undef, n_samples)
    cumulative_time = Array{Float64}(undef, n_samples)
    @showprogress for i=1:n_samples
        t = @elapsed step!(kernel, state, model, cv, i)
        θs[i] = copy(state.θ)
        cumulative_time[i] = (i == 1 ? t : cumulative_time[i-1]+t)
        cumulative_lp_evals[i] = state.lp_evals
        cumulative_grad_lp_evals[i] = state.grad_lp_evals
        cumulative_hess_lp_evals[i] = state.hess_lp_evals
        push!(weights, copy(cv.weights))
    end
    return θs, cumulative_lp_evals, cumulative_grad_lp_evals, cumulative_hess_lp_evals, cumulative_time, weights
end

function sample!(kernel::SHF, state::AbstractState, model::AbstractModel, 
                    cv::CoresetLogProbEstimator, n_samples::Int64)
    θs = Array{typeof(state.θ)}(undef, n_samples)
    weights = Vector{Vector{Float64}}(undef, 0)
    cumulative_lp_evals = Array{Int64}(undef, n_samples)
    cumulative_grad_lp_evals = Array{Int64}(undef, n_samples)
    cumulative_hess_lp_evals = Array{Int64}(undef, n_samples)
    cumulative_time = Array{Float64}(undef, n_samples)
    @showprogress for i=1:n_samples
        t = @elapsed step!(kernel, state, model, cv, i)
        θs[i] = copy(state.θ)
        cumulative_time[i] = (i == 1 ? t : cumulative_time[i-1]+t)
        cumulative_lp_evals[i] = state.lp_evals
        cumulative_grad_lp_evals[i] = state.grad_lp_evals
        cumulative_hess_lp_evals[i] = state.hess_lp_evals
        push!(weights, copy(cv.weights))
    end
    return θs, cumulative_lp_evals, cumulative_grad_lp_evals, cumulative_hess_lp_evals, cumulative_time, weights, kernel.ls
end
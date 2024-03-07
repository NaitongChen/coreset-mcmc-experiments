function joint_log_potential(state::AbstractState, r::AbstractVector, 
								model::AbstractModel, cv::AbstractLogProbEstimator)
	return log_potential(state, model, cv) - 0.5*norm(r)^2
end

function leapfrog!(kernel::SHF, state::AbstractState, r::AbstractVector, 
					model::AbstractModel, cv::AbstractLogProbEstimator)
	ϵ = exp.(kernel.logϵ)
	r += 0.5 * ϵ .* grad_log_potential(state, model, cv)
	state.θ += ϵ .* r
	r += 0.5 * ϵ .* grad_log_potential(state, model, cv)
	return state.θ, r
end

function find_map!(state::AbstractState, model::AbstractModel, cv::Union{ModeLogProbEstimator})
	@unpack α, β1, β2, ϵ, tol = cv
	cv_zero = ZeroLogProbEstimator(N = cv.mode_n)
	update_estimator!(state, model, cv_zero, nothing, nothing, nothing)
	m = -grad_log_potential(state, model, cv_zero)
	v = m .^2
	t = 0 
	while true
		t += 1
		update_estimator!(state, model, cv_zero, nothing, nothing, nothing)
		g = -grad_log_potential(state, model, cv_zero)
		m = β1 * m + (1-β1)*g
		v = β2 * v + (1-β2)*g.^2
		m̂ = m/(1-β1^t)
		v̂ = v/(1-β2^t)
		curr_state = copy(state.θ)
		state.θ -= α* m̂./(sqrt.(v̂) .+ ϵ)
		state_diffs = abs.(α* m̂./(sqrt.(v̂) .+ ϵ)) ./ abs.(curr_state)
		if all(state_diffs .≤ tol)
			return state.θ
		end
	end
end

function find_map!(kernel::LaplaceApproxDeterministic, state::AbstractState, 
					model::AbstractModel, cv::AbstractLogProbEstimator)
	# https://en.wikipedia.org/wiki/Wolfe_conditions
	update_estimator!(state, model, cv, nothing, nothing, nothing)
	f = -log_potential(state, model, cv)
	g = grad_log_potential(state, model, cv)
	state_diffs = ones(size(state.θ,1))
	@unpack c1, c2, τ, tol = kernel
	while true
		α = 1.0
		while true
			curr_state = copy(state.θ)
			state.θ = curr_state + α*g
			fa = -log_potential(state, model, cv)
			ga = grad_log_potential(state, model, cv)
			if (fa ≤ f + c1*α*dot(g, -g)) && abs(dot(g, -ga)) ≤ c2*abs(dot(g,-g))
				state_diffs = abs.(curr_state - state.θ) ./ abs.(curr_state)
				f = fa
				g = ga
				break
			else
				α *= τ
				state.θ = curr_state
			end
		end
		if all(state_diffs .≤ tol)
			return state.θ
		end
	end
end

function find_map!(kernel::LaplaceApproxStochastic, state::AbstractState, 
					model::AbstractModel, cv::AbstractLogProbEstimator)
	@unpack α, β1, β2, ϵ, tol = kernel
	update_estimator!(state, model, cv, nothing, nothing, nothing)
	m = -grad_log_potential(state, model, cv)
	v = m .^2
	t = 0 
	while true
		t += 1
		update_estimator!(state, model, cv, nothing, nothing, nothing)
		g = -grad_log_potential(state, model, cv)
		m = β1 * m + (1-β1)*g
		v = β2 * v + (1-β2)*g.^2
		m̂ = m/(1-β1^t)
		v̂ = v/(1-β2^t)
		curr_state = copy(state.θ)
		state.θ -= α* m̂./(sqrt.(v̂) .+ ϵ)
		state_diffs = abs.(α* m̂./(sqrt.(v̂) .+ ϵ)) ./ abs.(curr_state)
		if all(state_diffs .≤ tol)
			return state.θ
		end
	end
end

function project(metaState::AbstractMetaState, model::AbstractModel, cv::SizeBasedLogProbEstimator)
    proj = zeros(cv.N, length(metaState.states))
    Threads.@threads for i in [1:length(metaState.states);]
		proj[:, i] = log_likelihood_array(metaState.states[i], model, cv)
    end
    proj .-=  mean(proj, dims=2)
    return proj
end

function project(metaState::AbstractMetaState, model::AbstractModel, cv::SizeBasedLogProbEstimator, θs::AbstractArray)
    θ0 = copy(metaState.states[1].θ)
	proj = zeros(cv.N, length(θs))
    for i in [1:length(θs);]
		metaState.states[1].θ = θs[i]
		proj[:, i] = log_likelihood_array(metaState.states[1], model, cv)
    end
    proj .-=  mean(proj, dims=2)
	metaState.states[1].θ = θ0
    return proj
end

function project_sum(metaState::AbstractMetaState, model::AbstractModel, cv_zero::SizeBasedLogProbEstimator, cv::CoresetLogProbEstimator)
    proj = zeros(cv_zero.N, length(metaState.states))
	if cv_zero.N != model.N
		if isnothing(cv_zero.inds_set)
			if !isnothing(model.sampler)
				cv_zero.inds_set = [1:model.N;]
				cv_zero.total_size = model.N
				cv_zero.current_location = 1
				cv_zero.inds_length = cv_zero.N
			else
				cv_zero.inds_set = setdiff([1:model.N;], cv.inds)
				cv_zero.total_size = model.N - cv.N
				cv_zero.current_location = 1
				cv_zero.inds_length = cv_zero.N - cv.N
			end
		end

		if cv_zero.current_location + cv_zero.inds_length - 1 <= cv_zero.total_size
			inds = cv_zero.inds_set[cv_zero.current_location:(cv_zero.current_location + cv_zero.inds_length - 1)]
			if cv_zero.current_location + cv_zero.inds_length <= cv_zero.total_size
				cv_zero.current_location = cv_zero.current_location + cv_zero.inds_length
			else
				cv_zero.current_location = 1
			end
		else
			inds = cv_zero.inds_set[cv_zero.current_location:end]
			l = length(inds)
			inds = vcat(cv_zero.inds_set[1:(cv_zero.inds_length - l)], inds)
			cv_zero.current_location = cv_zero.inds_length - l + 1
		end

		if isnothing(model.sampler)
			inds = vcat(cv.inds, inds)
		end
		update_estimator!(metaState.states[1], model, cv_zero, nothing, nothing, inds)
	end
    Threads.@threads for i in [1:length(metaState.states);]
		proj[:, i] = (model.N / cv_zero.N) * log_likelihood_array(metaState.states[i], model, cv_zero)
    end
    proj .-=  mean(proj, dims=2)
    return vec(sum(proj, dims=1))
end

function project_sum(metaState::AbstractMetaState, model::AbstractModel, cv::SizeBasedLogProbEstimator, θs::AbstractArray)
    θ0 = copy(metaState.states[1].θ)
	proj = zeros(model.N, length(θs))
    for i in [1:length(θs);]
		metaState.states[1].θ = θs[i]
		proj[:, i] = log_likelihood_array(metaState.states[1], model, cv)
    end
    proj .-=  mean(proj, dims=2)
	metaState.states[1].θ = θ0
    return vec(sum(proj, dims=1))
end

function kmeans!(data::Vector{T}, K::Int64, rng::AbstractRNG) where {T}
	N = length(data)
	centres = Array{T}(undef, K)
	labels = fill(1, N)

	# Kmeans++
	centres[1] = data[rand(rng, 1:N)]
	mindists = [evaluate(Euclidean(), centres[1], data[i]) for i=1:N]
	for k=2:K
		centres[k] = data[rand(rng, Categorical(mindists.^2/sum(mindists.^2)))]
		new_mindists = min.([evaluate(Euclidean(), centres[k], data[i]) for i=1:N], mindists)
		labels[.!isapprox.(new_mindists, mindists)] .= k
		mindists = new_mindists
	end

	# Kmeans
	moved = true
	while moved
		moved = false

		# update centres
		for k=1:K
			clus_k = (labels .== k)
			n_k = sum(clus_k)
			new_center = sum(data[clus_k])/n_k
			if !isapprox(evaluate(Euclidean(), centres[k], new_center), 0.0)
				moved = true
			end
			centres[k] = new_center
		end

		# update labels
		labels = [argmin([evaluate(Euclidean(), centres[k], data[i]) for k=1:K]) for i=1:N]
	end

	counts = zeros(Int, K)
	for k=1:K
		counts[k] = sum(labels .== k)
	end

	return centres, labels, counts
end

function log_logistic(a::Real)
    return -log1pexp(-a)
end

function ordered(a::AbstractArray, coef::Real)
    return coef .* reverse(cumprod(vcat(1.0, logistic.(a))))
end

function neg_sigmoid(x::Real)
    return -1.0/(1.0 + exp(-x))
end

function log_sigmoid(x::Real)
    if x < -300
        return x
    else
        return -log1p(exp(-x))
    end
end

# mutable struct StreamedUrn{T}
# 	items::Vector{T}
# 	index::Uint
# end

# StreamedUrn(items) = StreamedUrn(items, 0)

# # TODO add rng input
# function sample!(stream::StreamedUrn{T}, n, rng::AbstractRNG) where T
# 	N = length(stream.items)
# 	if n > N - stream.index
# 		@warn "Cannot request more stream items (n=$n) than are available (remaining=$(N-stream.index)); returning max available"
# 		n = N - stream.index
# 	end
# 	ret = Vector{T}(undef, n)
#     for i=1:n
#     	stream.index += 1
# 		j = rand(rng, stream.index:N)
# 		ret[i] = stream.items[j]
# 		stream.items[j] = stream.items[stream.index]
#     end
#     return ret
# end
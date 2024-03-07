using CSV
using DataFrames
using Random
using JLD
using Statistics
using LinearAlgebra
using Distributions
include("../MCMCsampler/MCMCsampler.jl")
include("../util.jl")

function main(args)
    data = JLD.load("../data/sparse_regression_50000.jld")["data"]

    N = length(data)
    D = 10
    
    @assert length(args) == 4 "Error: script has 4 mandatory cmd line args"

    # Initialize the rng
    println("Initializing RNG")
    rng = Xoshiro(parse(Int, args[1])) 

    # Create the model
    println("Initializing model")
    datamat = Matrix(reduce(hcat, data)')
    model = MCMCsampler.SparseRegressionModel(length(data), D, data, datamat, 0.1, 0.1, 1, 10, 1/10, nothing)

    # parse number of samples
    n_samples = parse(Int, args[2])

    # Create the algorithm
    println("Initializing sampler")
    sizes = parse.(Float64, split(args[3], "_"))
    prop_sample = (r, θ) -> proposal_sample(r, θ, sizes[1], sizes[2])
    prop_logpdf = (r, θ) -> proposal_logpdf(r, θ, sizes[1], sizes[2])
    kernel = MCMCsampler.QualityBasedMetropolisHastings(σ = sizes[1], propose = prop_sample, proposal_logpdf = prop_logpdf)
    cv = MCMCsampler.ConfidenceLogProbEstimator(δ = 0.05)

    println("Running sampler")
    θs, c_lp, c_g_lp, c_h_lp, c_time = MCMCsampler.sample!(kernel, model, cv, 2*n_samples, rng)
    println(sum(θs[end-n_samples+1:end]) / length(θs[end-n_samples+1:end]))
    ts = reduce(hcat, θs)'[end-n_samples+1:end,:]
    D_stan = JLD.load("../stan_results/sparse_regression_results_50000.jld")["θs"]
    m_method = vec(mean(ts[:,vcat(1:10, 21)], dims=1))
    v_method = cov(ts[:,vcat(1:10, 21)])
    kl_est = kl_gaussian(m_method, v_method, vec(mean(D_stan[:,vcat(1:10, 21)], dims=1)), cov(D_stan[:,vcat(1:0, 21)]))
    println(c_lp[end])
    println(c_time[end])

    save("sparse_regression_confidence_" * args[3] * "_" * args[1] * ".jld", "θs", θs, "c_lp", c_lp, "c_g_lp", c_g_lp,
                                                    "c_h_lp", c_h_lp, "c_time", c_time,
                                                    "kl", kl_est, "mean", m_method, "cov", v_method)
end

function proposal_sample(rng, θ, σ, p)
    θnew = zeros(length(θ))
    θnew[1:10] = rand(rng, MvNormal(θ[1:10], σ * I))
    θnew[end] = rand(rng, truncated(Normal(θ[end], σ), lower=0))
    θnew[11:20] = rand(rng, Bernoulli(p), 10)
    return θnew
end

function proposal_logpdf(θ, θ_given, σ, p)
    return logpdf(MvNormal(θ_given[1:10], σ * I), θ[1:10]) + sum(logpdf.(Bernoulli(p), θ[11:20])) + logpdf(truncated(Normal(θ_given[end], σ), lower=0), θ[end])
end

println("Running sampler with $(ARGS)")
main(ARGS)
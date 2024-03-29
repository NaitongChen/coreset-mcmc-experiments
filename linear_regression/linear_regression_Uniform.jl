using CSV
using DataFrames
using Random
using JLD
using Statistics
using StatsBase
include("../MCMCsampler/MCMCsampler.jl")
include("../util.jl")

function main(args)
    data = CSV.read("../data/linear_reg.csv", DataFrame, header=false)
    data = hcat(ones(nrow(data)), Matrix(data))
    data = [data[i,:] for i=[1:size(data,1);]]
    
    N = length(data)
    d = 10
    
    @assert length(args) == 3 "Error: script has 3 mandatory cmd line args"

    # Initialize the rng
    println("Initializing RNG")
    rng = Xoshiro(parse(Int, args[1])) 

    # Create the model
    println("Initializing model")
    model = MCMCsampler.LinearRegressionModel(length(data), data, reduce(hcat, data)', d, 1, zeros(length(data[1])), nothing)

    # parse number of samples
    n_samples = parse(Int, args[2])

    # Create the algorithm
    println("Initializing sampler")
    kernel = MCMCsampler.SliceSamplerMD()
    cv = MCMCsampler.CoresetLogProbEstimator(N = parse(Int, args[3]))
    cv.inds = sample(rng, [1:model.N;], cv.N; replace = false)
    cv.sub_dataset = reduce(hcat, @view(model.dataset[cv.inds]))'
    cv.weights = (model.N / cv.N) * ones(cv.N)

    println("Running sampler")
    θs, c_lp, c_g_lp, c_h_lp, c_time, weights = MCMCsampler.sample!(kernel, model, cv, 2*n_samples, rng)
    println(sum(θs[end-n_samples+1:end]) / length(θs[end-n_samples+1:end]))
    ts = reduce(hcat, θs)'[end-n_samples+1:end,:]
    D_stan = JLD.load("../stan_results/stan_lin_reg.jld")["data"]
    m_method = vec(mean(ts, dims=1))
    v_method = cov(ts)
    kl_est = kl_gaussian(vec(mean(ts, dims=1)), cov(ts), vec(mean(D_stan, dims=1)), cov(D_stan))
    println(c_lp[end])
    println(c_time[end])

    save("linear_regression_uniform_" * args[3] * "_" * args[1] * ".jld", "θs", θs, "c_lp", c_lp, "c_g_lp", c_g_lp,
                                                    "c_h_lp", c_h_lp, "c_time", c_time,
                                                    "weights", weights, "inds", cv.inds,
                                                    "kl", kl_est, "mean", m_method, "cov", v_method)
end

println("Running sampler with $(ARGS)")
main(ARGS)
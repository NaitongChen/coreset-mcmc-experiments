using CSV
using DataFrames
using MCMCsampler
using Random
using JLD
using Statistics
include("../util.jl")

function main(args)
    data = CSV.read("../data/logistic_reg.csv", DataFrame, header=false)
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
    model = LogisticRegressionModel(length(data), data, reduce(hcat, data)', d, log_reg_stratified_sampling)

    # parse number of samples
    n_samples = parse(Int, args[2])

    # Create the algorithm
    println("Initializing sampler")
    kernel = ULA(a = 0.00002, b=0, γ=1, adapt=false)
    cv = ModeLogProbEstimator(N = parse(Int, args[3]), tol = 1e-3, mode_n = 100)

    println("Running sampler")
    θs, c_lp, c_g_lp, c_h_lp, c_time = MCMCsampler.sample!(kernel, model, cv, 2*n_samples, rng)
    println(sum(θs[end-n_samples+1:end]) / length(θs[end-n_samples+1:end]))
    ts = reduce(hcat, θs)'[end-n_samples+1:end,:]
    D_stan = JLD.load("../stan_results/stan_log_reg.jld")["data"]
    m_method = vec(mean(ts, dims=1))
    v_method = cov(ts)
    kl_est = kl_gaussian(vec(mean(ts, dims=1)), cov(ts), vec(mean(D_stan, dims=1)), cov(D_stan))
    println(c_lp[end])
    println(c_time[end])

    save("logistic_regression_sgld_" * args[1] * ".jld", "θs", θs, "c_lp", c_lp, "c_g_lp", c_g_lp,
                                                    "c_h_lp", c_h_lp, "c_time", c_time, 
                                                    "kl", kl_est, "mean", m_method, "cov", v_method)
end

println("Running sampler with $(ARGS)")
main(ARGS)
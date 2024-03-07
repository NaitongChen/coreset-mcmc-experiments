using CSV
using DataFrames
using Random
using JLD
using Statistics
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
    if length(parse.(Float64, split(args[4], "_"))) == 1
        kernel = MCMCsampler.CoresetMCMC(kernel = MCMCsampler.GibbsSR(), replicas = 2, α = t -> parse(Float64, args[4]), delay = 1, train_iter = 25000, proj_n = model.N)
    else
        sizes = parse.(Float64, split(args[4], "_"))
        kernel = MCMCsampler.CoresetMCMC(kernel = MCMCsampler.GibbsSR(), replicas = 2, α = t -> sizes[1]/(t^sizes[2]), delay = 1, train_iter = 25000, proj_n = 10 * parse(Int, args[3]))
    end
    cv = MCMCsampler.CoresetLogProbEstimator(N = parse(Int, args[3]))

    println("Running sampler")
    θs, c_lp, c_g_lp, c_h_lp, c_time, weights = MCMCsampler.sample!(kernel, model, cv, n_samples, rng)
    println(sum(θs[end-n_samples+1:end]) / length(θs[end-n_samples+1:end]))
    ts = reduce(hcat, θs)'[end-n_samples+1:end,:]
    D_stan = JLD.load("../stan_results/sparse_regression_results_50000.jld")["θs"]
    m_method = vec(mean(ts[:,vcat(1:10, 21)], dims=1))
    v_method = cov(ts[:,vcat(1:10, 21)])
    kl_est = kl_gaussian(m_method, v_method, vec(mean(D_stan[:,vcat(1:10, 21)], dims=1)), cov(D_stan[:,vcat(1:0, 21)]))
    println(c_lp[end])
    println(c_time[end])

    save("sparse_regression_coresetMCMC_" * args[4] * "_" * args[3] * "_" * args[1] * ".jld", "θs", θs, "c_lp", c_lp, "c_g_lp", c_g_lp,
                                                    "c_h_lp", c_h_lp, "c_time", c_time,
                                                    "weights", weights, "inds", cv.inds,
                                                    "kl", kl_est, "mean", m_method, "cov", v_method)
end

println("Running sampler with $(ARGS)")
main(ARGS)
using CSV
using DataFrames
using Random
using JLD
using Statistics
include("../MCMCsampler/MCMCsampler.jl")
include("../util.jl")

function main(args)
    data = CSV.read("../data/poisson_reg.csv", DataFrame, header=false)
    data = Matrix(data)
    data = hcat(hcat(data[:,end-1], data[:,1:end-2]), data[:,end])
    data = [data[i,:] for i=[1:size(data,1);]]
    
    N = length(data)
    d = 8
    
    @assert length(args) == 4 "Error: script has 4 mandatory cmd line args"

    # Initialize the rng
    println("Initializing RNG")
    rng = Xoshiro(parse(Int, args[1])) 

    # Create the model
    println("Initializing model")
    model = MCMCsampler.PoissonRegressionModel(length(data), data, reduce(hcat, data)', d, 1, nothing)

    # parse number of samples
    n_samples = parse(Int, args[2])

    # Create the algorithm
    println("Initializing sampler")
    if parse(Float64, args[4]) == 1
        kernel = MCMCsampler.CoresetMCMC(kernel = MCMCsampler.SliceSamplerMD(), replicas = 2, α = t -> 0.5, delay = 1, train_iter = 50000, proj_n = model.N)
    else
        kernel = MCMCsampler.CoresetMCMC(kernel = MCMCsampler.SliceSamplerMD(), replicas = 2, α = t -> 1/(t^parse(Float64, args[4])), delay = 1, train_iter = 50000, proj_n = 10 * parse(Int, args[3]))
    end
    cv = MCMCsampler.CoresetLogProbEstimator(N = parse(Int, args[3]))

    println("Running sampler")
    θs, c_lp, c_g_lp, c_h_lp, c_time, weights = MCMCsampler.sample!(kernel, model, cv, n_samples, rng)
    println(sum(θs[end-n_samples+1:end]) / length(θs[end-n_samples+1:end]))
    ts = reduce(hcat, θs)'[end-n_samples+1:end,:]
    D_stan = JLD.load("../stan_results/stan_poisson_reg.jld")["data"]
    m_method = vec(mean(ts, dims=1))
    v_method = cov(ts)
    kl_est = kl_gaussian(vec(mean(ts, dims=1)), cov(ts), vec(mean(D_stan, dims=1)), cov(D_stan))
    println(c_lp[end])
    println(c_time[end])

    save("new_poisson_regression_coresetMCMC_" * args[4] * "_" * args[3] * "_" * args[1] * ".jld", "θs", θs, "c_lp", c_lp, "c_g_lp", c_g_lp,
                                                    "c_h_lp", c_h_lp, "c_time", c_time,
                                                    "weights", weights, "inds", cv.inds,
                                                    "kl", kl_est, "mean", m_method, "cov", v_method)
end

println("Running sampler with $(ARGS)")
main(ARGS)

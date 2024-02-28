using JLD2
using LinearAlgebra
using Plots
using Measures
using Statistics
using MCMCDiagnosticTools
include("../plotting_util.jl")

########################
# legends
########################
names = ["CoresetMCMC (SG)", "QNC", "SHF", "Uniform", "CoresetMCMC"]
colours = [palette(:Paired_8)[2], palette(:Paired_10)[10], palette(:Paired_8)[4], palette(:Paired_10)[8], palette(:Paired_8)[1]]

Ms = [10, 20, 50, 100, 200, 500]
n_run = 10
dat_size = length(Ms)

KL_cm = zeros(n_run, dat_size)
KL_cmf = zeros(n_run, dat_size)
KL_qnc = zeros(n_run, dat_size)
KL_shf = zeros(n_run, dat_size)
KL_unif = zeros(n_run, dat_size)

mrel_cm = zeros(n_run, dat_size)
mrel_cmf = zeros(n_run, dat_size)
mrel_qnc = zeros(n_run, dat_size)
mrel_shf = zeros(n_run, dat_size)
mrel_unif = zeros(n_run, dat_size)

srel_cm = zeros(n_run, dat_size)
srel_cmf = zeros(n_run, dat_size)
srel_qnc = zeros(n_run, dat_size)
srel_shf = zeros(n_run, dat_size)
srel_unif = zeros(n_run, dat_size)

ess_cm = zeros(n_run, dat_size)
ess_cmf = zeros(n_run, dat_size)
ess_qnc = zeros(n_run, dat_size)
ess_shf = zeros(n_run, dat_size)
ess_unif = zeros(n_run, dat_size)

train_cm = zeros(n_run, dat_size)
train_cmf = zeros(n_run, dat_size)
train_qnc = zeros(n_run, dat_size)
train_shf = zeros(n_run, dat_size)
train_unif = zeros(n_run, dat_size)

################################
################################
# KL coreset size
################################
################################
Threads.@threads for i in 1:n_run
    println(i)
    for j in 1:dat_size
        println(j)
        KL_cm[i,j] = load("new_poisson_regression_coresetMCMC_" * "0.3_" * string(Ms[j]) * "_" * string(i) * ".jld", "kl")
        KL_cmf[i,j] = load("new_poisson_regression_coresetMCMC_" * "1_" * string(Ms[j]) * "_" * string(i) * ".jld", "kl")
        KL_shf[i,j] = load("poisson_regression_SHF_" * string(Ms[j]) * "_" * string(i) * ".jld", "kl")
        KL_unif[i,j] = load("poisson_regression_uniform_" * string(Ms[j]) * "_" * string(i) * ".jld", "kl")
        if isfile("new_poisson_regression_QNC_" * string(Ms[j]) * "_" * string(i) * ".jld")
            KL_qnc[i,j] = load("new_poisson_regression_QNC_" * string(Ms[j]) * "_" * string(i) * ".jld", "kl")
        else
            KL_qnc[i,j] = NaN
        end
    end
end

plot(Ms[4:6], get_medians(KL_qnc[:, 4:6]), label = names[2], color = colours[2], ribbon = get_percentiles(KL_qnc[:,4:6]))
plot!(Ms, get_medians(KL_cm), label = names[1], color = colours[1], ribbon = get_percentiles(KL_cm))
plot!(Ms, get_medians(KL_cmf), label = names[5], color = colours[5], ribbon = get_percentiles(KL_cmf))
plot!(Ms, get_medians(KL_shf), label = names[3], color = colours[3], ribbon = get_percentiles(KL_shf))
plot!(Ms, get_medians(KL_unif), label = names[4], color = colours[4], ribbon = get_percentiles(KL_unif), yscale = :log10, guidefontsize=20, tickfontsize=15, formatter=:plain, legendfontsize=10, legend=(0.65,1), margin=5mm)
yticks!(10. .^[-2:1:20;])
xticks!(Ms)
ylabel!("Two-Moment KL")
xlabel!("coreset size")
savefig("plots/kl_coreset_cut.png")

plot(Ms, get_medians(KL_qnc), label = names[2], color = colours[2], ribbon = get_percentiles(KL_qnc))
plot!(Ms, get_medians(KL_cm), label = names[1], color = colours[1], ribbon = get_percentiles(KL_cm))
plot!(Ms, get_medians(KL_cmf), label = names[5], color = colours[5], ribbon = get_percentiles(KL_cmf))
plot!(Ms, get_medians(KL_shf), label = names[3], color = colours[3], ribbon = get_percentiles(KL_shf))
plot!(Ms, get_medians(KL_unif), label = names[4], color = colours[4], ribbon = get_percentiles(KL_unif), yscale = :log10, guidefontsize=20, tickfontsize=15, formatter=:plain, legendfontsize=10, legend=(0.65,1), margin=5mm)
yticks!(10. .^[-2:1:20;])
xticks!(Ms)
ylabel!("Two-Moment KL")
xlabel!("coreset size")
savefig("plots/kl_coreset.png")

################################
################################
# mrel coreset size
################################
################################
D_stan = load("../stan_results/stan_poisson_reg.jld", "data")
stan_mean = vec(mean(D_stan, dims=1))
Threads.@threads for i in 1:n_run
    println(i)
    for j in 1:dat_size
        println(j)
        m_method = load("new_poisson_regression_coresetMCMC_" * "0.3_" * string(Ms[j]) * "_" * string(i) * ".jld", "mean")
        mrel_cm[i,j] = norm(m_method - stan_mean) / norm(stan_mean)
        m_method = load("new_poisson_regression_coresetMCMC_" * "1_" * string(Ms[j]) * "_" * string(i) * ".jld", "mean")
        mrel_cmf[i,j] = norm(m_method - stan_mean) / norm(stan_mean)
        m_method = load("poisson_regression_SHF_" * string(Ms[j]) * "_" * string(i) * ".jld", "mean")
        mrel_shf[i,j] = norm(m_method - stan_mean) / norm(stan_mean)
        m_method = load("poisson_regression_uniform_" * string(Ms[j]) * "_" * string(i) * ".jld", "mean")
        mrel_unif[i,j] = norm(m_method - stan_mean) / norm(stan_mean)
        if isfile("new_poisson_regression_QNC_" * string(Ms[j]) * "_" * string(i) * ".jld")
            m_method = load("new_poisson_regression_QNC_" * string(Ms[j]) * "_" * string(i) * ".jld", "mean")
            mrel_qnc[i,j] = norm(m_method - stan_mean) / norm(stan_mean)
        else
            mrel_qnc[i,j] = NaN
        end
    end
end

plot(Ms, get_medians(mrel_cm), label = names[1], color = colours[1], ribbon = get_percentiles(mrel_cm))
plot!(Ms, get_medians(mrel_cmf), label = names[5], color = colours[5], ribbon = get_percentiles(mrel_cmf))
plot!(Ms[4:6], get_medians(mrel_qnc[:, 4:6]), label = names[2], color = colours[2], ribbon = get_percentiles(mrel_qnc[:,4:6]))
plot!(Ms, get_medians(mrel_shf), label = names[3], color = colours[3], ribbon = get_percentiles(mrel_shf))
plot!(Ms, get_medians(mrel_unif), label = names[4], color = colours[4], ribbon = get_percentiles(mrel_unif), yscale = :log10, guidefontsize=20, tickfontsize=15, formatter=:plain, legendfontsize=8, legend=(0.11,0.27), margin=5mm)
yticks!(10. .^[-10:1:6;])
xticks!(Ms)
ylabel!("Relative mean error")
xlabel!("coreset size")
savefig("plots/mrel_coreset_cut.png")

plot(Ms, get_medians(mrel_cm), label = names[1], color = colours[1], ribbon = get_percentiles(mrel_cm))
plot!(Ms, get_medians(mrel_cmf), label = names[5], color = colours[5], ribbon = get_percentiles(mrel_cmf))
plot!(Ms, get_medians(mrel_qnc), label = names[2], color = colours[2], ribbon = get_percentiles(mrel_qnc))
plot!(Ms, get_medians(mrel_shf), label = names[3], color = colours[3], ribbon = get_percentiles(mrel_shf))
plot!(Ms, get_medians(mrel_unif), label = names[4], color = colours[4], ribbon = get_percentiles(mrel_unif), yscale = :log10, guidefontsize=20, tickfontsize=15, formatter=:plain, legendfontsize=8, legend=(0.11,0.27), margin=5mm)
yticks!(10. .^[-10:1:6;])
xticks!(Ms)
ylabel!("Relative mean error")
xlabel!("coreset size")
savefig("plots/mrel_coreset.png")

################################
################################
# srel coreset size
################################
################################
D_stan = load("../stan_results/stan_poisson_reg.jld", "data")
stan_cov = cov(D_stan, dims=1)
Threads.@threads for i in 1:n_run
    println(i)
    for j in 1:dat_size
        println(j)
        m_method = load("new_poisson_regression_coresetMCMC_" * "1_" * string(Ms[j]) * "_" * string(i) * ".jld", "cov")
        srel_cmf[i,j] = norm(m_method - stan_cov) / norm(stan_cov)
        m_method = load("new_poisson_regression_coresetMCMC_" * "0.3_" * string(Ms[j]) * "_" * string(i) * ".jld", "cov")
        srel_cm[i,j] = norm(m_method - stan_cov) / norm(stan_cov)
        m_method = load("poisson_regression_SHF_" * string(Ms[j]) * "_" * string(i) * ".jld", "cov")
        srel_shf[i,j] = norm(m_method - stan_cov) / norm(stan_cov)
        m_method = load("poisson_regression_uniform_" * string(Ms[j]) * "_" * string(i) * ".jld", "cov")
        srel_unif[i,j] = norm(m_method - stan_cov) / norm(stan_cov)
        if isfile("new_poisson_regression_QNC_" * string(Ms[j]) * "_" * string(i) * ".jld")
            m_method = load("new_poisson_regression_QNC_" * string(Ms[j]) * "_" * string(i) * ".jld", "cov")
            srel_qnc[i,j] = norm(m_method - stan_cov) / norm(stan_cov)
        else
            srel_qnc[i,j] = NaN
        end
    end
end

plot(Ms, get_medians(srel_cm), label = names[1], color = colours[1], ribbon = get_percentiles(srel_cm))
plot!(Ms, get_medians(srel_cmf), label = names[5], color = colours[5], ribbon = get_percentiles(srel_cmf))
plot!(Ms[4:6], get_medians(srel_qnc[:, 4:6]), label = names[2], color = colours[2], ribbon = get_percentiles(srel_qnc[:,4:6]))
plot!(Ms, get_medians(srel_shf), label = names[3], color = colours[3], ribbon = get_percentiles(srel_shf))
plot!(Ms, get_medians(srel_unif), label = names[4], color = colours[4], ribbon = get_percentiles(srel_unif), yscale = :log10, guidefontsize=20, tickfontsize=15, formatter=:plain, legendfontsize=10, legend=(0.6,0.6), margin=5mm)
ylabel!("Relative cov error")
xlabel!("coreset size")
yticks!(10. .^[-2:1:6;])
xticks!(Ms)
savefig("plots/srel_coreset_cut.png")

plot(Ms, get_medians(srel_cm), label = names[1], color = colours[1], ribbon = get_percentiles(srel_cm))
plot!(Ms, get_medians(srel_cmf), label = names[5], color = colours[5], ribbon = get_percentiles(srel_cmf))
plot!(Ms, get_medians(srel_qnc), label = names[2], color = colours[2], ribbon = get_percentiles(srel_qnc))
plot!(Ms, get_medians(srel_shf), label = names[3], color = colours[3], ribbon = get_percentiles(srel_shf))
plot!(Ms, get_medians(srel_unif), label = names[4], color = colours[4], ribbon = get_percentiles(srel_unif), yscale = :log10, guidefontsize=20, tickfontsize=15, formatter=:plain, legendfontsize=10, legend=(0.6,1), margin=5mm)
ylabel!("Relative cov error")
xlabel!("coreset size")
yticks!(10. .^[-2:1:6;])
xticks!(Ms)
savefig("plots/srel_coreset.png")

################################
################################
# ess coreset size
################################
################################
D_stan = load("../stan_results/stan_poisson_reg.jld")["data"]
D_stan_split = rand(10000, 10, size(D_stan, 2))
b = 1
f = 10000
for i in 1:10
    if i > 1
        b += 10000
        f += 10000
    end
    D_stan_split[:,i,:] = D_stan[b:f,:]
end
ess_stan = ess(D_stan_split)
for i in 1:n_run
    println(i)
    for j in 1:dat_size
        println(j)
        m_method = load("new_poisson_regression_coresetMCMC_" * "0.3_" *string(Ms[j]) * "_" * string(i) * ".jld", "θs")
        time = load("new_poisson_regression_coresetMCMC_" * "0.3_" * string(Ms[j]) * "_" * string(i) * ".jld", "c_time")
        time = time[end] - time[50001]
        m_method = reduce(hcat, m_method)'
        m_method = m_method[100001:end, :]
        split = rand(10000, 1, size(D_stan, 2))
        split[:,1,:] = m_method
        ess_cm[i,j] = minimum(ess(split)) / time

        m_method = load("new_poisson_regression_coresetMCMC_" * "1_" * string(Ms[j]) * "_" * string(i) * ".jld", "θs")
        time = load("new_poisson_regression_coresetMCMC_" * "1_" * string(Ms[j]) * "_" * string(i) * ".jld", "c_time")
        time = time[end] - time[50001]
        m_method = reduce(hcat, m_method)'
        m_method = m_method[100001:end, :]
        split = rand(10000, 1, size(D_stan, 2))
        split[:,1,:] = m_method
        ess_cmf[i,j] = minimum(ess(split)) / time
        
        m_method = load("poisson_regression_SHF_" * string(Ms[j]) * "_" * string(i) * ".jld", "θs")
        time = load("poisson_regression_SHF_" * string(Ms[j]) * "_" * string(i) * ".jld", "c_time")
        time = time[end] - time[1]
        m_method = reduce(hcat, m_method)'
        split = rand(10000, 1, size(D_stan, 2))
        split[:,1,:] = m_method
        ess_shf[i,j] = minimum(ess(split)) / time
        
        m_method = load("poisson_regression_uniform_" * string(Ms[j]) * "_" * string(i) * ".jld", "θs")
        time = load("poisson_regression_uniform_" * string(Ms[j]) * "_" * string(i) * ".jld", "c_time")
        time = time[end] - time[10001]
        m_method = reduce(hcat, m_method)'
        m_method = m_method[10001:end, :]
        split = rand(10000, 1, size(D_stan, 2))
        split[:,1,:] = m_method
        ess_unif[i,j] = minimum(ess(split)) / time
        
        if isfile("new_poisson_regression_QNC_" * string(Ms[j]) * "_" * string(i) * ".jld")
            m_method = load("new_poisson_regression_QNC_" * string(Ms[j]) * "_" * string(i) * ".jld", "θs")
            time = load("new_poisson_regression_QNC_" * string(Ms[j]) * "_" * string(i) * ".jld", "c_time")
            time = time[end] - time[end-10000+1]
            m_method = reduce(hcat, m_method)'
            m_method = m_method[end-10000+1:end, :]
            split = rand(10000, 1, size(D_stan, 2))
            split[:,1,:] = m_method
            ess_qnc[i,j] = minimum(ess(split)) / time
        else
            ess_qnc[i,j] = NaN
        end
    end
end

plot(Ms, get_medians(ess_cm), label = names[1], color = colours[1], ribbon = get_percentiles(ess_cm))
plot!(Ms, get_medians(ess_cmf), label = names[5], color = colours[5], ribbon = get_percentiles(ess_cmf))
plot!(Ms[4:6], get_medians(ess_qnc[:, 4:6]), label = names[2], color = colours[2], ribbon = get_percentiles(ess_qnc[:,4:6]))
plot!(Ms, get_medians(ess_shf), label = names[3], color = colours[3], ribbon = get_percentiles(ess_shf))
plot!(Ms, get_medians(ess_unif), label = names[4], color = colours[4], ribbon = get_percentiles(ess_unif), yscale = :log10, guidefontsize=20, tickfontsize=15, formatter=:plain, legendfontsize=10, legend=(0.6,0.4), margin=5mm)
ylabel!("min ESS/s")
xlabel!("coreset size")
yticks!(10. .^[-2:1:6;])
ylims!((10^0, 10^3))
xticks!(Ms)
savefig("plots/ess_coreset_cut.png")

plot(Ms, get_medians(ess_cm), label = names[1], color = colours[1], ribbon = get_percentiles(ess_cm))
plot!(Ms, get_medians(ess_cmf), label = names[5], color = colours[5], ribbon = get_percentiles(ess_cmf))
plot!(Ms, get_medians(ess_qnc), label = names[2], color = colours[2], ribbon = get_percentiles(ess_qnc))
plot!(Ms, get_medians(ess_shf), label = names[3], color = colours[3], ribbon = get_percentiles(ess_shf))
plot!(Ms, get_medians(ess_unif), label = names[4], color = colours[4], ribbon = get_percentiles(ess_unif), yscale = :log10, guidefontsize=20, tickfontsize=15, formatter=:plain, legendfontsize=10, legend=(0.6,0.4), margin=5mm)
ylabel!("min ESS/s")
xlabel!("coreset size")
yticks!(10. .^[-2:1:6;])
ylims!((10^0, 10^3))
xticks!(Ms)
savefig("plots/ess_coreset.png")

################################
################################
# training time coreset size
################################
################################
D_stan = load("../stan_results/stan_poisson_reg.jld")["data"]
stan_cov = cov(D_stan, dims=1)
Threads.@threads for i in 1:n_run
    println(i)
    for j in 1:dat_size
        println(j)
        m_method = load("new_poisson_regression_coresetMCMC_" * "0.3_" * string(Ms[j]) * "_" * string(i) * ".jld", "c_time")
        train_cm[i,j] = m_method[50000]
        m_method = load("new_poisson_regression_coresetMCMC_" * "1_" * string(Ms[j]) * "_" * string(i) * ".jld", "c_time")
        train_cmf[i,j] = m_method[50000]
        m_method = load("poisson_regression_SHF_" * string(Ms[j]) * "_" * string(i) * ".jld", "c_time")
        train_shf[i,j] = m_method[1]
        m_method = load("poisson_regression_uniform_" * string(Ms[j]) * "_" * string(i) * ".jld", "c_time")
        train_unif[i,j] = m_method[1]
        if isfile("new_poisson_regression_QNC_" * string(Ms[j]) * "_" * string(i) * ".jld")
            m_method = load("new_poisson_regression_QNC_" * string(Ms[j]) * "_" * string(i) * ".jld", "c_time")
            train_qnc[i,j] = m_method[100]
        else
            train_qnc[i,j] = NaN
        end
    end
end

plot(Ms, get_medians(train_cm), label = names[1], color = colours[1], ribbon = get_percentiles(train_cm))
plot!(Ms, get_medians(train_cmf), label = names[5], color = colours[5], ribbon = get_percentiles(train_cmf))
plot!(Ms[4:6], get_medians(train_qnc[:, 4:6]), label = names[2], color = colours[2], ribbon = get_percentiles(train_qnc[:,4:6]))
plot!(Ms, get_medians(train_shf), label = names[3], color = colours[3], ribbon = get_percentiles(train_shf), guidefontsize=20, tickfontsize=15, formatter=:plain, legendfontsize=10, legend=(0.6,0.5), margin=5mm)
ylabel!("Training time (s)")
xlabel!("coreset size")
yticks!([100, 500, 1000, 2000, 3000, 4000, 5000, 6000])
# ylims!((10^2.2, 10^2.8))
xticks!(Ms)
savefig("plots/train_coreset_cut.png")

plot(Ms, get_medians(train_cm), label = names[1], color = colours[1], ribbon = get_percentiles(train_cm))
plot!(Ms, get_medians(train_cmf), label = names[5], color = colours[5], ribbon = get_percentiles(train_cmf))
plot!(Ms, get_medians(train_qnc), label = names[2], color = colours[2], ribbon = get_percentiles(train_qnc))
plot!(Ms, get_medians(train_shf), label = names[3], color = colours[3], ribbon = get_percentiles(train_shf), guidefontsize=20, tickfontsize=15, formatter=:plain, legendfontsize=10, legend=(0.6,0.5), margin=5mm)
ylabel!("Training time (s)")
xlabel!("coreset size")
yticks!([100, 500, 1000, 2000, 3000, 4000, 5000, 6000])
# ylims!((10^2.2, 10^2.8))
xticks!(Ms)
savefig("plots/train_coreset.png")

JLD2.save("results_coresets.jld", "KL_cm", KL_cm,
                            "KL_cmf", KL_cmf,
                            "KL_qnc", KL_qnc,
                            "KL_shf", KL_shf,
                            "KL_unif", KL_unif,
                            "mrel_cm", mrel_cm,
                            "mrel_cmf", mrel_cmf,
                            "mrel_qnc", mrel_qnc,
                            "mrel_shf", mrel_shf,
                            "mrel_unif", mrel_unif,
                            "srel_cm", srel_cm,
                            "srel_cmf", srel_cmf,
                            "srel_qnc", srel_qnc,
                            "srel_shf", srel_shf,
                            "srel_unif", srel_unif,
                            "ess_cm", ess_cm,
                            "ess_cmf", ess_cmf,
                            "ess_qnc", ess_qnc,
                            "ess_shf", ess_shf,
                            "ess_unif", ess_unif,
                            "train_cm", train_cm,
                            "train_cmf", train_cmf,
                            "train_qnc", train_qnc,
                            "train_shf", train_shf,
                            "train_unif", train_unif)

# KL_cm = JLD2.load("results_coresets.jld", "KL_cm")
# KL_cmf = JLD2.load("results_coresets.jld", "KL_cmf")
# KL_qnc = JLD2.load("results_coresets.jld", "KL_qnc")
# KL_shf = JLD2.load("results_coresets.jld", "KL_shf")
# KL_unif = JLD2.load("results_coresets.jld", "KL_unif")

# mrel_cm = JLD2.load("results_coresets.jld", "mrel_cm")
# mrel_cmf = JLD2.load("results_coresets.jld", "mrel_cmf")
# mrel_qnc = JLD2.load("results_coresets.jld", "mrel_qnc")
# mrel_shf = JLD2.load("results_coresets.jld", "mrel_shf")
# mrel_unif = JLD2.load("results_coresets.jld", "mrel_unif")

# srel_cm = JLD2.load("results_coresets.jld", "srel_cm")
# srel_cmf = JLD2.load("results_coresets.jld", "srel_cmf")
# srel_qnc = JLD2.load("results_coresets.jld", "srel_qnc")
# srel_shf = JLD2.load("results_coresets.jld", "srel_shf")
# srel_unif = JLD2.load("results_coresets.jld", "srel_unif")

# ess_cm = JLD2.load("results_coresets.jld", "ess_cm")
# ess_cmf = JLD2.load("results_coresets.jld", "ess_cmf")
# ess_qnc = JLD2.load("results_coresets.jld", "ess_qnc")
# ess_shf = JLD2.load("results_coresets.jld", "ess_shf")
# ess_unif = JLD2.load("results_coresets.jld", "ess_unif")

# train_cm = JLD2.load("results_coresets.jld", "train_cm")
# train_cmf = JLD2.load("results_coresets.jld", "train_cmf")
# train_qnc = JLD2.load("results_coresets.jld", "train_qnc")
# train_shf = JLD2.load("results_coresets.jld", "train_shf")
# train_unif = JLD2.load("results_coresets.jld", "train_unif")
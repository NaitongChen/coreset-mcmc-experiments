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
names = ["CoresetMCMC-S", "QNC", "SHF", "Uniform", "CoresetMCMC"]
colours = [palette(:Paired_8)[1], palette(:Paired_10)[10], palette(:Paired_8)[4], palette(:Paired_10)[8], palette(:Paired_8)[2]]

Ms = [10, 20, 50, 100, 200, 500]
n_run = 10
n_samples = 10000
dat_size = length(Ms)

KL_cm = zeros(n_run, dat_size)
KL_cmf = zeros(n_run, dat_size)
KL_qnc = zeros(n_run, dat_size)
KL_shf = zeros(n_run, dat_size)
KL_unif = zeros(n_run, dat_size)

KLd_cm = zeros(n_run, dat_size)
KLd_cmf = zeros(n_run, dat_size)
KLd_qnc = zeros(n_run, dat_size)
KLd_shf = zeros(n_run, dat_size)
KLd_unif = zeros(n_run, dat_size)

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

essd_cm = zeros(n_run, dat_size)
essd_cmf = zeros(n_run, dat_size)
essd_qnc = zeros(n_run, dat_size)
essd_shf = zeros(n_run, dat_size)
essd_unif = zeros(n_run, dat_size)

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
        KL_cm[i,j] = load("sparse_regression_coresetMCMC_" * "0.5_0.5_" * string(Ms[j]) * "_" * string(i) * ".jld", "kl")
        KL_cmf[i,j] = load("sparse_regression_coresetMCMC_" * "0.01_" * string(Ms[j]) * "_" * string(i) * ".jld", "kl")
        # KL_shf[i,j] = load("sparse_regression_SHF_" * string(Ms[j]) * "_" * string(i) * ".jld", "kl")
        KL_unif[i,j] = load("sparse_regression_uniform_" * string(Ms[j]) * "_" * string(i) * ".jld", "kl")
        if isfile("sparse_regression_QNC_" * string(Ms[j]) * "_" * string(i) * ".jld")
            KL_qnc[i,j] = load("sparse_regression_QNC_" * string(Ms[j]) * "_" * string(i) * ".jld", "kl")
        else
            KL_qnc[i,j] = NaN
        end
    end
end

# coreset size 10 really bad, hence ignore
plot(Ms, get_medians(KL_qnc), label = names[2], color = colours[2], ribbon = get_percentiles(KL_qnc))
plot!(Ms, get_medians(KL_cm), label = names[1], color = colours[1], ribbon = get_percentiles(KL_cm))
plot!(Ms, get_medians(KL_cmf), label = names[5], color = colours[5], ribbon = get_percentiles(KL_cmf))
# plot!(Ms, get_medians(KL_shf), label = names[3], color = colours[3], ribbon = get_percentiles(KL_shf))
plot!(Ms, get_medians(KL_unif), label = names[4], color = colours[4], ribbon = get_percentiles(KL_unif), yscale = :log10, guidefontsize=20, tickfontsize=15, formatter=:plain, legendfontsize=10, legend=false, margin=5mm)
yticks!(10. .^[-3, -2, -1, 0, 1, 2, 3, 4])
ylabel!("Two-Moment KL")
xticks!(Ms[vcat(1,3:6)])
ylims!((10^-2, 10^4))
xlabel!("coreset size")
savefig("plots/kl_coreset.png")

################################
################################
# KLd coreset size
################################
################################
D_stan = JLD.load("../stan_results/sparse_regression_big.jld")["θs"]
gamma_stan = D_stan[:, 6:10]
gamma_stan = [gamma_stan[i,:] for i in 1:size(gamma_stan,1)]
mapstan = countmap(gamma_stan)

for i in 1:n_run
    println(i)
    for j in 1:dat_size
        println(j)
        gamma_m = reduce(hcat, load("sparse_regression_coresetMCMC_" * "0.5_0.5_" * string(Ms[j]) * "_" * string(i) * ".jld", "θs"))'[end-n_samples+1:end,6:10]
        gamma_m = [gamma_m[i,:] for i in 1:size(gamma_m,1)]
        mapm = countmap(gamma_m)
        for (key, value) in mapstan
            if haskey(mapm, key)
                KLd_cm[i,j] += mapstan[key]/100000 * (log(mapstan[key]/100000) - log(mapm[key]/10000))
            end
        end

        gamma_m = reduce(hcat, load("sparse_regression_coresetMCMC_" * "0.01_" * string(Ms[j]) * "_" * string(i) * ".jld", "θs"))'[end-n_samples+1:end,6:10]
        gamma_m = [gamma_m[i,:] for i in 1:size(gamma_m,1)]
        mapm = countmap(gamma_m)
        for (key, value) in mapstan
            if haskey(mapm, key)
                KLd_cmf[i,j] += mapstan[key]/100000 * (log(mapstan[key]/100000) - log(mapm[key]/10000))
            end
        end

        if isfile("sparse_regression_QNC_" * string(Ms[j]) * "_" * string(i) * ".jld")
            gamma_m = reduce(hcat, load("sparse_regression_QNC_" * string(Ms[j]) * "_" * string(i) * ".jld", "θs"))'[end-n_samples+1:end,6:10]
            gamma_m = [gamma_m[i,:] for i in 1:size(gamma_m,1)]
            mapm = countmap(gamma_m)
            for (key, value) in mapstan
                if haskey(mapm, key)
                    KLd_qnc[i,j] += mapstan[key]/100000 * (log(mapstan[key]/100000) - log(mapm[key]/10000))
                end
            end
        else
            KLd_qnc[i,j] = NaN
        end

        gamma_m = reduce(hcat, load("sparse_regression_uniform_" * string(Ms[j]) * "_" * string(i) * ".jld", "θs"))'[end-n_samples+1:end,6:10]
        gamma_m = [gamma_m[i,:] for i in 1:size(gamma_m,1)]
        mapm = countmap(gamma_m)
        for (key, value) in mapstan
            if haskey(mapm, key)
                KLd_unif[i,j] += mapstan[key]/100000 * (log(mapstan[key]/100000) - log(mapm[key]/10000))
            end
        end
    end
end

# coreset size 10 really bad, hence ignore
plot(Ms, get_medians(KLd_qnc), label = names[2], color = colours[2], ribbon = get_percentiles(KLd_qnc))
plot!(Ms, get_medians(KLd_cm), label = names[1], color = colours[1], ribbon = get_percentiles(KLd_cm))
plot!(Ms, get_medians(KLd_cmf), label = names[5], color = colours[5], ribbon = get_percentiles(KLd_cmf))
# plot!(Ms, get_medians(KLd_shf), label = names[3], color = colours[3], ribbon = get_percentiles(KLd_shf))
plot!(Ms, get_medians(KLd_unif), label = names[4], color = colours[4], ribbon = get_percentiles(KLd_unif) , yscale = :log10, guidefontsize=20, tickfontsize=15, formatter=:plain, legendfontsize=10, legend=false, margin=5mm)
yticks!(10. .^[-3, -2, -1, 0, 1, 2, 3, 4])
ylabel!("KL")
xticks!(Ms[vcat(1,3:6)])
ylims!((10^-5, 10^1))
yticks!(10. .^[-4:1:1;])
xlabel!("coreset size")
savefig("plots/kld_coreset.png")

################################
################################
# mrel coreset size
################################
################################
D_stan = JLD.load("../stan_results/sparse_regression_big.jld")["θs"]
stan_mean = vec(mean(D_stan[:,vcat(1:5, 11)], dims=1))
Threads.@threads for i in 1:n_run
    println(i)
    for j in 1:dat_size
        println(j)
        m_method = load("sparse_regression_coresetMCMC_" * "0.5_0.5_" * string(Ms[j]) * "_" * string(i) * ".jld", "mean")
        mrel_cm[i,j] = norm(m_method - stan_mean) / norm(stan_mean)
        m_method = load("sparse_regression_coresetMCMC_" * "0.01_" * string(Ms[j]) * "_" * string(i) * ".jld", "mean")
        mrel_cmf[i,j] = norm(m_method - stan_mean) / norm(stan_mean)
        # m_method = load("sparse_regression_SHF_" * string(Ms[j]) * "_" * string(i) * ".jld", "mean")
        # mrel_shf[i,j] = norm(m_method - stan_mean) / norm(stan_mean)
        m_method = load("sparse_regression_uniform_" * string(Ms[j]) * "_" * string(i) * ".jld", "mean")
        mrel_unif[i,j] = norm(m_method - stan_mean) / norm(stan_mean)
        if isfile("sparse_regression_QNC_" * string(Ms[j]) * "_" * string(i) * ".jld")
            m_method = load("sparse_regression_QNC_" * string(Ms[j]) * "_" * string(i) * ".jld", "mean")
            mrel_qnc[i,j] = norm(m_method - stan_mean) / norm(stan_mean)
        else
            mrel_qnc[i,j] = NaN
        end
    end
end

plot(Ms, get_medians(mrel_qnc), label = names[2], color = colours[2], ribbon = get_percentiles(mrel_qnc))
plot!(Ms, get_medians(mrel_cm), label = names[1], color = colours[1], ribbon = get_percentiles(mrel_cm))
plot!(Ms, get_medians(mrel_cmf), label = names[5], color = colours[5], ribbon = get_percentiles(mrel_cmf))
# plot!(Ms, get_medians(mrel_shf), label = names[3], color = colours[3], ribbon = get_percentiles(mrel_shf))
plot!(Ms, get_medians(mrel_unif), label = names[4], color = colours[4], ribbon = get_percentiles(mrel_unif), yscale = :log10, guidefontsize=20, tickfontsize=15, formatter=:plain, legendfontsize=10, legend=false, margin=5mm)
yticks!(10. .^[-7:1:6;])
# ylims!((10^-5, 10^0))
ylabel!("Relative mean error")
xlabel!("coreset size")
xticks!(Ms[vcat(1,3:6)])
savefig("plots/mrel_coreset.png")

################################
################################
# srel coreset size
################################
################################
D_stan = JLD.load("../stan_results/sparse_regression_big.jld")["θs"]
stan_cov = cov(D_stan[:,vcat(1:5, 11)], dims=1)
Threads.@threads for i in 1:n_run
    println(i)
    for j in 1:dat_size
        println(j)
        m_method = load("sparse_regression_coresetMCMC_" * "0.01_" * string(Ms[j]) * "_" * string(i) * ".jld", "cov")
        srel_cmf[i,j] = norm(m_method - stan_cov) / norm(stan_cov)
        m_method = load("sparse_regression_coresetMCMC_" * "0.5_0.5_" * string(Ms[j]) * "_" * string(i) * ".jld", "cov")
        srel_cm[i,j] = norm(m_method - stan_cov) / norm(stan_cov)
        # m_method = load("sparse_regression_SHF_" * string(Ms[j]) * "_" * string(i) * ".jld", "cov")
        # srel_shf[i,j] = norm(m_method - stan_cov) / norm(stan_cov)
        m_method = load("sparse_regression_uniform_" * string(Ms[j]) * "_" * string(i) * ".jld", "cov")
        srel_unif[i,j] = norm(m_method - stan_cov) / norm(stan_cov)
        if isfile("sparse_regression_QNC_" * string(Ms[j]) * "_" * string(i) * ".jld")
            m_method = load("sparse_regression_QNC_" * string(Ms[j]) * "_" * string(i) * ".jld", "cov")
            srel_qnc[i,j] = norm(m_method - stan_cov) / norm(stan_cov)
        else
            srel_qnc[i,j] = NaN
        end
    end
end

plot(Ms, get_medians(srel_qnc), label = names[2], color = colours[2], ribbon = get_percentiles(srel_qnc))
plot!(Ms, get_medians(srel_cm), label = names[1], color = colours[1], ribbon = get_percentiles(srel_cm))
plot!(Ms, get_medians(srel_cmf), label = names[5], color = colours[5], ribbon = get_percentiles(srel_cmf))
# plot!(Ms, get_medians(srel_shf), label = names[3], color = colours[3], ribbon = get_percentiles(srel_shf))
plot!(Ms, get_medians(srel_unif), label = names[4], color = colours[4], ribbon = get_percentiles(srel_unif), yscale = :log10, guidefontsize=20, tickfontsize=15, formatter=:plain, legendfontsize=10, legend=false, margin=5mm)
ylabel!("Relative cov error")
xlabel!("coreset size")
yticks!(10. .^[-2:1:10;])
xticks!(Ms[vcat(1,3:6)])
ylims!((10^-2, 10^0))
savefig("plots/srel_coreset.png")

################################
################################
# ess coreset size
################################
################################
D_stan = JLD.load("../stan_results/sparse_regression_big.jld")["θs"]
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
        m_method = load("sparse_regression_coresetMCMC_" * "0.5_0.5_" *string(Ms[j]) * "_" * string(i) * ".jld", "θs")
        time = load("sparse_regression_coresetMCMC_" * "0.5_0.5_" * string(Ms[j]) * "_" * string(i) * ".jld", "c_time")
        time = time[end] - time[25001]
        m_method = reduce(hcat, m_method)'
        m_method = m_method[50001:end, vcat(1:5, 11)]
        split = rand(10000, 1, 6)
        split[:,1,:] = m_method
        ess_cm[i,j] = minimum(ess(split)) / time

        m_method = load("sparse_regression_coresetMCMC_" * "0.01_" * string(Ms[j]) * "_" * string(i) * ".jld", "θs")
        time = load("sparse_regression_coresetMCMC_" * "0.01_" * string(Ms[j]) * "_" * string(i) * ".jld", "c_time")
        time = time[end] - time[25001]
        m_method = reduce(hcat, m_method)'
        m_method = m_method[50001:end, vcat(1:5, 11)]
        split = rand(10000, 1, 6)
        split[:,1,:] = m_method
        ess_cmf[i,j] = minimum(ess(split)) / time
        
        # m_method = load("sparse_regression_SHF_" * string(Ms[j]) * "_" * string(i) * ".jld", "θs")
        # time = load("sparse_regression_SHF_" * string(Ms[j]) * "_" * string(i) * ".jld", "c_time")
        # time = time[end] - time[1]
        # m_method = reduce(hcat, m_method)'
        # split = rand(10000, 1, size(D_stan, 2))
        # split[:,1,:] = m_method
        # ess_shf[i,j] = minimum(ess(split)) / time
        
        m_method = load("sparse_regression_uniform_" * string(Ms[j]) * "_" * string(i) * ".jld", "θs")
        time = load("sparse_regression_uniform_" * string(Ms[j]) * "_" * string(i) * ".jld", "c_time")
        time = time[end] - time[10001]
        m_method = reduce(hcat, m_method)'
        m_method = m_method[10001:end, vcat(1:5, 11)]
        split = rand(10000, 1, 6)
        split[:,1,:] = m_method
        ess_unif[i,j] = minimum(ess(split)) / time
        
        if isfile("sparse_regression_QNC_" * string(Ms[j]) * "_" * string(i) * ".jld")
            m_method = load("sparse_regression_QNC_" * string(Ms[j]) * "_" * string(i) * ".jld", "θs")
            time = load("sparse_regression_QNC_" * string(Ms[j]) * "_" * string(i) * ".jld", "c_time")
            time = time[end] - time[end-10000+1]
            m_method = reduce(hcat, m_method)'
            m_method = m_method[end-10000+1:end, vcat(1:5, 11)]
            split = rand(10000, 1, 6)
            split[:,1,:] = m_method
            ess_qnc[i,j] = minimum(ess(split)) / time
        else
            ess_qnc[i,j] = NaN
        end
    end
end

plot(Ms, get_medians(ess_qnc), label = names[2], color = colours[2], ribbon = get_percentiles(ess_qnc))
plot!(Ms, get_medians(ess_cm), label = names[1], color = colours[1], ribbon = get_percentiles(ess_cm))
plot!(Ms, get_medians(ess_cmf), label = names[5], color = colours[5], ribbon = get_percentiles(ess_cmf))
# plot!(Ms, get_medians(ess_shf), label = names[3], color = colours[3], ribbon = get_percentiles(ess_shf))
plot!(Ms, get_medians(ess_unif), label = names[4], color = colours[4], ribbon = get_percentiles(ess_unif), yscale = :log10, guidefontsize=20, tickfontsize=15, formatter=:plain, legendfontsize=15, legend=(0.4, 0.6), margin=5mm)
ylabel!("min ESS/s")
xlabel!("coreset size")
yticks!(10. .^[-2:1:6;])
ylims!((10^1, 10^4))
xticks!(Ms[vcat(1,3:6)])
savefig("plots/ess_coreset.png")

################################
################################
# essd coreset size
################################
################################
for i in 1:n_run
    println(i)
    for j in 1:dat_size
        println(j)
        m_method = load("sparse_regression_coresetMCMC_" * "0.5_0.5_" *string(Ms[j]) * "_" * string(i) * ".jld", "θs")
        time = load("sparse_regression_coresetMCMC_" * "0.5_0.5_" * string(Ms[j]) * "_" * string(i) * ".jld", "c_time")
        time = time[end] - time[25001]
        m_method = reduce(hcat, m_method)'
        m_method = m_method[50001:end, 6:10]
        prec = zeros(n_samples)
        for i in [1:n_samples;]
            prec[i] = sum(m_method[i,:] .== [0., 0., 0., 1., 1.]) / 5
        end
        split = rand(10000, 1, 1)
        split[:,1,:] = prec
        essd_cm[i,j] = minimum(ess(split)) / time

        m_method = load("sparse_regression_coresetMCMC_" * "0.01_" * string(Ms[j]) * "_" * string(i) * ".jld", "θs")
        time = load("sparse_regression_coresetMCMC_" * "0.01_" * string(Ms[j]) * "_" * string(i) * ".jld", "c_time")
        time = time[end] - time[25001]
        m_method = reduce(hcat, m_method)'
        m_method = m_method[50001:end, 6:10]
        prec = zeros(n_samples)
        for i in [1:n_samples;]
            prec[i] = sum(m_method[i,:] .== [0., 0., 0., 1., 1.]) / 5
        end
        split = rand(10000, 1, 1)
        split[:,1,:] = prec
        essd_cmf[i,j] = minimum(ess(split)) / time
        
        m_method = load("sparse_regression_uniform_" * string(Ms[j]) * "_" * string(i) * ".jld", "θs")
        time = load("sparse_regression_uniform_" * string(Ms[j]) * "_" * string(i) * ".jld", "c_time")
        time = time[end] - time[10001]
        m_method = reduce(hcat, m_method)'
        m_method = m_method[10001:end, 6:10]
        prec = zeros(n_samples)
        for i in [1:n_samples;]
            prec[i] = sum(m_method[i,:] .== [0., 0., 0., 1., 1.]) / 5
        end
        split = rand(10000, 1, 1)
        split[:,1,:] = prec
        essd_unif[i,j] = minimum(ess(split)) / time
        
        if isfile("sparse_regression_QNC_" * string(Ms[j]) * "_" * string(i) * ".jld")
            m_method = load("sparse_regression_QNC_" * string(Ms[j]) * "_" * string(i) * ".jld", "θs")
            time = load("sparse_regression_QNC_" * string(Ms[j]) * "_" * string(i) * ".jld", "c_time")
            time = time[end] - time[end-10000+1]
            m_method = reduce(hcat, m_method)'
            m_method = m_method[end-10000+1:end, 6:10]
            prec = zeros(n_samples)
            for i in [1:n_samples;]
                prec[i] = sum(m_method[i,:] .== [0., 0., 0., 1., 1.]) / 5
            end
            split = rand(10000, 1, 1)
            split[:,1,:] = prec
            essd_qnc[i,j] = minimum(ess(split)) / time
        else
            essd_qnc[i,j] = NaN
        end
    end
end

plot(Ms, get_medians(essd_qnc), label = names[2], color = colours[2], ribbon = get_percentiles(essd_qnc))
plot!(Ms, get_medians(essd_cm), label = names[1], color = colours[1], ribbon = get_percentiles(essd_cm))
plot!(Ms, get_medians(essd_cmf), label = names[5], color = colours[5], ribbon = get_percentiles(essd_cmf))
# plot!(Ms, get_medians(essd_shf), label = names[3], color = colours[3], ribbon = get_percentiles(essd_shf))
plot!(Ms, get_medians(essd_unif), label = names[4], color = colours[4], ribbon = get_percentiles(essd_unif), yscale = :log10, guidefontsize=20, tickfontsize=15, formatter=:plain, legendfontsize=8, legend=false, margin=5mm)
ylabel!("ESS/s")
xlabel!("coreset size")
yticks!(10. .^[-2:1:6;])
ylims!((10^2, 10^4))
xticks!(Ms[vcat(1,3:6)])
savefig("plots/essd_coreset.png")

################################
################################
# training time coreset size
################################
################################
Threads.@threads for i in 1:n_run
    println(i)
    for j in 1:dat_size
        println(j)
        m_method = load("sparse_regression_coresetMCMC_" * "0.5_0.5_" * string(Ms[j]) * "_" * string(i) * ".jld", "c_time")
        train_cm[i,j] = m_method[25000]
        m_method = load("sparse_regression_coresetMCMC_" * "0.01_" * string(Ms[j]) * "_" * string(i) * ".jld", "c_time")
        train_cmf[i,j] = m_method[25000]
        # m_method = load("sparse_regression_SHF_" * string(Ms[j]) * "_" * string(i) * ".jld", "c_time")
        # train_shf[i,j] = m_method[1]
        m_method = load("sparse_regression_uniform_" * string(Ms[j]) * "_" * string(i) * ".jld", "c_time")
        train_unif[i,j] = m_method[1]
        if isfile("sparse_regression_QNC_" * string(Ms[j]) * "_" * string(i) * ".jld")
            m_method = load("sparse_regression_QNC_" * string(Ms[j]) * "_" * string(i) * ".jld", "c_time")
            train_qnc[i,j] = m_method[50]
        else
            train_qnc[i,j] = NaN
        end
    end
end

plot(Ms, get_medians(train_qnc), label = names[2], color = colours[2], ribbon = get_percentiles(train_qnc))
plot!(Ms, get_medians(train_cm), label = names[1], color = colours[1], ribbon = get_percentiles(train_cm))
plot!(Ms, get_medians(train_cmf), label = names[5], color = colours[5], ribbon = get_percentiles(train_cmf), yscale = :log10, guidefontsize=20, tickfontsize=15, formatter=:plain, legendfontsize=10, legend=false, margin=5mm)
# plot!(Ms, get_medians(train_shf), label = names[3], color = colours[3], ribbon = get_percentiles(train_shf), yscale = :log10, guidefontsize=20, tickfontsize=15, formatter=:plain, legendfontsize=10, legend=(0.6,0.25), margin=5mm)
ylabel!("Training time (s)")
xlabel!("coreset size")
yticks!(10. .^[1:1:4;])
xticks!(Ms[vcat(1,3:6)])
ylims!((10^1, 10^4))
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
                            "train_unif", train_unif,
                            "KLd_cm", KLd_cm, 
                            "KLd_cmf", KLd_cmf,
                            "KLd_qnc", KLd_qnc, 
                            "KLd_shf", KLd_shf, 
                            "KLd_unif", KLd_unif,
                            "essd_cm", essd_cm,
                            "essd_cmf", essd_cmf,
                            "essd_qnc", essd_qnc,
                            "essd_shf", essd_shf,
                            "essd_unif", essd_unif)

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

# KLd_cm = JLD2.load("results_coresets.jld", "KLd_cm")
# KLd_cmf = JLD2.load("results_coresets.jld", "KLd_cmf")
# KLd_qnc = JLD2.load("results_coresets.jld", "KLd_qnc")
# KLd_shf = JLD2.load("results_coresets.jld", "KLd_shf")
# KLd_unif = JLD2.load("results_coresets.jld", "KLd_unif")

# essd_cm = JLD2.load("results_coresets.jld", "essd_cm")
# essd_cmf = JLD2.load("results_coresets.jld", "essd_cmf")
# essd_qnc = JLD2.load("results_coresets.jld", "essd_qnc")
# essd_shf = JLD2.load("results_coresets.jld", "essd_shf")
# essd_unif = JLD2.load("results_coresets.jld", "essd_unif")
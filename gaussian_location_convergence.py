import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import sys
import pandas as pd
import os
from scipy.optimize import minimize
from scipy.integrate import quad

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 16}
matplotlib.rc('font', **font)

def kl_heuristic(x, N, M, t, S, K, d):
    c, alpha = x # assuming gamma_t = (c*N/M)(1+t)^{alpha-1}
    if alpha < 1e-8:
        mean = N/(2*M)*np.exp(-2*c*np.log(t))
    else:
        mean = N/(2*M)*np.exp(-2*c*(t**alpha-1.)/alpha)
    #var = gamma**2*(N-S)*M**2*(K+d)*d/(2*S*(K-1))*var_integral(gamma,alpha,M,t)
    var = c*(N-S)*(K+d)*d/(4*S*(K-1))*(1+(1-alpha)*np.log(t))/t**(1-alpha)
    return mean+var

#def var_integral(gamma, alpha, M, t):
#    if alpha < 1e-8:
#        return quad( lambda tau : tau**(2*(alpha-1))*np.exp(-2*gamma*M*(np.log(t) - np.log(tau))), 1.0, t, limit=200, epsabs=1e-8, epsrel=1e-8)[0]
#    else:
#        return quad( lambda tau : tau**(2*(alpha-1))*np.exp(-2*gamma*M*(t**alpha-tau**alpha)/alpha), 1.0, t, limit=200, epsabs=1e-8, epsrel=1e-8)[0]

def tune_gamma_alpha(N, M, S, K, d, T):
    alphas = np.linspace(0, 1, 500)
    gammas = np.logspace(-12, 0, 500)
    klmin = np.inf
    alphamin = -1
    gammamin = -1
    for i in range(alphas.shape[0]):
        for j in range(gammas.shape[0]):
            kl = kl_heuristic([gammas[j], alphas[i]], N, M, T, S, K, d)
            if kl < klmin:
                klmin = kl
                alphamin = alphas[i]
                gammamin = gammas[j]
    return gammamin*N/M, alphamin

def tune_gamma(N, M, S, K, d, T, alpha):
    gammas = np.logspace(-12, 0, 500)
    klmin = np.inf
    alphamin = -1
    gammamin = -1
    for j in range(gammas.shape[0]):
        kl = kl_heuristic([gammas[j], alpha], N, M, T, S, K, d)
        if kl < klmin:
            klmin = kl
            gammamin = gammas[j]
    return gammamin*N/M


#def tune_gamma_alpha(N, M, S, K, d, T):
#    c0 = 1.
#    alpha0 = 0.5
#    x0 = np.array([c0, alpha0])
#    res = minimize(lambda x : kl_heuristic(x, N, M, T, S, K, d), x0, bounds = [(0., 0.1), (0., 1.)], tol=1e-12)
#    return res.x[0]*N/M, res.x[1]

#def tune_gamma(N, M, S, K, d, T, alpha):
#    x0 = np.array([1.])
#    res = minimize(lambda x : kl_heuristic(np.array([x[0], alpha]), N, M, T, S, K, d), x0, bounds = [(0., 0.1)], tol=1e-12)
#    return res.x*N/M

def gamma_t(t, alpha, gamma):
    return gamma*(t+1.)**(alpha-1.)

def draw_w(n, w, Y):
    d = Y.shape[0]
    mu_w = (Y*w).sum(axis=1)/(1+w.sum())
    std_w = np.sqrt(1/(1+w.sum()))
    return mu_w[:,np.newaxis] + std_w*np.random.randn(d, n)

def markov_step(beta, Z, w, Y):
    Znew = draw_w(Z.shape[1], w, Y)
    mu_w = (Y*w).sum(axis=1)/(1+w.sum())
    return np.sqrt(beta)*Z + np.sqrt(1-beta)*Znew + (1-np.sqrt(beta)-np.sqrt(1-beta))*mu_w[:,np.newaxis]

def kl(w, Y, X):
    d, N = X.shape
    ret = d*np.log((1+w.sum())/(1+N)) - d + d*(1+N)/(1+w.sum())
    dy = (Y*w).sum(axis=1)/(1+w.sum()) - X.sum(axis=1)/(1+N)
    ret += (1+N) * (dy**2).sum()
    return 0.5*ret

def dkl(w, Y, X):
    d, N = X.shape
    dy = (Y*w).sum(axis=1)/(1+w.sum()) - X.sum(axis=1)/(1+N)
    ret = (d/(1+N) - d/(1+w.sum()))*np.ones(w.shape[0])
    ret += 2*Y.T.dot(dy)
    ret -= 2*((Y*w).sum(axis=1)/(1+w.sum())).dot(dy)*np.ones(w.shape[0])
    return (N+1)/2/(1+w.sum())*ret

def est_dkl(w, Y, X, Z, S, replace):
    d, N = X.shape
    K = Z.shape[1]
    M = Y.shape[1]
    dZ = Z - (Z.sum(axis=1)/K)[:,np.newaxis]
    R = (Z**2).sum(axis=0)
    dR = R - R.sum()/K
    g1 = Y.T.dot(dZ) - 0.5*dR
    Xhat = X[:, np.random.choice(X.shape[1], size=S, replace=replace)].sum(axis=1)*(N/S)
    dy = (Y*w).sum(axis=1) - Xhat
    g2 = dZ.T.dot(dy) - (w.sum()-N)/2*dR
    return (g1*g2).sum(axis=1)/(K-1)

#def expected_kl(ts, w0, Y, X, S, K, gamma):
#    d, N = X.shape
#    M = Y.shape[1]
#    gammas = gamma(ts)
#    kl0 = (((Y*w0).sum(axis=1))**2).sum()/(2*(N+1))
#    cgammas = np.hstack((0, np.cumsum(gammas)))[:-1]
#    bias_t = np.exp(-2*M/(N+1)*cgammas)*kl0
#    noise_t = M**2*N**2*(K+d)*d*(N-S)/(2*S*(K-1)*(N-1)*(N+1)**3) * (gammas**2*np.exp(-2*M/(N+1)*cgammas)).sum() # NOTE: the last term here is slightly wrong due to indexing (off by one). shouldn't affect asymptotics
#    return bias_t + noise_t


np.random.seed(0)
d = 20 # dimension
N = 10000 # number of data pts
X = np.random.randn(d, N) # dataset
T = 100000 # total number of iterations
L = 1 # number of trials

## varying S test
#if False:
#    # Coreset MCMC, KL vs work done & KL vs iteration, varying S
#    # test with/without replace
#    alpha = 0.5 # learning rate power
#    beta = 0.5 # uniform ergodicity constant (0 is iid, 1 is no mixing)
#    K = d
#    M = 30 # number of coreset pts
#    Y = X[:, :M] # uniformly random coreset pts (because X is drawn iid, can just take first M)
#    replace = False
#    project = False
#    S = np.array([1, 10, 100, 1000, 10000])
#    fn = f"vary_S_replace_{replace}_beta_{beta}_alpha_{alpha}.csv"
#    if os.path.exists(fn):
#        df = pd.read_csv(fn)
#    else:
#        df = pd.DataFrame(columns=["trial", "subsample_size", "iteration", "work", "KL"])
#    for ell in range(L):
#        print(f"Trial {ell+1}/{L}")
#        for i in range(S.shape[0]):
#            if df[ (df["subsample_size"] == S[i]) & (df["trial"] == ell) ].empty:
#                np.random.seed(ell*S.shape[0]+i)
#                print(f"S = {S[i]}, i = {i+1}/{S.shape[0]}")
#                # initialize at even weighting
#                w = (N/M)*np.ones(M)
#                Z = draw_w(K, w, Y)
#                C = 0.01*N/M *(M+S[i])**alpha # learning rate constant
#                Ws = np.zeros(T)
#                KLs = np.zeros(T)
#                sumws = np.zeros(T)
#                KLs[0] = kl(w,Y,X)
#                sumws[0] = w.sum()
#                for t in range(1, T):
#                    if t % 100 == 0:
#                        sys.stdout.write(f"t = {t+1}/{T}              \r")
#                        sys.stdout.flush()
#                    # projected gradient estimate
#                    Z = markov_step(beta, Z, w, Y)
#                    g = est_dkl(w, Y, X, Z, S[i], replace=replace)
#                    if project:
#                        g -= g.sum()/g.shape[0]*np.ones(g.shape[0])
#                    w -= gamma(t, alpha, C)*g
#                    # record KL / work done
#                    KLs[t] = kl(w,Y,X)
#                    sumws[t] = w.sum()
#                    Ws[t] = Ws[t-1] + (M+S[i])*d
#                # write to df and save
#                dfi = pd.DataFrame({"trial": ell*np.ones(T),
#                                    "subsample_size": S[i]*np.ones(T),
#                                    "iteration": np.arange(T),
#                                    "work": Ws,
#                                    "KL": KLs,
#                                    "sumw": sumws
#                                })
#                df = pd.concat((df, dfi), ignore_index=True)
#                df.to_csv(fn, index=False)
#            else:
#                print(f"Already ran S = {S[i]}, i = {i+1}/{S.shape[0]}")
#            print("")
#
#    # plot the results
#    fig = plt.figure()
#    ax = fig.add_subplot(111)
#    for i in range(S.shape[0]):
#        # plot individual traces
#        subdf = df[(df["subsample_size"] == S[i]) & (df["trial"] == 0)]
#        line, = ax.plot(subdf["work"], subdf["KL"], label=f"S = {S[i]}")
#        for ell in range(1, L):
#            subdf = df[(df["subsample_size"] == S[i]) & (df["trial"] == ell)]
#            ax.plot(subdf["work"], subdf["KL"], color=line.get_color())
#    ax.set_yscale('log')
#    ax.set_xscale('log')
#    ax.set_title(fn)
#    ax.set_ylabel('KL Divergence')
#    ax.set_xlabel('Cost')
#    ax.legend()
#    plt.show()
#
#    # plot the results
#    fig = plt.figure()
#    ax = fig.add_subplot(111)
#    for i in range(S.shape[0]):
#        # plot individual traces
#        subdf = df[(df["subsample_size"] == S[i]) & (df["trial"] == 0)]
#        line, = ax.plot(subdf["work"], subdf["sumw"], label=f"S = {S[i]}")
#        for ell in range(1, L):
#            subdf = df[(df["subsample_size"] == S[i]) & (df["trial"] == ell)]
#            ax.plot(subdf["work"], subdf["sumw"], color=line.get_color())
#    ax.set_yscale('log')
#    ax.set_xscale('log')
#    ax.set_title(fn)
#    ax.set_ylabel('Sum of Weights')
#    ax.set_xlabel('Cost')
#    ax.legend()
#    plt.show()

# varying K test
if False:
    beta = 0.8 # uniform ergodicity constant (0 is iid, 1 is no mixing)
    M = 30 # number of coreset pts
    Y = X[:, :M] # uniformly random coreset pts (because X is drawn iid, can just take first M)
    replace = False
    project = True
    S = M
    Ks = np.array([2, 10, 20, 50, 100, 500])
    fn = f"vary_K_replace_{replace}_project_{project}_beta_{beta}_S_{S}_M_{M}.csv"
    if os.path.exists(fn):
        df = pd.read_csv(fn)
    else:
        df = pd.DataFrame(columns=["trial", "K", "iteration", "work", "KL", "gamma", "alpha"])
    for ell in range(L):
        print(f"Trial {ell+1}/{L}")
        for i in range(Ks.shape[0]):
            if df[ (df["K"] == Ks[i]) & (df["trial"] == ell) ].empty:
                np.random.seed(ell*Ks.shape[0]+i)
                print(f"K = {Ks[i]}, i = {i+1}/{Ks.shape[0]}")
                # initialize at even weighting
                #gamma, alpha = tune_gamma_alpha(N, M, S, Ks[i], d, 10000)
                #alpha = 0.5
                #gamma = tune_gamma(N, M, S, Ks[i], d, 10000, alpha)
                gamma = 0.05*N/M
                alpha = 1.0
                gamma_max = 0.1*N/M
                gamma = min(gamma, gamma_max)
                print(f"Tuned gamma, alpha: {gamma}, {alpha}")
                w = (N/M)*np.ones(M)
                Z = draw_w(Ks[i], w, Y)
                Ws = np.zeros(T)
                KLs = np.zeros(T)
                sumws = np.zeros(T)
                KLs[0] = kl(w,Y,X)
                sumws[0] = w.sum()
                for t in range(1, T):
                    if t % 100 == 0:
                        sys.stdout.write(f"t = {t+1}/{T}              \r")
                        sys.stdout.flush()
                    # projected gradient estimate
                    Z = markov_step(beta, Z, w, Y)
                    g = est_dkl(w, Y, X, Z, S, replace=replace)
                    if project:
                        g -= g.sum()/g.shape[0]*np.ones(g.shape[0])
                    w -= gamma_t(t, alpha, gamma)*g
                    # record KL / work done
                    KLs[t] = kl(w,Y,X)
                    sumws[t] = w.sum()
                    Ws[t] = Ws[t-1] + (M+S)*d
                # write to df and save
                dfi = pd.DataFrame({"trial": ell*np.ones(T),
                                    "K": Ks[i]*np.ones(T),
                                    "iteration": np.arange(T),
                                    "gamma": gamma*np.ones(T),
                                    "alpha": alpha*np.ones(T),
                                    "work": Ws,
                                    "KL": KLs,
                                    "sumw": sumws
                                })
                df = pd.concat((df, dfi), ignore_index=True)
                df.to_csv(fn, index=False)
            else:
                print(f"Already ran K = {Ks[i]}, i = {i+1}/{Ks.shape[0]}")
            print("")

    # plot the KLs
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(Ks.shape[0]):
        # plot individual traces
        subdf = df[(df["K"] == Ks[i]) & (df["trial"] == 0)]
        line, = ax.plot(subdf["work"], subdf["KL"], label=f"K = {Ks[i]}")
        for ell in range(1, L):
            subdf = df[(df["K"] == Ks[i]) & (df["trial"] == ell)]
            ax.plot(subdf["work"], subdf["KL"], color=line.get_color())
    ax.set_yscale('log')
    ax.set_xscale('log')
    #ax.set_title(fn)
    ax.set_ylabel('KL Divergence')
    ax.set_xlabel('Cost')
    ax.legend()
    plt.show()

    # plot the sumws
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(Ks.shape[0]):
        # plot individual traces
        subdf = df[(df["K"] == Ks[i]) & (df["trial"] == 0)]
        line, = ax.plot(subdf["iteration"], subdf["sumw"], label=f"K = {Ks[i]}")
        for ell in range(1, L):
            subdf = df[(df["K"] == Ks[i]) & (df["trial"] == ell)]
            ax.plot(subdf["iteration"], subdf["sumw"], color=line.get_color())
    ax.set_yscale('log')
    ax.set_xscale('log')
    #ax.set_title(fn)
    ax.set_ylabel('Sum of Weights')
    ax.set_xlabel('Iteration')
    ax.legend()
    plt.show()

    # plot the gammas
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for ell in range(L):
        gammas = np.zeros(Ks.shape[0])
        for i in range(Ks.shape[0]):
            gammas[i] = df.loc[(df["K"] == Ks[i]) & (df["trial"] == ell), "gamma"].iloc[0]
        line, = ax.plot(Ks, gammas)
    ax.set_yscale('log')
    ax.set_xscale('log')
    #ax.set_title(fn)
    ax.set_ylabel('Gamma')
    ax.set_xlabel('K')
    ax.legend()
    plt.show()

    # plot the alphas
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for ell in range(L):
        alphas = np.zeros(Ks.shape[0])
        for i in range(Ks.shape[0]):
            alphas[i] = df.loc[(df["K"] == Ks[i]) & (df["trial"] == ell), "alpha"].iloc[0]
        line, = ax.plot(Ks, alphas)
    ax.set_yscale('log')
    ax.set_xscale('log')
    #ax.set_title(fn)
    ax.set_ylabel('Alpha')
    ax.set_xlabel('K')
    ax.legend()
    plt.show()

# varying M test
if False:
    # Coreset MCMC, KL vs work done & KL vs iteration, varying K
    # test with/without replace
    #alpha = 0.5 # learning rate power
    beta = 0.8 # uniform ergodicity constant (0 is iid, 1 is no mixing)
    K = d # number of coreset pts
    replace = False
    project = True
    Ms = np.array([2, 10, 20, 25, 30, 40, 100, 200, 400])
    S = 30
    fn = f"vary_M_replace_{replace}_beta_{beta}_S_{S}.csv"
    if os.path.exists(fn):
        df = pd.read_csv(fn)
    else:
        df = pd.DataFrame(columns=["trial", "M", "iteration", "work", "KL"])
    for ell in range(L):
        print(f"Trial {ell+1}/{L}")
        for i in range(Ms.shape[0]):
            if df[ (df["M"] == Ms[i]) & (df["trial"] == ell) ].empty:
                np.random.seed(ell*Ms.shape[0]+i)
                print(f"M = {Ms[i]}, i = {i+1}/{Ms.shape[0]}")
                # initialize at even weighting
                M = Ms[i]
                Y = X[:, :M] # uniformly random coreset pts (because X is drawn iid, can just take first M)
                #C = 0.1*N/M #* (M+S)**alpha # learning rate constant
                # initialize at even weighting
                #gamma, alpha = tune_gamma_alpha(N, Ms[i], S, K, d, 10000)
                gamma = 0.1*N/M
                alpha = 0.5
                print(f"Tuned gamma, alpha: {gamma}, {alpha}")
                #alpha = 0.5
                #gamma = tune_gamma(N, Ms[i], S, K, d, 10000, alpha)
                #print(f"Tuned gamma, alpha=0.5: {gamma}")
                w = (N/M)*np.ones(M)
                Z = draw_w(Ms[i], w, Y)
                Ws = np.zeros(T)
                KLs = np.zeros(T)
                sumws = np.zeros(T)
                KLs[0] = kl(w,Y,X)
                sumws[0] = w.sum()
                for t in range(1, T):
                    if t % 100 == 0:
                        sys.stdout.write(f"t = {t+1}/{T}              \r")
                        sys.stdout.flush()
                    # projected gradient estimate
                    Z = markov_step(beta, Z, w, Y)
                    g = est_dkl(w, Y, X, Z, S, replace=replace)
                    if project:
                        g -= g.sum()/g.shape[0]*np.ones(g.shape[0])
                    w -= gamma_t(t, alpha, gamma)*g
                    # record KL / work done
                    KLs[t] = kl(w,Y,X)
                    sumws[t] = w.sum()
                    Ws[t] = Ws[t-1] + (M+S)*d
                # write to df and save
                dfi = pd.DataFrame({"trial": ell*np.ones(T),
                                    "M": Ms[i]*np.ones(T),
                                    "iteration": np.arange(T),
                                    "gamma": gamma*np.ones(T),
                                    "alpha": alpha*np.ones(T),
                                    "work": Ws,
                                    "KL": KLs,
                                    "sumw": sumws
                                })
                df = pd.concat((df, dfi), ignore_index=True)
                df.to_csv(fn, index=False)
            else:
                print(f"Already ran M = {Ms[i]}, i = {i+1}/{Ms.shape[0]}")
            print("")

    # plot the results
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(Ms.shape[0]):
        # plot individual traces
        subdf = df[(df["M"] == Ms[i]) & (df["trial"] == 0)]
        line, = ax.plot(subdf["work"], subdf["KL"], label=f"M = {Ms[i]}")
        for ell in range(1, L):
            subdf = df[(df["M"] == Ms[i]) & (df["trial"] == ell)]
            ax.plot(subdf["work"], subdf["KL"], color=line.get_color())
    ax.set_yscale('log')
    ax.set_xscale('log')
    #ax.set_title(fn)
    ax.set_ylabel('KL Divergence')
    ax.set_xlabel('Cost')
    ax.legend(loc='upper right', fontsize=12)
    plt.show()

    # plot the results
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(Ms.shape[0]):
        # plot individual traces
        subdf = df[(df["M"] == Ms[i]) & (df["trial"] == 0)]
        line, = ax.plot(subdf["iteration"], subdf["sumw"], label=f"M = {Ms[i]}")
        for ell in range(1, L):
            subdf = df[(df["M"] == Ms[i]) & (df["trial"] == ell)]
            ax.plot(subdf["iteration"], subdf["sumw"], color=line.get_color())
    ax.set_yscale('log')
    ax.set_xscale('log')
    #ax.set_title(fn)
    ax.set_ylabel('Sum of Weights')
    ax.set_xlabel('Iteration')
    ax.legend(loc='upper right', fontsize=12)
    plt.show()

    # plot the gammas
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for ell in range(L):
        gammas = np.zeros(Ms.shape[0])
        for i in range(Ms.shape[0]):
            gammas[i] = df.loc[(df["M"] == Ms[i]) & (df["trial"] == ell), "gamma"].iloc[0]
        line, = ax.plot(Ms, gammas)
    ax.set_yscale('log')
    ax.set_xscale('log')
    #ax.set_title(fn)
    ax.set_ylabel('Gamma')
    ax.set_xlabel('M')
    ax.legend()
    plt.show()

    # plot the alphas
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for ell in range(L):
        alphas = np.zeros(Ms.shape[0])
        for i in range(Ms.shape[0]):
            alphas[i] = df.loc[(df["M"] == Ms[i]) & (df["trial"] == ell), "alpha"].iloc[0]
        line, = ax.plot(Ms, alphas)
    ax.set_yscale('log')
    ax.set_xscale('log')
    #ax.set_title(fn)
    ax.set_ylabel('Alpha')
    ax.set_xlabel('M')
    ax.legend()
    plt.show()

# varying beta test
if True:
    # Coreset MCMC, KL vs work done & KL vs iteration, varying K
    # test with/without replace
    #alpha = 0.5 # learning rate power
    K = d # number of coreset pts
    replace = False
    project = True
    betas = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99])
    M = 30
    S = M
    fn = f"vary_beta_replace_{replace}_project_{project}_M_{M}_S_{S}.csv"
    if os.path.exists(fn):
        df = pd.read_csv(fn)
    else:
        df = pd.DataFrame(columns=["trial", "beta", "iteration", "work", "KL"])
    for ell in range(L):
        print(f"Trial {ell+1}/{L}")
        for i in range(betas.shape[0]):
            if df[ (df["beta"] == betas[i]) & (df["trial"] == ell) ].empty:
                np.random.seed(ell*betas.shape[0]+i)
                print(f"beta = {betas[i]}, i = {i+1}/{betas.shape[0]}")
                # initialize at even weighting
                beta = betas[i]
                Y = X[:, :M] # uniformly random coreset pts (because X is drawn iid, can just take first M)
                #C = 0.1*N/M #* (M+S)**alpha # learning rate constant
                # initialize at even weighting
                # gamma, alpha = tune_gamma_alpha(N, M, S, K, d, 10000)
                # print(f"Tuned gamma, alpha: {gamma}, {alpha}")
                alpha = 0.5
                gamma = 0.1*N/M
                #gamma = tune_gamma(N, M, S, K, d, 10000, alpha)
                #print(f"Tuned gamma, alpha: {gamma}, {alpha}")
                w = (N/M)*np.ones(M)
                Z = draw_w(M, w, Y)
                Ws = np.zeros(T)
                KLs = np.zeros(T)
                sumws = np.zeros(T)
                KLs[0] = kl(w,Y,X)
                sumws[0] = w.sum()
                for t in range(1, T):
                    if t % 100 == 0:
                        sys.stdout.write(f"t = {t+1}/{T}              \r")
                        sys.stdout.flush()
                    # projected gradient estimate
                    Z = markov_step(beta, Z, w, Y)
                    g = est_dkl(w, Y, X, Z, S, replace=replace)
                    if project:
                        g -= g.sum()/g.shape[0]*np.ones(g.shape[0])
                    w -= gamma_t(t, alpha, gamma)*g
                    # record KL / work done
                    KLs[t] = kl(w,Y,X)
                    sumws[t] = w.sum()
                    Ws[t] = Ws[t-1] + (M+S)*d
                # write to df and save
                dfi = pd.DataFrame({"trial": ell*np.ones(T),
                                    "beta": betas[i]*np.ones(T),
                                    "iteration": np.arange(T),
                                    "gamma": gamma*np.ones(T),
                                    "alpha": alpha*np.ones(T),
                                    "work": Ws,
                                    "KL": KLs,
                                    "sumw": sumws
                                })
                df = pd.concat((df, dfi), ignore_index=True)
                df.to_csv(fn, index=False)
            else:
                print(f"Already ran M = {betas[i]}, i = {i+1}/{betas.shape[0]}")
            print("")

    # plot the results
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(betas.shape[0]):
        # plot individual traces
        subdf = df[(df["beta"] == betas[i]) & (df["trial"] == 0)]
        line, = ax.plot(subdf["work"], subdf["KL"], label=f"Beta = {betas[i]}")
        for ell in range(1, L):
            subdf = df[(df["beta"] == betas[i]) & (df["trial"] == ell)]
            ax.plot(subdf["work"], subdf["KL"], color=line.get_color())
    ax.set_yscale('log')
    ax.set_xscale('log')
    #ax.set_title(fn)
    ax.set_ylabel('KL Divergence')
    ax.set_xlabel('Cost')
    ax.legend(loc='lower left', fontsize=12)
    plt.show()

    # plot the results
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(betas.shape[0]):
        # plot individual traces
        subdf = df[(df["beta"] == betas[i]) & (df["trial"] == 0)]
        line, = ax.plot(subdf["iteration"], subdf["sumw"], label=f"Beta = {betas[i]}")
        for ell in range(1, L):
            subdf = df[(df["beta"] == betas[i]) & (df["trial"] == ell)]
            ax.plot(subdf["iteration"], subdf["sumw"], color=line.get_color())
    ax.set_yscale('log')
    ax.set_xscale('log')
    #ax.set_title(fn)
    ax.set_ylabel('Sum of Weights')
    ax.set_xlabel('Iteration')
    ax.legend(loc='upper right', fontsize=12)
    plt.show()

    # plot the gammas
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for ell in range(L):
        gammas = np.zeros(betas.shape[0])
        for i in range(betas.shape[0]):
            gammas[i] = df.loc[(df["beta"] == betas[i]) & (df["trial"] == ell), "gamma"].iloc[0]
        line, = ax.plot(betas, gammas)
    ax.set_yscale('log')
    ax.set_xscale('log')
    #ax.set_title(fn)
    ax.set_ylabel('Gamma')
    ax.set_xlabel('Beta')
    ax.legend()
    plt.show()

    # plot the alphas
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for ell in range(L):
        alphas = np.zeros(betas.shape[0])
        for i in range(betas.shape[0]):
            alphas[i] = df.loc[(df["beta"] == betas[i]) & (df["trial"] == ell), "alpha"].iloc[0]
        line, = ax.plot(betas, alphas)
    ax.set_yscale('log')
    ax.set_xscale('log')
    #ax.set_title(fn)
    ax.set_ylabel('Alpha')
    ax.set_xlabel('Beta')
    ax.legend()
    plt.show()



# test 3: Coreset MCMC, KL vs work done & KL vs iteration, factor: K (with replacement)
# test 4: Coreset MCMC, KL vs work done & KL vs iteration, factor: K (without replacement)
# test 5:




#K = 5 # number of Markov chains
#Z = draw_w(K, w, Y)
#
#
#
#ts = np.arange(10000)
#S = np.sqrt(N)
#kls = expected_kl(ts, w, Y, X, S, K, lambda t : N/M*(1/np.sqrt(N))*(1./np.sqrt(1+t)))
#
#plt.plot(ts, kls)
#plt.yscale('log')
#plt.show()
#quit()




# # test full/stochastic gradient (these tests pass)
# g = dkl(w,Y,X)
# gn = np.zeros(g.shape)
# for i in range(g.shape[0]):
#     wi = w.copy()
#     wi[i] += 1e-4
#     gn[i] = (kl(wi, Y, X) - kl(w, Y, X))/1e-4
# print(g)
# print(gn)
# gh = np.zeros(g.shape)
# B = 100000
# for i in range(B):
#     gh += est_dkl(w,Y,X,Z)
#     Z = markov_step(0.0, Z, w, Y)
# print(g)
# print(gh/B)
# quit()

#
## full gradient descent
#for i in range(1000000):
#    print(f"Current KL: {kl(w,Y,X)} Current wsum: {w.sum()}")
#    w = w - 50*dkl(w,Y,X)
#quit()
#
## stochastic gradient descent
#for i in range(1000):
#    print(f"Current KL: {kl(w,Y,X)} Current wsum: {w.sum()}")
#



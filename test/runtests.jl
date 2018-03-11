using Distributions
using LinearManifoldModels
import StatsBase
import KernelDensity
import LMCLUS
using Base.Test

const SEED = 986163731

# generate dataset
srand(SEED)
X1 = rand(Normal(0.), 2, 100)
X2 = rand(MvNormal([2., 2.], diagm([0.02, 0.6])), 100)
X = [X1 X2]
Xn = X .- mean(X,2)

# generate clusters
p = LMCLUS.Parameters(1)
p.random_seed = SEED
p.sampling_heuristic = 1
p.sampling_factor = 1.5
p.max_bin_portion = 0.2
p.best_bound = 0.55
p.basis_alignment = true
p.dim_adjustment = true
p.dim_adjustment_ratio = 0.5
CL = LMCLUS.lmclus(Xn, p)

# test distribution properties
@testset for E in [StatsBase.Histogram, KernelDensity.UnivariateKDE]
    for C in LMCLUS.manifolds(CL)
        lmcm = estimate(NonparametricEstimator(EmpiricalLinearManifold, E), C, Xn[:,LMCLUS.labels(C)])
        @test length(lmcm) == 2
        @test params(lmcm) == (mean(C), LMCLUS.projection(C))
        @test partype(lmcm) == typeof(Xn[1])
        @test pdf(lmcm, Xn[:,1]) >= 0.
        @test all(pdf(lmcm, Xn) .>= 0.)
        @test logpdf(lmcm, Xn[:,1]) <= 10.
        @test all(logpdf(lmcm, Xn) .<= 10.)
        @test entropy(lmcm) > 0.
    end
end

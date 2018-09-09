### Fitting by KDE

function calcrange(::Type{KernelDensity.UnivariateKDE}, x::AbstractVector, binNumber)
    bw = KernelDensity.default_bandwidth(x)
    bb = KernelDensity.kde_boundary(x, bw)
    mps = Float64[]
    i = 1
    while length(mps) < binNumber # range correction
        stp = (bb[2]-bb[1])/(binNumber-i)
        mps = bb[1]:stp:bb[2]+eps()
        # println("$i: $binNumber => $(bb[1]):$stp:$(bb[2]) ", length(mps))
        i -=1
    end
    return mps
end

function fit(::Type{EmpiricalLinearManifold}, ::Type{KernelDensity.UnivariateKDE},
             μ::AbstractVector{T}, B::AbstractMatrix{T}, X::AbstractMatrix{T}; kvargs...) where {T <: Real}

    K = size(B,2)

    nquants = 1024 # default
    for (k,v) in kvargs
        if k == :nquants
            nquants = Int(v)
        end
    end

    estimates = Array{KernelDensity.UnivariateKDE}(undef, K+1)

    zk = B'*(X .- μ)
    for k in 1:K
        estimates[k] = KernelDensity.kde(zk[k,:], calcrange(KernelDensity.UnivariateKDE, zk[k,:], nquants))
    end

    d = LMCLUS.distance_to_manifold(X, μ, B)
    estimates[K+1] = KernelDensity.kde(d, calcrange(KernelDensity.UnivariateKDE, d, nquants))

	return EmpiricalLinearManifold(μ, B, estimates)
end

fit(::Type{EmpiricalLinearManifold}, ::Type{KernelDensity.UnivariateKDE},
    LMC::LMCLUS.Manifold, X::AbstractMatrix{T}; kvargs...) where {T <: Real} =
    fit(EmpiricalLinearManifold, KernelDensity.UnivariateKDE, LMCLUS.mean(LMC), LMCLUS.projection(LMC), X; kvargs...)


### Evaluation

rectint(y, r) = y*r
rectint(y, x1, x2) = rectint(y, x2 - x1)
trapint(y1, y2, r) = r.*(y1.+y2)/2
trapint(y1, y2, x1, x2) = trapint(y1, y2, x2 .- x1)
simpint(y1, y2, y3, x1, x2) = (x2 .- x1).*(y1 .+ 4*y2 .+ y3)/6

function _pdf(d::EmpiricalLinearManifold{T, KernelDensity.UnivariateKDE},
              x::AbstractVector{T}) where {T<:Real}
    K = size(d.B,2)
    prob = one(T)
    z = d.B'*(x - d.μ)
    dist = LMCLUS.distance_to_manifold(collect(x), d.μ, d.B) |> first
    for k in 1:K+1
        val = k > K ? dist : z[k] # Orthonormal subspace distances for K+1
        stp = step(d.estimate[k].x)

        # Search-based
        # binIndex = searchsortedlast(d.estimate[k].x, val)
        # dv, x1, x2 = if binIndex < 1
        #     d.estimate[k].density[1], d.estimate[k].x[1]-stp, d.estimate[k].x[1]
        # elseif binIndex > length(d.estimate[k].x)
        #     d.estimate[k].density[end], d.estimate[k].x[end], d.estimate[k].x[end]+stp
        # else
        #     d.estimate[k].density[binIndex], d.estimate[k].x[binIndex], d.estimate[k].x[binIndex]+stp
        # end
        # tmp = rectint(dv, x1, x2)

        # Approximation-based
        x1 = val-stp/2
        x2 = val
        x3 = val+stp/2
        # tmp = rectint(pdf(d.estimate[k], x2), x1, x3)
        tmp = trapint(pdf(d.estimate[k], x1), pdf(d.estimate[k], x3), x1, x3)
        # tmp = simpint(pdf(d.estimate[k], x1), pdf(d.estimate[k], x2), pdf(d.estimate[k], x3), x1, x3)

        # if probability is negative
        if tmp > 0.
            prob *= tmp
        else
            prob = 0.
        end
    end
    return prob
end

function _pdf!(r::AbstractArray{T}, d::EmpiricalLinearManifold{T,KernelDensity.UnivariateKDE},
               X::AbstractMatrix{T}) where {T<:Real}
    K = size(d.B,2)
    N, n = size(X)
    Z = (d.B'*(X .- d.μ))'
    dist = LMCLUS.distance_to_manifold(X, d.μ, d.B)
    for i in 1 : n
        @inbounds r[i] = one(T)
    end
    for k in 1 : K+1
        val = k > K ? dist : Z[:,k] # Orthonormal subspace distances for K+1
        stp = step(d.estimate[k].x)
        x1 = val .- stp/2
        x3 = val .+ stp/2
        tmp = trapint(pdf(d.estimate[k], x1), pdf(d.estimate[k], x3), x1, x3)
        # tmp = simpint(pdf(d.estimate[k], x1), pdf(d.estimate[k], val), pdf(d.estimate[k], x3), x1, x3)
        for i in 1 : n
            @inbounds r[i] *= tmp[i] < zero(T) ? zero(T) : tmp[i]
        end
    end
    return r
end


### Entropy

# function Distributions.entropy(d::EmpiricalLinearManifold{T, KernelDensity.UnivariateKDE}) where {T<:Real}
#     H = 0.0
#     for est in d.estimate
#         tot = length(est.density)
#         ds = filter(x->x != zero(T), est.density)
#         H += -sum(log(ds))/tot
#     end
#     return H
# end

function entropy(d::EmpiricalLinearManifold{T, KernelDensity.UnivariateKDE}) where {T<:Real}
    H = 0.0
    qb = first(d.estimate).x |> length
    dp = ones(qb-1)
    for est in d.estimate
        stp = step(est.x)
        tmp = [trapint(est.density[i], est.density[i+1], est.x[i], est.x[i+1]) for i in 1:length(est.density)-1]
        dp = dp .* tmp
    end
    H = entropy(dp)
    @debug "Empirical LM model entropy" EPDF=dp H=H
    return H
end

function entropy(M::MixtureModel{Q,R,EmpiricalLinearManifold{T, KernelDensity.UnivariateKDE}}) where {Q,R,T<:Real}
    qb = first(first(components(M)).estimate).x |> length
    c = length(components(M))
    H = zero(T)
    for i in 1:qb-1
        Pᵢⱼ = zero(T)
        for j in 1:c
            d = components(M)[j]
            pⱼ = probs(M)[j]
            prob = one(T)
            for est in d.estimate
                prob *= trapint(est.density[i], est.density[i+1], est.x[i], est.x[i+1])
            end
            Pᵢⱼ += pⱼ*prob
            @debug "Pᵢⱼ" i j Pᵢⱼ
        end
        H += Pᵢⱼ > zero(T) ? -log(Pᵢⱼ) : zero(T)
        # H += Pᵢⱼ > zero(T) ? -Pᵢⱼ*log(Pᵢⱼ) : zero(T)
        @debug "H" i H
    end
    @debug "Empirical LM mixture model entropy" H=H
    return H
end


### Sampling

function _rand!(s::EmpiricalLinearManifoldSampler{T, KernelDensity.UnivariateKDE}, x::AbstractVector{T}) where {T <: Real}
    return x
end

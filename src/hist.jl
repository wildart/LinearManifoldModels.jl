### Fitting by histogram

function calcrange(::Type{Histogram}, x::AbstractVector, binNumber)
    histExtrema = extrema(x)
    histRange = histExtrema[2] - histExtrema[1]
    binSize = histRange/binNumber
    edges = collect(histExtrema[1]:binSize:histExtrema[2]+eps())
    edges[1] -= eps()
    edges[end] += eps()
    return edges
end

function fit(::Type{EmpiricalLinearManifold}, ::Type{Histogram},
             μ::AbstractVector{T}, B::AbstractMatrix{T}, X::AbstractMatrix{T};
             kvargs...) where {T <: Real}

    N, n = size(X)
    K = size(B,2)

    nbins = sturges(n)
    for (k,v) in kvargs
        if k == :nbins
            nbins = Int(v)
        end
    end

    estimates = Array{Histogram}(undef, K+1)

    zk = B'*(X .- μ)
    for k in 1:K
        histEdges = calcrange(Histogram, zk[k,:], nbins)
        estimates[k] = normalize(fit(Histogram, zk[k,:], histEdges, closed=:left), mode=:probability)
    end

    d = LMCLUS.distance_to_manifold(X, μ, B)
    histEdges = calcrange(Histogram, d, nbins)
    estimates[K+1] = normalize(fit(Histogram, d, histEdges, closed=:left), mode=:probability)

	return EmpiricalLinearManifold(μ, B, estimates)
end

fit(::Type{EmpiricalLinearManifold}, ::Type{Histogram},
    LMC::LMCLUS.Manifold, X::AbstractMatrix{T}; kvargs...) where {T <: Real} =
    fit(EmpiricalLinearManifold, Histogram, LMCLUS.mean(LMC), LMCLUS.projection(LMC), X; kvargs...)


### Evaluation

function _pdf(d::EmpiricalLinearManifold{T, Histogram}, x::AbstractVector{T}) where {T<:Real}
    # Basis subspace
    K = size(d.B,2)
    probValues = zeros(T, K+1)
    binCount = length(d.estimate[1].weights)
    z = d.B'*(x - d.μ)
    dist = LMCLUS.distance_to_manifold(x, d.μ, d.B) |> first
    for k in 1:K+1
        val = k > K ? dist : z[k] # Orthonormal subspace distances for K+1
        edges = first(d.estimate[k].edges)
        binIndex = searchsortedlast(edges, val, lt=(x,y)->x<=y)
        if 0 < binIndex <= binCount
            probValues[k] = d.estimate[k].weights[binIndex]
        end
    end
    return prod(probValues)
end


### Entropy

function entropy(d::EmpiricalLinearManifold{T, Histogram}) where {T<:Real}
    H = 0.0
    for est in d.estimate
        dp = est.weights
        filter!(x->x != zero(T), dp)
        H += entropy(dp)
        @debug "Empirical PDF" EPDF=dp H=H
    end
    @debug "Empirical LM model entropy" H=H
    return H
end

function entropy(M::MixtureModel{Q,R,EmpiricalLinearManifold{T, Histogram}}) where {Q,R,T<:Real}
    qb = first(first(components(M)).estimate).weights |> length
    c = length(components(M))
    H = zero(T)
    for i in 1:qb
        Pᵢⱼ = zero(T)
        for j in 1:c
            d = components(M)[j]
            pⱼ = probs(M)[j]
            prob = one(T)
            for est in d.estimate
                prob *= est.weights[i]/sum(est.weights)
            end
            Pᵢⱼ += pⱼ*prob
            @debug "Pᵢⱼ" i j Pᵢⱼ
        end
        H += Pᵢⱼ > zero(T) ? -Pᵢⱼ*log(Pᵢⱼ) : zero(T)
        @debug "H" i H
    end
    @debug "Empirical LM mixture model entropy" H=H
    return H
end


### Sampling

function _rand!(s::EmpiricalLinearManifoldSampler{T, Histogram}, x::AbstractVector{T}) where T <: Real
    return x
end

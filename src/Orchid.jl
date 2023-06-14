module Orchid

using SparseArrays
import OptimalTransport, ThreadsX

abstract type DisperseUnweightedClique end
abstract type DisperseWeightedClique end
abstract type DisperseUnweightedStar end

function disperse(::Type{DisperseUnweightedClique}, node, alpha, neighbors, _...)
    total = length(neighbors)
    N = neighbors[node]
    if isempty(N) || length(N) == 1
        sparsevec(Int64[node], Float64[1.0], total)
    else
        x = sparsevec(N, (1 - alpha) / (length(N) - 1), total)
        x[node] = alpha
        x
    end
end

function disperse(::Type{DisperseWeightedClique}, node, alpha, neighbors, rc, cr, rw::Vector, _...)
    total = length(rc)
    N = neighbors[node]
    W = rw[node]
    if isempty(N) || length(N) == 1
        sparsevec(Int64[node], Float64[1.0], total)
    else
        factor = (1 - alpha) / (sum(W) - W[findfirst(==(node), N)])
        x = sparsevec(N, W .* factor, total)
        x[node] = alpha
        x
    end
end

function disperse(::Type{DisperseUnweightedStar}, node, alpha, neighbors, rc, cr, _...)
    total = length(rc)
    dispersion = sparsevec(Int64[node], Float64[1.0], total)
    k = 0
    for e in rc[node]
        k += length(cr[e]) > 1
        for x in cr[e]
            dispersion[x] += (1 - alpha) / (length(cr[e]) - 1)
        end
    end
    if k > 0
        dispersion ./= k
    end
    dispersion[node] = alpha
    dispersion
end

abstract type CostMatrix end
abstract type CostOndemand end

function prepare_cost_matrix(::Type{CostMatrix}, neighbors)
    @info "Preparing Cost Matrix"

    K = length(neighbors)
    C = fill(Int8(3), (K, K))
    Threads.@threads for m in 1:K
        N = neighbors[m]
        for n in N
            C[n, m] = C[m, n] = 1
        end
        for i = eachindex(N), j = i:length(N)
            s, t = N[i], N[j]
            if s != t && s != m && C[s, t] == 3
                C[t, s] = C[s, t] = 2
            end
        end
        C[m, m] = 0
    end
    C
end

function prepare_cost_matrix(::Type{CostOndemand}, neighbors)
    @info "Preparing Ondemand Cost Computation"
    ThreadsX.map(BitSet, neighbors)
end

function any_bits(f, s::BitSet, t::BitSet)
    a1, b1 = s.bits, s.offset
    a2, b2 = t.bits, t.offset
    l1, l2 = length(a1), length(a2)
    bdiff = b2 - b1
    @inbounds for i = max(1, 1 + bdiff):min(l1, l2 + bdiff)
        f(a1[i], a2[i-bdiff]) && return true
    end
    return false
end
# Efficiently checks whether Base.BitSets intersect using BitSet internals.
intersects(u::BitSet, v::BitSet) = any_bits((a, b) -> (a & b) != 0, u, v)

@inbounds function truncated_cost(m::Int, n::Int, neighbors::Vector)
    if n == m
        0
    elseif n in neighbors[m]
        1
    elseif intersects(neighbors[n], neighbors[m])
        2
    else
        3
    end
end

get_cost_submatrix(C::AbstractMatrix, U, V) = view(C, U, V)
get_cost_submatrix(neighbors::Union{Vector{BitSet},Vector{Vector{Int}}}, U, V) = Int8[truncated_cost(u, v, neighbors) for u in U, v in V]
function wasserstein(u, v, C, dispersions)
    U, X = findnz(dispersions[u])
    V, Y = findnz(dispersions[v])
    C = get_cost_submatrix(C, U, V)
    OptimalTransport.sinkhorn2(X, Y, C, 1e-1; maxiter=500, atol=1e-2)
end

@inline mm(i, j) = i < j ? CartesianIndex(i, j) : CartesianIndex(j, i)

abstract type AggregateMean end
abstract type AggregateMax end

function aggregate(::Type{AggregateMean}, S::Vector, W)
    s, n = 0.0, length(S)
    (n <= 1) && return 0.0
    @inbounds for i = 1:n, j = (i+1):n
        s += W[mm(S[i], S[j])]
    end
    s * 2 / (n * (n - 1))
end

function aggregate(::Type{AggregateMax}, S::Vector, W)
    s, n = 0.0, length(S)
    (n <= 1) && return 0.0
    @inbounds for i = 1:n, j = (i+1):n
        s = max(s, W[mm(S[i], S[j])])
    end
    s
end

function node_curvature_neighborhood(i::Int, W, neighbors)
    N = neighbors[i]
    if length(N) <= 1
        1.0
    else
        sum(N) do j
            j == i ? 0.0 : 1.0 - W[mm(i, j)]
        end / (length(N) - 1)
    end
end

function node_curvature_edges(node, dist, rc)
    degree = length(rc[node])
    if degree == 0
        1.0
    else
        sum(edge -> dist[edge], rc[node]) / degree
    end
end

function prepare_weights(rc, cr, neighbors)
    ThreadsX.map(eachindex(neighbors)) do node
        map(x -> sum(e -> x in e, view(cr, rc[node])), neighbors[node])
    end
end

function neighborhoods(rc, cr)
    ThreadsX.map(eachindex(rc)) do i
        [x for c in rc[i] for x in cr[c]] |> unique
    end
end

function hypergraph_curvatures(dispersion::Type, aggregations, rc, cr, alpha, cost)
    @info "Preparing Neighborhoods"
    neighbors = neighborhoods(rc, cr)

    C = prepare_cost_matrix(cost, neighbors)

    @info "Preparing Dispersion"
    rw = dispersion == DisperseWeightedClique ? prepare_weights(rc, cr, neighbors) : nothing

    @info "Computing Dispersions"
    D = ThreadsX.map(n -> disperse(dispersion, n, alpha, neighbors, rc, cr, rw), eachindex(rc))

    @info "Computing Directional Curvature"
    w = zeros(Float32, length(rc), length(rc))
    ThreadsX.foreach(eachindex(rc)) do i 
        for j in (i+1):length(rc)
            w[mm(j, i)] = wasserstein(i, j, C, D) 
        end 
    end

    @info "Computing Node Curvature Neighborhood"
    nc = ThreadsX.map(n -> node_curvature_neighborhood(n, w, neighbors), eachindex(rc))

    ac = map(aggregations) do aggregation
        @info "Computing Edge Curvature"
        ec = ThreadsX.map(e -> 1 - aggregate(aggregation, cr[e], w), eachindex(cr))

        @info "Computing Node Curvature Edges"
        nce = ThreadsX.map(n -> node_curvature_edges(n, ec, rc), eachindex(rc))

        (aggregation=Symbol(aggregation), edge_curvature=ec, node_curvature_edges=nce)
    end

    (dispersions=D, directional_curvature=1 .- w, node_curvature_neighborhood=nc, aggregations=ac)
end

function edgelist_format(I::Vector{Int}, J::Vector{Int}, n::Int)
    x = ThreadsX.collect(Int[] for _ in 1:n)
    Threads.@threads for i in ThreadsX.unique(I)
        x[i] = J[I.==i]
    end
    x
end

function transpose_edgelist(cr::Vector{T}) where {T}
    rc = [T() for _ in 1:maximum(e -> maximum(e; init=0), cr)]
    for (j, e) in enumerate(cr), i in e
        push!(rc[i], j)
    end
    filter!(!(isempty), rc)
end

"""
    hypergraph_curvatures

# Arguments
- `dispersion`: Dispersion (options: DisperseUnweightedClique, DisperseWeightedClique, or DisperseUnweightedStar)
- `aggregation`: Aggregation (options: AggregateMean, AggregateMax, or (AggregateMean, AggregateMax))
- `incidence`: Incidence matrix or edge lists encoding of the input hypergraph
- `alpha`: Self-dispersion weight
- `cost`: Cost computation method (options: CostOndemand, CostMatrix)
"""
function hypergraph_curvatures(dispersion::Type, aggregation::A, incidence::AbstractSparseMatrix, alpha::Float64, cost::Type) where {A}
    @info "Preparing Input"
    n, m = size(incidence)
    I, J, _ = findnz(incidence)
    rc, cr = edgelist_format(J, I, m), edgelist_format(I, J, n)
    aggregation = hasmethod(length, Tuple{A}) ? aggregation : [aggregation]
    hypergraph_curvatures(dispersion, aggregation, rc, cr, alpha, cost)
end

function hypergraph_curvatures(dispersion::Type, aggregation::A, incidence::Vector{B}, alpha::Float64, cost::Type) where {A,B}
    @info "Preparing Input"
    rc, cr = transpose_edgelist(incidence), incidence
    aggregation = hasmethod(length, Tuple{A}) ? aggregation : [aggregation]
    hypergraph_curvatures(dispersion, aggregation, rc, cr, alpha, cost)
end

export hypergraph_curvatures

end # module Orchid

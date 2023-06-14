#!/usr/bin/env -S julia -O3 --threads=auto --check-bounds=no

using Orchid
using SparseArrays
using LinearAlgebra
using Glob
using JSON
using ArgParse
using CodecZlib: GzipCompressor, GzipDecompressorStream

parse_edgelist(fp) = [parse.(Int, split(r)) for r in readlines(fp) for s in split(r) if s != ""]
function parse_edgelist_collection(fp)
    rc, y = Vector{Int}[], Int[]
    for r in readlines(fp)
        t = parse.(Int, split(r))
        push!(y, t[1])
        push!(rc, t[2:end])
    end
    y, rc
end

convert(m::AbstractSparseVector) = findnz(m)
convert(m::AbstractMatrix) = findnz(sparse(triu(m, 1))) 
convert(m::Vector{<:Number}) = m
convert(m::Vector) = map(convert, m)
convert(s::Symbol) = String(s)
convert(s) = s

function get_entry(r, a, input, dispersion, alpha)
    (node_curvature_neighborhood = convert(r.node_curvature_neighborhood),
     directional_curvature       = convert(r.directional_curvature),
     node_curvature_edges        = convert(a.node_curvature_edges),
     edge_curvature              = convert(a.edge_curvature),
     aggregation                 = convert(a.aggregation),
     dispersion                  = convert(dispersion),
     input                       = convert(input),
     alpha                       = convert(alpha)) 
end

function run(input, dispersion, aggregation, alpha)
    !(0 <= alpha <= 1) && throw("!(0 <= alpha <= 1)")

    D = Dict{String,Type}(
        lowercase("UnweightedClique") => Orchid.DisperseUnweightedClique,
        lowercase("WeightedClique")   => Orchid.DisperseWeightedClique,
        lowercase("UnweightedStar")   => Orchid.DisperseUnweightedStar
    )[lowercase(dispersion)]
    A = Dict{String,Any}(
        "mean" => Orchid.AggregateMean,
        "max"  => Orchid.AggregateMax,
        "all"  => (Orchid.AggregateMean, Orchid.AggregateMax)
    )[lowercase(aggregation)]

    guess_cost_calc(E) = length(E) > 10_000 || maximum(e -> maximum(e; init=0), E) > 10_000 ? Orchid.CostOndemand : Orchid.CostMatrix
    open_() = endswith(input, ".gz") ? GzipDecompressorStream(open(input)) : open(input)

    if occursin(".chg.tsv", input)
        @info "Reading Hypergraphs"
        y, rc = parse_edgelist_collection(open_())
        ys = unique(y)
        Tot = length(ys)
        results = []
        foreach(ys) do Y
            @info "Importing Hypergraph $Y/$Tot"
            E = rc[y.==Y]
            r = Orchid.hypergraph_curvatures(D, A, E, alpha, guess_cost_calc(E))
            for a in r.aggregations
                push!(results, get_entry(r, a, input, dispersion, alpha))
            end
        end
        results
    else
        @info "Importing Hypergraph"
        E = parse_edgelist(open_())
        r = Orchid.hypergraph_curvatures(D, A, E, alpha, guess_cost_calc(E))
        map(r.aggregations) do a
            get_entry(r, a, input, dispersion, alpha)
        end
    end
end

function orchid_main(input::String, output::String="-"; dispersion::String="UnweightedClique", aggregation::String="All", alpha=0.1)
    if !occursin("*", input)
        results = run(input, dispersion, aggregation, alpha)
        @info "Converting Curvatures to JSON"
        j = JSON.json(results)
        @info "Writing JSON to $output"
        write(output == "-" ? stdout : open(output, "w"), endswith(output, "gz") ? transcode(GzipCompressor, j) : j * "\n")
    else
        @info "Globbing $input"
        results = [a for input in glob(input) for a in run(input, dispersion, aggregation, alpha)]
        @info "Converting Curvatures to JSON"
        j = JSON.json(results)
        @info "Writing JSON to $output"
        write(output == "-" ? stdout : open(output, "w"), endswith(output, "gz") ? transcode(GzipCompressor, j) : j * "\n")
    end
end

function main()
    s = ArgParseSettings(description="""
        This is a command line interface for the ORCHID hypergraph curvature framework described in 
        
        Coupette, C., Dalleiger, S. and Rieck, B., 
        Ollivier-Ricci Curvature for Hypergraphs: A Unified Framework, 
        ICLR 2023, doi:10.48550/arXiv.2210.12048.
    """)
    @add_arg_table! s begin
        "-i", "--input"
            required = true
            help = "Input hypergraph(s) in edgelist format (options: individual edgelist for one hypergraph, collection of edgelists [ext: chg.tsv[.gz]], or a globbing pattern ['*' in `input`] both for multiple hypergraphs)"
        "-o", "--output"
            required = false
            default = "-"
            help = "Output destination ['-' denotes stdout]"
        "--dispersion"
            default = "UnweightedClique"
            help = "Dispersion (options: UnweightedClique, WeightedClique, or UnweightedStar)"
        "--aggregation"
            default = "Mean"
            help = "Aggregation (options: Mean, Max, or All)"
        "--alpha"
            arg_type = Float64
            help = "Self-Dispersion Weight"
            default = 0.0
    end
    opts = parse_args(s)

    orchid_main(opts["input"], opts["output"]; dispersion=opts["dispersion"], aggregation=opts["aggregation"], alpha=opts["alpha"])
end

main()


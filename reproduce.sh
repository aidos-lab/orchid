#!/usr/bin/env bash

echo "Setting exit on error"
set -e

ihgs=("dblp" "ndc-ai" "ndc-pc")
chgs=("mus" "dblp-v" "sha" "stex")
syn=("syn_hcm" "syn_hcm-hsbm" "syn_hnmp" "syn_hsbm")
dispersions=("UnweightedClique" "UnweightedStar" "WeightedClique")
alphas=(0.0 0.1 0.2 0.3 0.4 0.5)

echo "Computing curvatures for individual hypergraphs..."
for dataset in "${ihgs[@]}"; do
  for dispersion in "${dispersions[@]}"; do
    for alpha in "${alphas[@]}"; do
      #echo "results/$dataset.alpha-$alpha.dispersion-$dispersion.orc.json"
      bin/orchid.jl --aggregation All --dispersion $dispersion --alpha $alpha -i data/$dataset.ihg.tsv.gz -o results/$dataset.alpha-$alpha.dispersion-$dispersion.orc.json.gz
      done
    done
  done

echo "Computing curvatures for real-world collections..."
for dataset in "${chgs[@]}"; do
  for dispersion in "${dispersions[@]}"; do
    for alpha in "${alphas[@]}"; do
      #echo "results/$dataset.alpha-$alpha.dispersion-$dispersion.orc.json"
      bin/orchid.jl --aggregation All --dispersion $dispersion --alpha $alpha -i data/$dataset.chg.tsv.gz -o results/$dataset.alpha-$alpha.dispersion-$dispersion.orc.json.gz
      done
    done
  done

echo "Computing curvatures for synthetic collections..."
for dataset in "${syn[@]}"; do
  for dispersion in "${dispersions[@]}"; do
    for alpha in "${alphas[@]}"; do
      #echo "results/$dataset.alpha-$alpha.dispersion-$dispersion.orc.json"
      bin/orchid.jl --aggregation All --dispersion $dispersion --alpha $alpha -i data/$dataset.chg.tsv.gz -o results/$dataset.alpha-$alpha.dispersion-$dispersion.orc.json.gz
      done
    done
  done

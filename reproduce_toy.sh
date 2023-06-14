#!/usr/bin/env bash

echo "Setting exit on error"
set -e

dispersions=("UnweightedClique" "UnweightedStar" "WeightedClique")
alphas=(0.0 0.1 0.2 0.3 0.4 0.5)

echo "Computing curvatures for toy hypergraphs..."
for dispersion in "${dispersions[@]}"; do
  for alpha in "${alphas[@]}"; do
      dataset="toy"
      #echo "results/$dataset.alpha-$alpha.dispersion-$dispersion.orc.json"
      bin/orchid.jl --aggregation All --dispersion $dispersion --alpha $alpha -i data/$dataset.ihg.tsv -o results/$dataset.alpha-$alpha.dispersion-$dispersion.orc.json
      dataset="toys"
      #echo "results/$dataset.alpha-$alpha.dispersion-$dispersion.orc.json"
      bin/orchid.jl --aggregation All --dispersion $dispersion --alpha $alpha -i data/$dataset.chg.tsv -o results/$dataset.alpha-$alpha.dispersion-$dispersion.orc.json
    done
  done
#!/bin/bash

# This scripts reorganizes folder structure to WL/API/HW by creating symlinks.
# Usage: cd dpbench && ./scripts/structure.sh

rm -rf workloads

link() {
  WL=$1
  API=$2

  echo $WL $API

  mkdir -p workloads/$WL/$API
  ln -s ../../../$API/$WL/CPU workloads/$WL/$API/CPU
  ln -s ../../../$API/$WL/GPU workloads/$WL/$API/GPU
}


for API in dpnp native native_dpcpp numba
do
  for WL in blackscholes dbscan gaussian_elim gpairs kmeans knn l2_distance pairwise_distance pathfinder pca
  do
    link $WL $API
  done
done

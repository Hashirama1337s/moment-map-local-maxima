#!/bin/bash
# batch runner for search.py; each run <= budget seconds (+ final classification)
cd <repo>/theory/proof
PY=python
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
run() { $PY search.py $1 $2 $3 $4 $5 > search_$1x$2_$4_$3.log 2>&1; }
run 2 2 101 gue 150 & run 2 3 101 gue 180 & run 2 4 101 gue 180 & run 3 3 101 gue 180 & wait
run 2 3 102 gue 180 & run 2 4 102 gue 180 & run 3 3 102 gue 180 & run 2 2 102 eqsp 150 & wait
run 2 3 103 eqsp 180 & run 2 4 103 eqsp 180 & run 3 3 103 eqsp 180 & run 3 4 101 gue 300 & wait
echo ALLDONE

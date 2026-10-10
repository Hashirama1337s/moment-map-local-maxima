#!/bin/bash
cd <repo>/theory/proof
PY=python
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
run() { $PY search_k.py $1 $2 $3 $4 > searchk_${1//,/x}_$3_$2.log 2>&1; }
run 2,2,2 201 gue 200 & run 2,2,2 202 gue 200 & run 2,2,3 201 gue 240 & run 2,2,3 202 gue 240 & wait
run 2,2,2 203 eqsp 200 & run 2,2,3 203 eqsp 240 & run 2,2,2,2 201 gue 300 & wait
echo ALLDONE

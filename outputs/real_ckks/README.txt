PrivNet-Krylov CKKS performance files

There are two different types of CKKS results in this directory.

1. CALIBRATED ESTIMATES

   The bundled representative files are model-based estimates, not hardware
   measurements. They use published TenSEAL microbenchmarks as a reference and
   rescale them to the parameters used by this project. The calculation can be
   reproduced with:

     python code/calibrated_ckks_estimator.py

   Relevant files:
     MODELED_RESULTS_NOTICE.txt
     representative_ckks_estimates.csv
     representative_ckks_estimates.json
     representative_ckks_table.tex
     scaling_envelope.csv
     scaling_envelope_table.tex

2. DIRECT TENSEAL MEASUREMENTS

   To collect real timings, install the optional dependencies and run:

     pip install -r requirements-ckks.txt
     python code/real_ckks_public_benchmark.py \
       --artifact outputs/public_graph/karate_seed7_ckks_artifact.npz \
       --out outputs/real_ckks \
       --security-estimator-json /path/to/security_estimator.json \
       --repeats 5 --warmup 1

   A real run writes files beginning with `measured_`, including:
     measured_ckks_results.json
     measured_ckks_results.csv
     measured_ckks_table.tex

The representative files must remain labelled as estimates. Measured files should
only be kept when they were produced by an actual TenSEAL run on documented hardware
and include a real security-estimator result for the exact CKKS parameter set.

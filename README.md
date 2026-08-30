# Bridging stochastic compartmental and generative models for epidemic scenario forecasting

Results and code accompanying the manuscript.

The study trains three generative families - variance-preserving diffusion, flow
matching and stochastic interpolants - under one shared one-dimensional U-Net, one
dataset and one compute budget, as emulators of a stage-structured SEIR
continuous-time Markov chain with Erlang residence times and a log-Brownian
transmission path. The same weights are then reused, conditioned on an observed
prefix, to complete partially observed outbreaks.

## Contents

| Path | Description |
|---|---|
| `results/fidelity.csv` | Emulation fidelity per model and per population stratum: energy distance, variogram discrepancy, C2ST overall and per functional, Wasserstein-1 distances. Source of table 3 |
| `results/forecasting_summary.csv` | Skill, coverage and absolute proper scores by phase and horizon on held-out established outbreaks. Source of tables 4 and 5 |
| `results/model_diff_bootstrap.csv` | Paired bootstrap of the energy score between the three routes, with 95% intervals. Source of the comparisons in §3.4 |
| `results/realdata_rho_sweep.csv` | Reporting-rate sweep and scores on the 1978 boarding-school influenza and 1861 Hagelloch measles outbreaks. Source of table 6 |


`fidelity.csv` reports one row per (model, stratum) with `stratum = all` for the
pooled values quoted in the manuscript; `OracleFloor` is the null baseline, obtained
by scoring two independent halves of the simulator pool against each other.

## Reproducing the results

```r
Rscript seir_generative.R      # simulate the pool, train the three routes, evaluate
```


## Data

The two historical outbreaks are public.

## Licence

Code released under the MIT Licence.

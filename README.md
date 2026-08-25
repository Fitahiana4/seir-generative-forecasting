# Bridging stochastic compartmental and generative models for epidemic scenario forecasting

Results, figure code and figures accompanying the manuscript.

The study trains three generative families - variance-preserving diffusion, flow
matching and stochastic interpolants - under one shared one-dimensional U-Net, one
dataset and one compute budget, as emulators of a stage-structured SEIR
continuous-time Markov chain with Erlang residence times and a log-Brownian
transmission path. The same weights are then reused, conditioned on an observed
prefix, to complete partially observed outbreaks.

## Contents

| Path | Description |
|---|---|
| `figures_manuscript.R` | Script producing figures 2, 3 and 4 of the manuscript from the result tables below |
| `figures_manuscript/` | Figures as they appear in the manuscript |
| `results/fidelity.csv` | Emulation fidelity per model and per population stratum: energy distance, variogram discrepancy, C2ST overall and per functional, Wasserstein-1 distances. Source of table 3 |
| `results/forecasting_summary.csv` | Skill, coverage and absolute proper scores by phase and horizon on held-out established outbreaks. Source of tables 4 and 5 |
| `results/model_diff_bootstrap.csv` | Paired bootstrap of the energy score between the three routes, with 95% intervals. Source of the comparisons in §3.4 |
| `results/realdata_rho_sweep.csv` | Reporting-rate sweep and scores on the 1978 boarding-school influenza and 1861 Hagelloch measles outbreaks. Source of table 6 |
| `training_losses.rds` | Training loss traces for the three models |
| `.recipe_tag` | Identifier of the training recipe used for the reported run |

`fidelity.csv` reports one row per (model, stratum) with `stratum = all` for the
pooled values quoted in the manuscript; `OracleFloor` is the null baseline, obtained
by scoring two independent halves of the simulator pool against each other.

## Reproducing the results

```r
Rscript seir_generative.R      # simulate the pool, train the three routes, evaluate
```

The first script writes the tables in `results/` and the training traces; the second
reads them and writes to `figures_manuscript/`. Figure 1 is a schematic of the
architectures and is not produced by either script. Training the three models is the
expensive step and requires a GPU; the tables shipped here are the output of the run
identified in `recipe_tag.txt`, so the figures can be regenerated without retraining.


## Simulator and training code

The simulator, the training code and the trained checkpoints are not held in seir_generative.R

## Data

The two historical outbreaks are public.

## Licence

Code released under the MIT Licence. Result tables and figures released under
CC-BY 4.0.

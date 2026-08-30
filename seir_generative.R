# Bridging stochastic compartmental and generative models for epidemic
# scenario forecasting.


required <- c("dplyr", "tibble", "tidyr", "data.table", "progress", "ggplot2",
              "torch", "outbreaks", "lhs", "surveillance")
missing <- required[!sapply(required, requireNamespace, quietly = TRUE)]
if (length(missing)) install.packages(missing, repos = "https://cloud.r-project.org")

suppressPackageStartupMessages({
  library(dplyr); library(tibble); library(tidyr); library(data.table)
  library(progress); library(ggplot2); library(torch); library(outbreaks)
  library(lhs); library(surveillance)
})
select <- dplyr::select; filter <- dplyr::filter

set.seed(42); torch_manual_seed(42)

DEVICE <- if (cuda_is_available()) torch_device("cuda") else
          if (exists("backends_mps_is_available") && backends_mps_is_available()) torch_device("mps") else
          torch_device("cpu")
cat("device:", DEVICE$type, "\n")

OUT_DIR <- "output"
FIG_DIR <- file.path(OUT_DIR, "figures")
for (d in c(OUT_DIR, FIG_DIR)) dir.create(d, showWarnings = FALSE, recursive = TRUE)


# Simulation grid and pool

T_MAX <- 125; DT <- 1
L <- as.integer(T_MAX / DT) + 1L

N_DECADE_RANGES <- list(small = c(100, 1000), medium = c(1000, 10000),
                        large = c(10000, 100000))
N_CELL_QUOTAS   <- list(small = 1375L, medium = 900L, large = 475L)

# Four transmission regimes crossed with three population decades. R0 ranges are
# deliberately wide; note that uniform sampling in R0 is not uniform in growth
# rate, which is what verify_shift() at the end of this file measures.
STRATA <- list(
  short = list(latent_days_min = 1, latent_days_max = 3, m_min = 2, m_max = 6,
               infect_days_min = 1, infect_days_max = 3, n_min = 2, n_max = 6,
               R0_min = 4, R0_max = 12, I0_min = 1, I0_max = 3),
  medium = list(latent_days_min = 2, latent_days_max = 4, m_min = 2, m_max = 8,
                infect_days_min = 2, infect_days_max = 5, n_min = 2, n_max = 8,
                R0_min = 2.5, R0_max = 6, I0_min = 1, I0_max = 3),
  long = list(latent_days_min = 3, latent_days_max = 6, m_min = 3, m_max = 12,
              infect_days_min = 3, infect_days_max = 7, n_min = 3, n_max = 12,
              R0_min = 1.5, R0_max = 4, I0_min = 1, I0_max = 3),
  extinction = list(latent_days_min = 1, latent_days_max = 5, m_min = 2, m_max = 10,
                    infect_days_min = 1, infect_days_max = 6, n_min = 2, n_max = 10,
                    R0_min = 0.8, R0_max = 1.8, I0_min = 1, I0_max = 3)
)

# Log-Brownian transmission path: extra-demographic variability on top of the
# demographic noise the chain already produces. sigma and gamma stay constant,
# since a varying gamma would break the Erlang identities.
TV_BETA_SD_MIN <- 0.05
TV_BETA_SD_MAX <- 0.12

RHO_RANGE <- c(0.05, 1.00)
N_HELDOUT <- 1000L


# Model and training

BETA_MIN <- 0.1; BETA_MAX <- 10.0
N_DIFF <- 1000L; N_FLOW <- 50L; N_SI <- 50L
N_EPOCHS <- 200L; BATCH_SIZE <- 256L; LR <- 5e-4
BASE_CH <- 64L; FOURIER_K <- 8L
EMA_DECAY <- 0.999; MIN_SNR_GAMMA <- 5.0; WEIGHT_DECAY <- 1e-4; WARMUP_EPOCHS <- 5L
SI_SIGMA_MAX <- 0.5


# Evaluation

N_GEN_FIDELITY <- 10000L
EVAL_ENS       <- 60L
EVAL_FC_N      <- 250L
ENERGY_SUB     <- 3000L
GEN_CHUNK      <- 1000L
EXT_PEAK_FRAC  <- 0.01

EVAL_HORIZONS <- c(5L, 10L, 15L)
DECADES       <- c("small", "medium", "large")
CUT_LABELS    <- c("rising", "peak", "declining")
MODEL_NAMES   <- c("Diffusion", "FlowMatching", "StochasticInterpolants")
RHO_SWEEP     <- c(0.2, 0.5, 1.0)
DISEASES      <- c("flu1978", "hagelloch")


# Utilities

.logt   <- function(x) log1p(pmax(x, 0))
.Nstrat <- function(N) ifelse(N < 1000, "small", ifelse(N < 10000, "medium", "large"))

# Mean pairwise Euclidean distance over all nx*ny pairs, in row blocks so peak
# memory is block*ny. No internal subsampling: the caller draws the sample once
# so that every term of an energy statistic uses the same draw.
.mean_pairdist <- function(X, Y, block = 512L) {
  nx <- nrow(X); ny <- nrow(Y)
  if (nx == 0L || ny == 0L) return(NA_real_)
  sx <- rowSums(X^2); sy <- rowSums(Y^2); tot <- 0
  for (st in seq(1L, nx, by = block)) {
    idx <- st:min(st + block - 1L, nx)
    d2  <- outer(sx[idx], sy, "+") - 2 * tcrossprod(X[idx, , drop = FALSE], Y)
    tot <- tot + sum(sqrt(pmax(d2, 0)))
  }
  tot / (nx * ny)
}

# Pure binomial thinning, one constant reporting probability per trajectory.
.apply_observation <- function(counts, rho) {
  stopifnot(length(rho) == nrow(counts))
  Iobs <- matrix(0L, nrow(counts), ncol(counts))
  for (i in seq_len(nrow(counts)))
    Iobs[i, ] <- rbinom(ncol(counts), pmax(round(counts[i, ]), 0L), rho[i])
  Iobs
}
.draw_rho <- function(n) runif(n, RHO_RANGE[1], RHO_RANGE[2])

# Two conservation bounds on an unknown reporting rate. Prevalence cannot exceed
# the population, and neither can the cumulative number infected; the second is
# usually the binding one.
.rho_feasible <- function(I_obs, N_val, gamma, grid = RHO_SWEEP) {
  rho_prev  <- max(I_obs) / N_val
  rho_fs    <- gamma * sum(I_obs) * DT / N_val
  rho_lower <- max(rho_prev, rho_fs)
  keep <- grid[grid >= rho_lower]
  if (length(keep) < length(grid) && 0.75 >= rho_lower) keep <- sort(unique(c(keep, 0.75)))
  if (!length(keep)) keep <- 1.0
  cat(sprintf("    rho_min = %.3f (peak %.3f, final size %.3f) -> sweep: %s\n",
              rho_lower, rho_prev, rho_fs, paste(keep, collapse = ", ")))
  keep
}

.rho_batch      <- function(rho_tensor, idx) rho_tensor[idx]$unsqueeze(2L)
.rho_vec_tensor <- function(rho_values, n) {
  v <- if (length(rho_values) == 1L) rep(rho_values, n) else rho_values
  torch_tensor(v, dtype = torch_float32(), device = DEVICE)$unsqueeze(2L)
}


# SEIR simulator

qint  <- function(u, lo, hi) if (lo == hi) rep(as.integer(lo), length(u)) else pmin(hi, lo + floor(u * (hi - lo + 1)))
qcont <- function(u, lo, hi) if (lo == hi) rep(lo, length(u)) else qunif(u, min = lo, max = hi)

build_lhs_one_regime <- function(regime_name, n) {
  Rg <- STRATA[[regime_name]]
  d  <- lhs::randomLHS(n, 6L)
  tibble(disease = "Generic", stratum = regime_name,
         latent_days     = qcont(d[, 1], Rg$latent_days_min, Rg$latent_days_max),
         infectious_days = qcont(d[, 2], Rg$infect_days_min, Rg$infect_days_max),
         m  = qint(d[, 3], Rg$m_min, Rg$m_max),
         n  = qint(d[, 4], Rg$n_min, Rg$n_max),
         R0_basic = qcont(d[, 5], Rg$R0_min, Rg$R0_max),
         I0 = qint(d[, 6], Rg$I0_min, Rg$I0_max))
}

params_from_lhs_row <- function(row) {
  list(disease = row$disease, stratum = row$stratum,
       latent_days = row$latent_days, infectious_days = row$infectious_days,
       sigma = 1 / row$latent_days, gamma = 1 / row$infectious_days,
       m = as.integer(row$m), n = as.integer(row$n),
       R0_basic = row$R0_basic, beta = row$R0_basic / row$infectious_days,
       I0 = as.integer(row$I0))
}

draw_beta_traj <- function(t_max, R0_base, gamma_base, R0_lo, R0_hi) {
  Td   <- as.integer(t_max)
  s_b  <- runif(1L, TV_BETA_SD_MIN, TV_BETA_SD_MAX)
  logb <- log(R0_base * gamma_base) + cumsum(c(0, rnorm(Td - 1L, 0, s_b)))
  list(beta_t = pmin(pmax(exp(logb), R0_lo * gamma_base), R0_hi * gamma_base), s_b = s_b)
}

# Gillespie SSA with Erlang sub-stages. The waiting time assumes a constant rate,
# but beta(t) changes at each day boundary; if a draw would cross a boundary the
# clock advances only to it and the rates are redrawn. Memorylessness makes this
# exact rather than an approximation.
simulate_seir_ctmc <- function(t_max, N, beta_t, sigma_base, gamma_base, m, n,
                               I0_total = 1L, E0_total = 0L, R0_init = 0L) {
  S <- as.integer(N - I0_total - E0_total - R0_init)
  E <- integer(m); if (E0_total > 0L) E[1L] <- as.integer(E0_total)
  I <- integer(n); if (I0_total > 0L) I[1L] <- as.integer(I0_total)
  R <- as.integer(R0_init); t <- 0
  E_total <- sum(E); I_total <- sum(I)
  out <- matrix(NA_real_, nrow = 10000L, ncol = 5L)
  k <- 1L; out[1L, ] <- c(t, S, E_total, I_total, R)
  n_events <- 1L + (m - 1L) + 1L + (n - 1L) + 1L
  rates <- numeric(n_events)
  Td <- length(beta_t)
  m_sigma <- m * sigma_base; n_gamma <- n * gamma_base
  cur_day <- -1L; beta <- 0
  while (t < t_max) {
    if (E_total == 0L && I_total == 0L) break
    d_now <- min(max(as.integer(floor(t)) + 1L, 1L), Td)
    if (d_now != cur_day) { cur_day <- d_now; beta <- beta_t[cur_day] }
    idx <- 1L
    rates[idx] <- beta * S * I_total / N; idx <- idx + 1L
    if (m > 1L) for (a in 1:(m - 1L)) { rates[idx] <- m_sigma * E[a]; idx <- idx + 1L }
    rates[idx] <- m_sigma * E[m]; idx <- idx + 1L
    if (n > 1L) for (b in 1:(n - 1L)) { rates[idx] <- n_gamma * I[b]; idx <- idx + 1L }
    rates[idx] <- n_gamma * I[n]
    lambda <- sum(rates); if (lambda <= 0) break
    next_boundary <- floor(t) + 1
    tau <- rexp(1L, lambda)
    if (t + tau > next_boundary && next_boundary < t_max) { t <- next_boundary; next }
    t <- t + tau; if (t > t_max) break
    event <- sample.int(n_events, 1L, prob = rates)
    ev <- 1L
    if (event == ev) {
      S <- S - 1L; E[1L] <- E[1L] + 1L; E_total <- E_total + 1L
    } else {
      ev <- ev + 1L; handled <- FALSE
      if (m > 1L) for (a in 1:(m - 1L)) {
        if (event == ev) { E[a] <- E[a] - 1L; E[a + 1L] <- E[a + 1L] + 1L; handled <- TRUE; break }
        ev <- ev + 1L
      }
      if (!handled) {
        if (event == ev) {
          E[m] <- E[m] - 1L; I[1L] <- I[1L] + 1L; E_total <- E_total - 1L; I_total <- I_total + 1L
        } else {
          ev <- ev + 1L
          if (n > 1L) for (b in 1:(n - 1L)) {
            if (event == ev) { I[b] <- I[b] - 1L; I[b + 1L] <- I[b + 1L] + 1L; handled <- TRUE; break }
            ev <- ev + 1L
          }
          if (!handled) { I[n] <- I[n] - 1L; R <- R + 1L; I_total <- I_total - 1L }
        }
      }
    }
    k <- k + 1L
    if (k > nrow(out)) out <- rbind(out, matrix(NA_real_, nrow = nrow(out), ncol = 5L))
    out[k, ] <- c(t, S, E_total, I_total, R)
  }
  out <- out[1:k, , drop = FALSE]
  data.table(time = out[, 1], S = as.integer(out[, 2]), E_total = as.integer(out[, 3]),
             I_total = as.integer(out[, 4]), R = as.integer(out[, 5]))
}

to_regular_grid <- function(sim_dt, t_max, dt = 1) {
  gt <- seq(0, t_max, by = dt)
  idx <- findInterval(gt, sim_dt$time); idx[idx < 1L] <- 1L
  data.table(time = gt, I_total = sim_dt$I_total[idx])
}

run_pool <- function(t_max, dt, seed = 2025) {
  set.seed(seed)
  grid_list <- list(); par_list <- list(); N_per_sim <- integer(0); global_id <- 0L
  for (rg in names(STRATA)) {
    Rg <- STRATA[[rg]]
    for (dec in names(N_CELL_QUOTAS)) {
      quota <- N_CELL_QUOTAS[[dec]]
      cat(sprintf("  %-10s x %-6s (%d trajectories)\n", rg, dec, quota))
      design <- build_lhs_one_regime(rg, quota)
      rng    <- N_DECADE_RANGES[[dec]]
      N_cell <- as.integer(round(exp(runif(quota, log(rng[1]), log(rng[2])))))
      g_cell <- vector("list", quota); p_cell <- vector("list", quota)
      pb <- progress_bar$new(format = paste0("    [", rg, "/", dec, "] [:bar] :percent ETA: :eta"),
                             total = quota, clear = FALSE, width = 60)
      for (j in seq_len(quota)) {
        pb$tick()
        pars  <- params_from_lhs_row(design[j, ])
        prof  <- draw_beta_traj(t_max, pars$R0_basic, pars$gamma, Rg$R0_min, Rg$R0_max)
        sim   <- simulate_seir_ctmc(t_max, N_cell[j], prof$beta_t, pars$sigma,
                                    pars$gamma, pars$m, pars$n, I0_total = pars$I0)
        pars$beta_mean <- mean(prof$beta_t); pars$s_beta <- prof$s_b
        g_cell[[j]] <- to_regular_grid(sim, t_max, dt)
        p_cell[[j]] <- c(pars, N = N_cell[j], decade = dec)
      }
      ids <- (global_id + 1L):(global_id + quota)
      gcell <- data.table::rbindlist(g_cell, idcol = ".j")
      gcell[, sim_id := ids[.j]][, .j := NULL]
      grid_list[[length(grid_list) + 1L]] <- gcell
      pcell <- data.table::rbindlist(p_cell, fill = TRUE)
      pcell[, sim_id := ids]
      par_list[[length(par_list) + 1L]] <- pcell
      N_per_sim[ids] <- N_cell
      global_id <- global_id + quota
    }
  }
  list(grid = data.table::rbindlist(grid_list),
       params = data.table::rbindlist(par_list, fill = TRUE),
       N_per_sim = N_per_sim)
}

cat("\nsimulating the pool\n")
res <- run_pool(T_MAX, DT)
print(table(factor(res$params$stratum, levels = names(STRATA)),
            factor(.Nstrat(res$N_per_sim), levels = DECADES)))

set.seed(2024)
all_sim_ids <- sort(unique(res$params$sim_id))
.dec_of <- cut(res$N_per_sim[all_sim_ids], breaks = c(0, 1000, 10000, Inf),
               labels = DECADES, right = FALSE)
.alloc <- round(N_HELDOUT * as.numeric(table(.dec_of)) / length(.dec_of))
.alloc[1] <- N_HELDOUT - sum(.alloc[-1]); names(.alloc) <- levels(.dec_of)
setaside_ids <- integer(0)
for (d in names(.alloc)) {
  ids_d <- all_sim_ids[.dec_of == d]
  setaside_ids <- c(setaside_ids, sample(ids_d, min(.alloc[[d]], length(ids_d))))
}
setaside_ids  <- sort(setaside_ids)
train_sim_ids <- setdiff(all_sim_ids, setaside_ids)
cat(sprintf("held-out %d | training %d\n", length(setaside_ids), length(train_sim_ids)))

prepare_I_counts <- function(grid_dt) {
  sims <- sort(unique(grid_dt$sim_id))
  stopifnot(nrow(grid_dt) == length(sims) * L, !is.unsorted(grid_dt$sim_id))
  matrix(grid_dt$I_total, nrow = length(sims), ncol = L, byrow = TRUE)
}

I_counts_all   <- prepare_I_counts(res$grid)
I_train        <- I_counts_all[which(all_sim_ids %in% train_sim_ids), , drop = FALSE]
I_held         <- I_counts_all[which(all_sim_ids %in% setaside_ids),  , drop = FALSE]
N_train_values <- res$N_per_sim[train_sim_ids]
N_held_values  <- res$N_per_sim[setaside_ids]
P              <- as.data.frame(res$params)
gamma_train    <- P$gamma[match(train_sim_ids, P$sim_id)]
rm(I_counts_all); invisible(gc())

# The network sees observed prevalence as a proportion of N, with rho supplied as
# a label rather than inferred.
set.seed(909)
.rho_tr <- .draw_rho(nrow(I_train))
X_obs_train     <- torch_tensor(.apply_observation(I_train, .rho_tr) / N_train_values,
                                dtype = torch_float32(), device = DEVICE)
log_N_obs_train <- torch_tensor(log(N_train_values), dtype = torch_float32(), device = DEVICE)
rho_obs_train   <- torch_tensor(.rho_tr, dtype = torch_float32(), device = DEVICE)


# Network

beta_schedule    <- function(tau) BETA_MIN + tau * (BETA_MAX - BETA_MIN)
alpha_bar_scalar <- function(tau) exp(-(tau * BETA_MIN + tau^2 * (BETA_MAX - BETA_MIN) / 2))
alpha_bar_tensor <- function(tt) torch_exp(-(tt * BETA_MIN + tt$pow(2) * (BETA_MAX - BETA_MIN) / 2))

sinusoidal_embedding <- function(t, dim) {
  half  <- as.integer(dim %/% 2)
  freqs <- torch_exp(-log(10000) * torch_tensor(seq(0, half - 1) / half,
                     dtype = torch_float32(), device = t$device))
  emb <- torch_cat(list(torch_sin(t * freqs$unsqueeze(1)),
                        torch_cos(t * freqs$unsqueeze(1))), dim = 2L)
  if (emb$size(2) < dim)
    emb <- torch_cat(list(emb, torch_zeros(c(emb$size(1), dim - emb$size(2)),
                     dtype = torch_float32(), device = t$device)), dim = 2L)
  emb
}

unet_block <- nn_module("UNetBlock",
  initialize = function(in_ch, out_ch, time_emb_dim, kernel_size = 5L) {
    self$conv1 <- nn_conv1d(in_ch, out_ch, kernel_size, padding = kernel_size %/% 2L)
    self$conv2 <- nn_conv1d(out_ch, out_ch, kernel_size, padding = kernel_size %/% 2L)
    self$time_mlp <- nn_linear(time_emb_dim, out_ch)
    self$norm1 <- nn_group_norm(min(8L, out_ch), out_ch)
    self$norm2 <- nn_group_norm(min(8L, out_ch), out_ch)
    self$act <- nn_silu()
    self$skip <- if (in_ch != out_ch) nn_conv1d(in_ch, out_ch, 1L) else nn_identity()
  },
  forward = function(x, t_emb) {
    h <- self$act(self$norm1(self$conv1(x)))
    h <- h + self$time_mlp(t_emb)$unsqueeze(3L)
    h <- self$act(self$norm2(self$conv2(h)))
    h + self$skip(x)
  })

attention_block_1d <- nn_module("AttentionBlock1D",
  initialize = function(channels, n_heads = 4L) {
    self$channels <- channels; self$n_heads <- n_heads
    self$norm   <- nn_group_norm(min(8L, channels), channels)
    self$to_qkv <- nn_conv1d(channels, channels * 3L, 1L)
    self$to_out <- nn_conv1d(channels, channels, 1L)
  },
  forward = function(x) {
    B <- x$size(1); C <- x$size(2); Lx <- x$size(3)
    qkv <- self$to_qkv(self$norm(x))$chunk(3L, dim = 2L)
    hd <- C %/% self$n_heads
    q <- qkv[[1]]$reshape(c(B, self$n_heads, hd, Lx))
    k <- qkv[[2]]$reshape(c(B, self$n_heads, hd, Lx))
    v <- qkv[[3]]$reshape(c(B, self$n_heads, hd, Lx))
    attn <- nnf_softmax(torch_matmul(q$transpose(3L, 4L), k) / sqrt(hd), dim = -1L)
    out  <- torch_matmul(attn, v$transpose(3L, 4L))$transpose(3L, 4L)$reshape(c(B, C, Lx))
    x + self$to_out(out)
  })

# Dilated convolutions at full resolution: they enlarge the receptive field
# without pooling, so small-N roughness is refined rather than smoothed away.
dilated_block <- nn_module("DilatedBlock",
  initialize = function(ch, dilations = c(1L, 2L, 4L, 8L), kernel_size = 3L) {
    self$convs <- nn_module_list(lapply(dilations, function(d)
      nn_conv1d(ch, ch, kernel_size, padding = d * (kernel_size %/% 2L), dilation = d)))
    self$norms <- nn_module_list(lapply(dilations, function(d) nn_group_norm(min(8L, ch), ch)))
    self$act <- nn_silu()
  },
  forward = function(x) {
    for (i in seq_along(self$convs)) x <- x + self$act(self$norms[[i]](self$convs[[i]](x)))
    x
  })

# One network shared by the three routes. Input is the current path plus a binary
# mask, with 2K sinusoidal positional channels appended; tau, log N and rho are
# embedded and summed into a single vector added inside every block.
score_unet_1d <- nn_module("ScoreUNet1D",
  initialize = function(seq_len, base_ch = 64L, time_emb_dim = 128L,
                        in_channels = 2L, out_channels = 1L) {
    self$seq_len <- seq_len; self$time_emb_dim <- time_emb_dim
    self$out_channels <- out_channels
    mlp <- function() nn_sequential(nn_linear(time_emb_dim, time_emb_dim * 2L), nn_silu(),
                                    nn_linear(time_emb_dim * 2L, time_emb_dim))
    self$time_mlp <- mlp(); self$N_mlp <- mlp(); self$rho_mlp <- mlp()
    self$ff_freqs <- 2^(0:(FOURIER_K - 1L)); self$n_ff <- 2L * FOURIER_K
    self$in_conv <- nn_conv1d(in_channels + self$n_ff, base_ch, 7L, padding = 3L)
    self$down1 <- unet_block(base_ch, base_ch, time_emb_dim)
    self$pool1 <- nn_conv1d(base_ch, base_ch * 2L, 3L, stride = 2L, padding = 1L)
    self$down2 <- unet_block(base_ch * 2L, base_ch * 2L, time_emb_dim)
    self$pool2 <- nn_conv1d(base_ch * 2L, base_ch * 4L, 3L, stride = 2L, padding = 1L)
    self$down3 <- unet_block(base_ch * 4L, base_ch * 4L, time_emb_dim)
    self$bottleneck <- unet_block(base_ch * 4L, base_ch * 4L, time_emb_dim)
    self$bottleneck_attn <- attention_block_1d(base_ch * 4L, 4L)
    self$up2_conv <- nn_conv_transpose1d(base_ch * 4L, base_ch * 2L, 4L, stride = 2L, padding = 1L)
    self$up2 <- unet_block(base_ch * 4L, base_ch * 2L, time_emb_dim)
    self$up1_conv <- nn_conv_transpose1d(base_ch * 2L, base_ch, 4L, stride = 2L, padding = 1L)
    self$up1 <- unet_block(base_ch * 2L, base_ch, time_emb_dim)
    self$refine <- dilated_block(base_ch, c(1L, 2L, 4L, 8L))
    self$out_conv <- nn_conv1d(base_ch, out_channels, 3L, padding = 1L)
  },
  forward = function(x_with_mask, tau, log_N, rho) {
    stopifnot(!is.null(rho))
    emb <- self$time_mlp(sinusoidal_embedding(tau, self$time_emb_dim)) +
           self$N_mlp(sinusoidal_embedding(log_N, self$time_emb_dim)) +
           self$rho_mlp(sinusoidal_embedding(rho, self$time_emb_dim))
    Lx <- x_with_mask$size(3); Bx <- x_with_mask$size(1)
    p  <- (torch_arange(start = 0, end = Lx - 1L, device = x_with_mask$device) / Lx)$unsqueeze(1L)
    ff_list <- list()
    for (f in self$ff_freqs) {
      ff_list[[length(ff_list) + 1L]] <- torch_sin(2 * pi * f * p)
      ff_list[[length(ff_list) + 1L]] <- torch_cos(2 * pi * f * p)
    }
    ff <- torch_stack(ff_list, dim = 2L)$expand(c(Bx, self$n_ff, Lx))
    h1 <- self$down1(self$in_conv(torch_cat(list(x_with_mask, ff), dim = 2L)), emb)
    h2 <- self$down2(self$pool1(h1), emb)
    h3 <- self$down3(self$pool2(h2), emb)
    hb <- self$bottleneck_attn(self$bottleneck(h3, emb))
    u2 <- self$up2_conv(hb); if (u2$size(3) != h2$size(3)) u2 <- u2[, , 1:h2$size(3)]
    u2o <- self$up2(torch_cat(list(u2, h2), dim = 2L), emb)
    u1 <- self$up1_conv(u2o); if (u1$size(3) != h1$size(3)) u1 <- u1[, , 1:h1$size(3)]
    u1o <- self$up1(torch_cat(list(u1, h1), dim = 2L), emb)
    out <- self$out_conv(self$refine(u1o))
    if (self$out_channels == 1L) out$squeeze(2L) else out
  })

T_OBS_MIN_TRAIN <- 3L
T_OBS_MAX_TRAIN <- as.integer(L - 5L)

# The revealed prefix is biased towards the early phase: a uniform cut point makes
# most examples trivial completions of an outbreak that has already ended. With
# probability 0.3 nothing is revealed, so the same weights learn both the marginal
# law used for emulation and the conditional law used for forecasting.
sample_random_masks <- function(batch_size, p_uncond = 0.3) {
  is_uncond <- runif(batch_size) < p_uncond
  frac <- rbeta(batch_size, 1.4, 4.0)
  t_obs_cond <- pmax(T_OBS_MIN_TRAIN,
                     pmin(T_OBS_MAX_TRAIN,
                          as.integer(round(T_OBS_MIN_TRAIN + frac * (T_OBS_MAX_TRAIN - T_OBS_MIN_TRAIN)))))
  t_obs <- ifelse(is_uncond, 0L, t_obs_cond)
  col_idx <- matrix(seq_len(L), nrow = batch_size, ncol = L, byrow = TRUE)
  torch_tensor((col_idx <= t_obs) * 1.0, dtype = torch_float32(), device = DEVICE)
}
stack_signal_mask <- function(x, mask) torch_stack(list(x, mask), dim = 2L)

# Normalised per example, not per batch, so a trajectory cut early does not
# dominate the gradient through its larger number of unobserved points.
.masked_loss_vec <- function(diff, mask) {
  keep   <- 1 - mask
  n_pred <- torch_clamp(keep$sum(dim = 2L), min = 1)
  (diff * diff * keep)$sum(dim = 2L) / n_pred
}
.masked_loss <- function(diff, mask) .masked_loss_vec(diff, mask)$mean()

.set_lr <- function(optimizer, epoch, base_lr) {
  f <- if (epoch <= WARMUP_EPOCHS) epoch / max(1L, WARMUP_EPOCHS)
       else 0.5 * (1 + cos(pi * (epoch - WARMUP_EPOCHS) / max(1L, N_EPOCHS - WARMUP_EPOCHS)))
  optimizer$param_groups[[1]]$lr <- base_lr * f
}

.ema_new <- function(model) lapply(model$state_dict(), function(t) t$detach()$clone())
.ema_update <- function(ema, model, gstep) {
  d  <- min(EMA_DECAY, (1 + gstep) / (10 + gstep))
  sd <- model$state_dict()
  with_no_grad({
    for (nm in names(ema))
      if (ema[[nm]]$dtype == torch_float32())
        ema[[nm]]$mul_(d)$add_(sd[[nm]]$detach(), alpha = 1 - d)
  })
  invisible(NULL)
}
.load_ema_into <- function(model, ema) {
  sd <- model$state_dict()
  with_no_grad({ for (nm in names(ema)) if (nm %in% names(sd)) sd[[nm]]$copy_(ema[[nm]]) })
  model
}

.logitnormal_t <- function(b) pmin(pmax(plogis(rnorm(b)), 1e-4), 1 - 1e-4)

.new_fit <- function(out_channels, lr) {
  model <- score_unet_1d(L, BASE_CH, 128L, in_channels = 2L, out_channels = out_channels)
  model$to(device = DEVICE)
  list(model = model,
       optimizer = optim_adamw(model$parameters, lr = lr, weight_decay = WEIGHT_DECAY),
       ema = .ema_new(model))
}


# Training

train_diffusion <- function(X, log_N, rho_tensor) {
  n_sims <- dim(X)[1]
  ini <- .new_fit(1L, LR)
  model <- ini$model; optimizer <- ini$optimizer; ema <- ini$ema; gstep <- 0L
  losses <- numeric(N_EPOCHS)
  pb <- progress_bar$new(format = "  [diffusion] [:bar] :percent | loss: :loss | :elapsed",
                         total = N_EPOCHS, clear = FALSE, width = 70)
  for (epoch in seq_len(N_EPOCHS)) {
    .set_lr(optimizer, epoch, LR)
    model$train(); perm <- sample.int(n_sims); epoch_loss <- 0; nb <- 0L
    for (start in seq(1, n_sims, by = BATCH_SIZE)) {
      bi <- perm[start:min(start + BATCH_SIZE - 1L, n_sims)]; b <- length(bi)
      x0 <- X[bi, , drop = FALSE]
      log_N_b <- log_N[bi]$unsqueeze(2L); rho_b <- .rho_batch(rho_tensor, bi)
      tau <- torch_tensor(runif(b, 1e-4, 1.0), dtype = torch_float32(), device = DEVICE)$unsqueeze(2L)
      ab <- alpha_bar_tensor(tau)
      noise <- torch_randn_like(x0)
      x_noisy <- torch_sqrt(ab) * x0 + torch_sqrt(1 - ab) * noise
      mask <- sample_random_masks(b)
      net_in <- stack_signal_mask(mask * x0 + (1 - mask) * x_noisy, mask)
      w <- torch_clamp(MIN_SNR_GAMMA * (1 - ab) / ab, max = 1)$view(c(-1L))
      per_ex <- .masked_loss_vec(model(net_in, tau, log_N_b, rho_b) - noise, mask)
      loss <- (w * per_ex)$mean()
      optimizer$zero_grad(); loss$backward()
      nn_utils_clip_grad_norm_(model$parameters, max_norm = 1.0); optimizer$step()
      gstep <- gstep + 1L; .ema_update(ema, model, gstep)
      epoch_loss <- epoch_loss + loss$detach(); nb <- nb + 1L
    }
    losses[epoch] <- as.numeric((epoch_loss / nb)$item())
    pb$tick(tokens = list(loss = sprintf("%.5f", losses[epoch])))
  }
  .load_ema_into(model, ema); model$eval()
  list(model = model, losses = losses, ema = ema, type = "diffusion")
}

train_flow_matching <- function(X, log_N, rho_tensor) {
  n_sims <- dim(X)[1]
  ini <- .new_fit(1L, LR)
  model <- ini$model; optimizer <- ini$optimizer; ema <- ini$ema; gstep <- 0L
  losses <- numeric(N_EPOCHS)
  pb <- progress_bar$new(format = "  [flow matching] [:bar] :percent | loss: :loss | :elapsed",
                         total = N_EPOCHS, clear = FALSE, width = 70)
  for (epoch in seq_len(N_EPOCHS)) {
    .set_lr(optimizer, epoch, LR)
    model$train(); perm <- sample.int(n_sims); epoch_loss <- 0; nb <- 0L
    for (start in seq(1, n_sims, by = BATCH_SIZE)) {
      bi <- perm[start:min(start + BATCH_SIZE - 1L, n_sims)]; b <- length(bi)
      x1 <- X[bi, , drop = FALSE]
      log_N_b <- log_N[bi]$unsqueeze(2L); rho_b <- .rho_batch(rho_tensor, bi)
      x0 <- torch_randn_like(x1)
      tv <- torch_tensor(.logitnormal_t(b), dtype = torch_float32(), device = DEVICE)$unsqueeze(2L)
      x_t <- (1 - tv) * x0 + tv * x1
      mask <- sample_random_masks(b)
      net_in <- stack_signal_mask(mask * x1 + (1 - mask) * x_t, mask)
      loss <- .masked_loss(model(net_in, tv, log_N_b, rho_b) - (x1 - x0), mask)
      optimizer$zero_grad(); loss$backward()
      nn_utils_clip_grad_norm_(model$parameters, max_norm = 1.0); optimizer$step()
      gstep <- gstep + 1L; .ema_update(ema, model, gstep)
      epoch_loss <- epoch_loss + loss$detach(); nb <- nb + 1L
    }
    losses[epoch] <- as.numeric((epoch_loss / nb)$item())
    pb$tick(tokens = list(loss = sprintf("%.5f", losses[epoch])))
  }
  .load_ema_into(model, ema); model$eval()
  list(model = model, losses = losses, ema = ema, type = "flow_matching")
}

SI_ALPHA     <- function(t) cos(pi * t / 2)
SI_BETA      <- function(t) sin(pi * t / 2)
SI_ALPHA_DOT <- function(t) -(pi / 2) * sin(pi * t / 2)
SI_BETA_DOT  <- function(t)  (pi / 2) * cos(pi * t / 2)
SI_GAMMA     <- function(t) SI_SIGMA_MAX * t * (1 - t)
SI_GAMMA_DOT <- function(t) SI_SIGMA_MAX * (1 - 2 * t)

# Lowest loss the noise head can reach, given that z is partly swamped by
# alpha*x0 in x_t. Computed rather than hard-coded, since it moves with
# SI_SIGMA_MAX; a value near 1 would mean the head carries no information.
.si_eta_floor <- local({
  set.seed(1L); tt <- plogis(rnorm(2e5L))
  a <- SI_ALPHA(tt); g <- SI_GAMMA(tt)
  mean(1 - g^2 / (a^2 + g^2))
})

train_stochastic_interpolants <- function(X, log_N, rho_tensor) {
  n_sims <- dim(X)[1]
  ini <- .new_fit(2L, LR)
  model <- ini$model; optimizer <- ini$optimizer; ema <- ini$ema; gstep <- 0L
  losses <- numeric(N_EPOCHS); losses_vd <- numeric(N_EPOCHS); losses_eta <- numeric(N_EPOCHS)
  mk <- function(v) torch_tensor(v, dtype = torch_float32(), device = DEVICE)$unsqueeze(2L)
  pb <- progress_bar$new(format = "  [interpolants] [:bar] :percent | vd: :lvd  eta: :leta | :elapsed",
                         total = N_EPOCHS, clear = FALSE, width = 78)
  for (epoch in seq_len(N_EPOCHS)) {
    .set_lr(optimizer, epoch, LR)
    model$train(); perm <- sample.int(n_sims)
    epoch_loss <- 0; e_vd <- 0; e_eta <- 0; nb <- 0L
    for (start in seq(1, n_sims, by = BATCH_SIZE)) {
      bi <- perm[start:min(start + BATCH_SIZE - 1L, n_sims)]; b <- length(bi)
      x1 <- X[bi, , drop = FALSE]
      log_N_b <- log_N[bi]$unsqueeze(2L); rho_b <- .rho_batch(rho_tensor, bi)
      x0 <- torch_randn_like(x1); z <- torch_randn_like(x1)
      tn <- .logitnormal_t(b); tv <- mk(tn)
      x_t <- mk(SI_ALPHA(tn)) * x0 + mk(SI_BETA(tn)) * x1 + mk(SI_GAMMA(tn)) * z
      target_vd <- mk(SI_ALPHA_DOT(tn)) * x0 + mk(SI_BETA_DOT(tn)) * x1
      mask <- sample_random_masks(b)
      net_in <- stack_signal_mask(mask * x1 + (1 - mask) * x_t, mask)
      out <- model(net_in, tv, log_N_b, rho_b)
      v_d <- out[, 1, ]; eta <- out[, 2, ]
      if (epoch == 1L && start == 1L)
        stopifnot(length(v_d$shape) == 2L, v_d$shape[2] == L, eta$shape[2] == L)
      l_vd  <- .masked_loss(v_d - target_vd, mask)
      l_eta <- .masked_loss(eta - z, mask)
      loss  <- l_vd + l_eta
      optimizer$zero_grad(); loss$backward()
      nn_utils_clip_grad_norm_(model$parameters, max_norm = 1.0); optimizer$step()
      gstep <- gstep + 1L; .ema_update(ema, model, gstep)
      epoch_loss <- epoch_loss + loss$detach()
      e_vd <- e_vd + l_vd$detach(); e_eta <- e_eta + l_eta$detach(); nb <- nb + 1L
    }
    losses[epoch]     <- as.numeric((epoch_loss / nb)$item())
    losses_vd[epoch]  <- as.numeric((e_vd  / nb)$item())
    losses_eta[epoch] <- as.numeric((e_eta / nb)$item())
    pb$tick(tokens = list(lvd = sprintf("%.5f", losses_vd[epoch]),
                          leta = sprintf("%.4f", losses_eta[epoch])))
  }
  cat(sprintf("  noise-head floor %.4f, reached %.4f\n", .si_eta_floor, losses_eta[N_EPOCHS]))
  .load_ema_into(model, ema); model$eval()
  list(model = model, losses = losses, losses_vd = losses_vd, losses_eta = losses_eta,
       ema = ema, type = "stochastic_interpolants")
}

cat("\ntraining\n")
FITS <- list(
  Diffusion              = train_diffusion(X_obs_train, log_N_obs_train, rho_obs_train),
  FlowMatching           = train_flow_matching(X_obs_train, log_N_obs_train, rho_obs_train),
  StochasticInterpolants = train_stochastic_interpolants(X_obs_train, log_N_obs_train, rho_obs_train)
)
saveRDS(lapply(FITS, `[[`, "losses"), file.path(OUT_DIR, "training_losses.rds"))
invisible(gc()); if (cuda_is_available()) cuda_empty_cache()


# Sampling

sample_diffusion <- function(fitted, log_N_values, rho_values) {
  model <- fitted$model; model$eval(); n <- length(log_N_values)
  log_N_t <- torch_tensor(log_N_values, dtype = torch_float32(), device = DEVICE)$unsqueeze(2L)
  rho_t <- .rho_vec_tensor(rho_values, n)
  x <- torch_randn(c(n, L), device = DEVICE)
  mask0 <- torch_zeros(c(n, L), dtype = torch_float32(), device = DEVICE)
  taus <- seq(1, 1e-4, length.out = N_DIFF)
  with_no_grad({
    for (i in seq_along(taus)) {
      tv <- taus[i]
      tt <- torch_tensor(rep(tv, n), dtype = torch_float32(), device = DEVICE)$unsqueeze(2L)
      eps <- model(stack_signal_mask(x, mask0), tt, log_N_t, rho_t)
      ab <- alpha_bar_scalar(tv); bt <- beta_schedule(tv) / N_DIFF
      x <- (1 / sqrt(1 - bt)) * (x - (bt / sqrt(max(1 - ab, 1e-8))) * eps)
      if (i < length(taus)) x <- x + sqrt(bt) * torch_randn_like(x)
    }
  })
  as.matrix(torch_clamp(x, min = 0)$cpu()$detach())
}

sample_flow_matching <- function(fitted, log_N_values, rho_values, n_steps = N_FLOW) {
  model <- fitted$model; model$eval(); n <- length(log_N_values)
  log_N_t <- torch_tensor(log_N_values, dtype = torch_float32(), device = DEVICE)$unsqueeze(2L)
  rho_t <- .rho_vec_tensor(rho_values, n)
  x <- torch_randn(c(n, L), device = DEVICE)
  mask0 <- torch_zeros(c(n, L), dtype = torch_float32(), device = DEVICE)
  dt_fm <- 1 / n_steps
  with_no_grad({
    for (i in seq_len(n_steps)) {
      tt <- torch_tensor(rep((i - 1) * dt_fm, n), dtype = torch_float32(), device = DEVICE)$unsqueeze(2L)
      x <- x + dt_fm * model(stack_signal_mask(x, mask0), tt, log_N_t, rho_t)
    }
  })
  as.matrix(torch_clamp(x, min = 0)$cpu()$detach())
}

sample_stochastic_interpolants <- function(fitted, log_N_values, rho_values, n_steps = N_SI) {
  model <- fitted$model; model$eval(); n <- length(log_N_values)
  log_N_t <- torch_tensor(log_N_values, dtype = torch_float32(), device = DEVICE)$unsqueeze(2L)
  rho_t <- .rho_vec_tensor(rho_values, n)
  x <- torch_randn(c(n, L), device = DEVICE)
  mask0 <- torch_zeros(c(n, L), dtype = torch_float32(), device = DEVICE)
  dt_si <- 1 / n_steps
  with_no_grad({
    for (i in seq_len(n_steps)) {
      t0 <- (i - 1) * dt_si + 1e-3
      tt <- torch_tensor(rep(t0, n), dtype = torch_float32(), device = DEVICE)$unsqueeze(2L)
      out <- model(stack_signal_mask(x, mask0), tt, log_N_t, rho_t)
      x <- x + dt_si * (out[, 1, ] + SI_GAMMA_DOT(t0) * out[, 2, ])
    }
  })
  as.matrix(torch_clamp(x, min = 0)$cpu()$detach())
}

# Conditional sampling: the observed prefix is re-imposed at every step and only
# the remaining points are generated.
.cond_common <- function(obs_vec, T_obs, log_N_val, n_samples, rho_val) {
  mask_vec <- c(rep(1.0, T_obs), rep(0.0, L - T_obs))
  list(obs  = torch_tensor(obs_vec, dtype = torch_float32(), device = DEVICE)$unsqueeze(1L)$`repeat`(c(n_samples, 1L)),
       mask = torch_tensor(mask_vec, dtype = torch_float32(), device = DEVICE)$unsqueeze(1L)$`repeat`(c(n_samples, 1L)),
       logN = torch_tensor(rep(log_N_val, n_samples), dtype = torch_float32(), device = DEVICE)$unsqueeze(2L),
       rho  = .rho_vec_tensor(rho_val, n_samples))
}

conditional_diffusion <- function(fitted, ob, T_obs, n_samples = EVAL_ENS, rho) {
  model <- fitted$model; model$eval(); T_obs <- as.integer(T_obs)
  cc <- .cond_common(ob$I_norm, T_obs, log(ob$N), n_samples, rho)
  x <- cc$mask * cc$obs + (1 - cc$mask) * torch_randn(c(n_samples, L), device = DEVICE)
  taus <- seq(1, 1e-4, length.out = N_DIFF)
  with_no_grad({
    for (i in seq_along(taus)) {
      tv <- taus[i]
      tt <- torch_tensor(rep(tv, n_samples), dtype = torch_float32(), device = DEVICE)$unsqueeze(2L)
      eps <- model(stack_signal_mask(x, cc$mask), tt, cc$logN, cc$rho)
      ab <- alpha_bar_scalar(tv); bt <- beta_schedule(tv) / N_DIFF
      xn <- (1 / sqrt(1 - bt)) * (x - (bt / sqrt(max(1 - ab, 1e-8))) * eps)
      if (i < length(taus)) xn <- xn + sqrt(bt) * torch_randn_like(xn)
      x <- cc$mask * cc$obs + (1 - cc$mask) * xn
    }
  })
  list(I_gen = as.matrix(torch_clamp(x, min = 0)$cpu()$detach()) * ob$N,
       T_obs = T_obs, I_obs_raw = ob$I_raw, label = ob$label, N = ob$N)
}

conditional_flow_matching <- function(fitted, ob, T_obs, n_samples = EVAL_ENS, rho,
                                      n_steps = N_FLOW) {
  model <- fitted$model; model$eval(); T_obs <- as.integer(T_obs)
  cc <- .cond_common(ob$I_norm, T_obs, log(ob$N), n_samples, rho)
  x <- cc$mask * cc$obs + (1 - cc$mask) * torch_randn(c(n_samples, L), device = DEVICE)
  dt_fm <- 1 / n_steps
  with_no_grad({
    for (i in seq_len(n_steps)) {
      tt <- torch_tensor(rep((i - 1) * dt_fm, n_samples), dtype = torch_float32(), device = DEVICE)$unsqueeze(2L)
      v <- model(stack_signal_mask(x, cc$mask), tt, cc$logN, cc$rho)
      x <- cc$mask * cc$obs + (1 - cc$mask) * (x + dt_fm * v)
    }
  })
  list(I_gen = as.matrix(torch_clamp(x, min = 0)$cpu()$detach()) * ob$N,
       T_obs = T_obs, I_obs_raw = ob$I_raw, label = ob$label, N = ob$N)
}

conditional_stochastic_interpolants <- function(fitted, ob, T_obs, n_samples = EVAL_ENS,
                                                rho, n_steps = N_SI) {
  model <- fitted$model; model$eval(); T_obs <- as.integer(T_obs)
  cc <- .cond_common(ob$I_norm, T_obs, log(ob$N), n_samples, rho)
  x <- cc$mask * cc$obs + (1 - cc$mask) * torch_randn(c(n_samples, L), device = DEVICE)
  dt_si <- 1 / n_steps
  with_no_grad({
    for (i in seq_len(n_steps)) {
      t0 <- (i - 1) * dt_si + 1e-3
      tt <- torch_tensor(rep(t0, n_samples), dtype = torch_float32(), device = DEVICE)$unsqueeze(2L)
      out <- model(stack_signal_mask(x, cc$mask), tt, cc$logN, cc$rho)
      v <- out[, 1, ] + SI_GAMMA_DOT(t0) * out[, 2, ]
      x <- cc$mask * cc$obs + (1 - cc$mask) * (x + dt_si * v)
    }
  })
  list(I_gen = as.matrix(torch_clamp(x, min = 0)$cpu()$detach()) * ob$N,
       T_obs = T_obs, I_obs_raw = ob$I_raw, label = ob$label, N = ob$N)
}

SAMPLER_UNCOND <- list(Diffusion = sample_diffusion,
                       FlowMatching = sample_flow_matching,
                       StochasticInterpolants = sample_stochastic_interpolants)
SAMPLER_COND   <- list(Diffusion = conditional_diffusion,
                       FlowMatching = conditional_flow_matching,
                       StochasticInterpolants = conditional_stochastic_interpolants)

denormalize_by_N <- function(X, N) sweep(X, 1, N, "*")

.sample_chunked <- function(sampler, fitted, logN, rho, chunk = GEN_CHUNK) {
  n <- length(logN); out <- vector("list", ceiling(n / chunk)); k <- 0L
  for (st in seq(1, n, by = chunk)) {
    en <- min(st + chunk - 1L, n); k <- k + 1L
    out[[k]] <- sampler(fitted, log_N_values = logN[st:en], rho_values = rho[st:en])
    if (cuda_is_available()) cuda_empty_cache()
  }
  do.call(rbind, out)
}


# Metrics

# Every functional is a function of (trajectory, N) only, so it is computable
# identically on simulated and generated data.
compute_functionals <- function(I_mat, N_vec, dt_val = DT) {
  stopifnot(length(N_vec) == nrow(I_mat))
  times <- seq(0, by = dt_val, length.out = ncol(I_mat))
  pk <- apply(I_mat, 1, max)
  tibble(peak_I = pk,
         peak_time = times[apply(I_mat, 1, which.max)],
         auc = rowSums(I_mat) * dt_val,
         extinction_time = apply(I_mat, 1, function(r) {
           nz <- which(r > 0); if (!length(nz)) 0 else times[max(nz)] }),
         extinction = (pk / N_vec < EXT_PEAK_FRAC))
}
FID_FUNCS <- c("peak_I", "peak_time", "auc", "extinction_time", "extinction")

final_size_frac <- function(I_mat, N_vec, gamma_vec, dt_val = DT)
  gamma_vec * rowSums(I_mat) * dt_val / N_vec

finalsize_z <- function(R0) {
  if (R0 <= 1) return(0)
  uniroot(function(z) 1 - z - exp(-R0 * z), c(1e-9, 1 - 1e-9))$root
}

# Extinction probability from one case under an Erlang(n, n*gamma) infectious
# period. At n = 1 this reduces to the classical q = 1/R0.
q_erlang <- function(R0, n) {
  if (R0 <= 1) return(1)
  uniroot(function(q) (1 + R0 * (1 - q) / n)^(-n) - q, c(1e-12, 1 - 1e-12))$root
}

cur_wasserstein1 <- function(a, b, ng = 1000) {
  a <- a[is.finite(a)]; b <- b[is.finite(b)]
  if (length(a) < 2 || length(b) < 2) return(NA_real_)
  p <- (seq_len(ng) - 0.5) / ng
  mean(abs(quantile(a, p, names = FALSE) - quantile(b, p, names = FALSE)))
}

# The sample is drawn once and shared by the three terms, so the algebraic
# identity holds and the statistic cannot come out negative.
cur_energy_distance <- function(X, Y, sub = ENERGY_SUB) {
  if (nrow(X) > sub) X <- X[sample(nrow(X), sub), , drop = FALSE]
  if (nrow(Y) > sub) Y <- Y[sample(nrow(Y), sub), , drop = FALSE]
  2 * .mean_pairdist(X, Y) - .mean_pairdist(X, X) - .mean_pairdist(Y, Y)
}

# Squared difference of two sample means, so biased upward by sampling variance:
# read it against the null baseline, never against zero.
cur_variogram_discrepancy <- function(X, Y, p = 0.5) {
  Lc <- ncol(X); tot <- 0
  for (lag in seq_len(min(Lc - 1, 20))) {
    i <- seq_len(Lc - lag); j <- i + lag
    vx <- colMeans(abs(X[, i, drop = FALSE] - X[, j, drop = FALSE])^p)
    vy <- colMeans(abs(Y[, i, drop = FALSE] - Y[, j, drop = FALSE])^p)
    tot <- tot + (1 / lag) * sum((vy - vx)^2)
  }
  tot
}

cur_c2st <- function(fX, fY, folds = 5) {
  d <- rbind(fX, fY); y <- c(rep(1, nrow(fX)), rep(0, nrow(fY)))
  ok <- complete.cases(d); d <- d[ok, , drop = FALSE]; y <- y[ok]
  d <- scale(d); d[!is.finite(d)] <- 0
  n <- length(y); fold <- sample(rep_len(seq_len(folds), n))
  acc <- vapply(seq_len(folds), function(k) {
    tr <- fold != k; te <- !tr
    if (length(unique(y[tr])) < 2) return(NA_real_)
    fit <- suppressWarnings(glm(y[tr] ~ ., data = as.data.frame(d[tr, , drop = FALSE]),
                                family = binomial()))
    pr <- suppressWarnings(predict(fit, as.data.frame(d[te, , drop = FALSE]), type = "response"))
    mean((pr > 0.5) == (y[te] == 1))
  }, numeric(1))
  a <- mean(acc, na.rm = TRUE)
  se <- sqrt(0.25 / n)
  list(accuracy = a, p_value = pnorm(a, 0.5, se, lower.tail = FALSE))
}

cur_energy_score <- function(ens, y) {
  K <- nrow(ens); if (K < 2) return(NA_real_)
  as.numeric(mean(sqrt(rowSums((ens - matrix(y, K, length(y), byrow = TRUE))^2))) -
             0.5 * .mean_pairdist(ens, ens))
}

cur_crps <- function(ens, y) {
  mean(vapply(seq_along(y), function(k) {
    x <- ens[, k]
    mean(abs(x - y[k])) - 0.5 * mean(abs(outer(x, x, "-")))
  }, numeric(1)))
}

compute_wis <- function(ens, y, alphas = c(0.02, 0.05, seq(0.1, 0.9, 0.1))) {
  K <- length(alphas)
  vapply(seq_along(y), function(j) {
    x <- ens[, j]; yy <- y[j]
    isum <- sum(vapply(alphas, function(a) {
      lo <- quantile(x, a / 2, names = FALSE); hi <- quantile(x, 1 - a / 2, names = FALSE)
      (a / 2) * ((hi - lo) + (2 / a) * max(lo - yy, 0) + (2 / a) * max(yy - hi, 0))
    }, numeric(1)))
    (0.5 * abs(yy - median(x)) + isum) / (K + 0.5)
  }, numeric(1))
}

cur_hdr_coverage <- function(ens, y, level = 0.80) {
  vapply(seq_len(ncol(ens)), function(t) {
    x <- ens[, t]
    if (sd(x) < 1e-9) return(abs(y[t] - x[1]) < 1e-6)
    dn <- density(x, n = 256)
    fx <- approx(dn$x, dn$y, x, rule = 2)$y
    approx(dn$x, dn$y, y[t], rule = 2)$y >= quantile(fx, 1 - level, names = FALSE)
  }, logical(1))
}

# Modified band depth: how central the observation sits inside the ensemble. N
# counts the ensemble plus the observation, otherwise the top-ranked curve gets a
# negative band count.
cur_banddepth_rank <- function(ens, y) {
  if (nrow(ens) < 2) return(1L)
  curves <- rbind(ens, matrix(y, 1))
  N <- nrow(curves)
  R <- apply(curves, 2, function(col) rank(col, ties.method = "average"))
  mbd <- (rowMeans((R - 1) * (N - R)) + (N - 1)) / (N * (N - 1) / 2)
  as.integer(sum(mbd[-length(mbd)] < mbd[length(mbd)]) + 1L)
}

# NA rather than NaN when the reference score is zero, which happens on extinct
# trajectories and would otherwise blank out a quarter of the rows.
.skill <- function(es, ref) {
  if (!is.finite(ref) || !is.finite(es) || ref <= 0) return(NA_real_)
  1 - es / ref
}

cur_bootstrap_ci <- function(d, B = 2000, level = 0.95) {
  d <- d[is.finite(d)]
  if (length(d) < 2) return(c(mean = NA, lo = NA, hi = NA))
  bs <- replicate(B, mean(sample(d, replace = TRUE))); a <- (1 - level) / 2
  c(mean = mean(d), lo = quantile(bs, a, names = FALSE), hi = quantile(bs, 1 - a, names = FALSE))
}

compute_distributional_metrics <- function(I_true, N_true, I_gen, N_gen,
                                           model_name, stratum = "all") {
  f_true <- compute_functionals(I_true, N_true)
  f_gen  <- compute_functionals(I_gen,  N_gen)
  w1 <- vapply(FID_FUNCS, function(f)
    cur_wasserstein1(as.numeric(f_true[[f]]), as.numeric(f_gen[[f]])), numeric(1))
  c2 <- cur_c2st(as.matrix(f_true[, FID_FUNCS]), as.matrix(f_gen[, FID_FUNCS]))
  # The classifier is also run on each functional alone and with each removed,
  # which is what localises the discrepancy.
  c2_solo <- vapply(FID_FUNCS, function(f)
    cur_c2st(as.matrix(f_true[, f, drop = FALSE]),
             as.matrix(f_gen[, f, drop = FALSE]))$accuracy, numeric(1))
  c2_sans <- vapply(FID_FUNCS, function(f) {
    rest <- setdiff(FID_FUNCS, f)
    cur_c2st(as.matrix(f_true[, rest, drop = FALSE]),
             as.matrix(f_gen[, rest, drop = FALSE]))$accuracy
  }, numeric(1))
  tb <- tibble(model = model_name, stratum = stratum, n_true = nrow(I_true),
               energy_distance = cur_energy_distance(.logt(I_true), .logt(I_gen)),
               variogram = cur_variogram_discrepancy(.logt(I_true), .logt(I_gen)),
               C2ST_acc = c2$accuracy, C2ST_p = c2$p_value)
  for (f in FID_FUNCS) tb[[paste0("W1_", f)]]        <- unname(w1[f])
  for (f in FID_FUNCS) tb[[paste0("C2ST_solo_", f)]] <- unname(c2_solo[f])
  for (f in FID_FUNCS) tb[[paste0("C2ST_sans_", f)]] <- unname(c2_sans[f])
  tb
}

# Unlike the fidelity block, generated trajectories are not rounded here: these
# scores contain no threshold functional, and rounding would only coarsen the
# predictive ensemble.
compute_forecasting_metrics <- function(cond, sim_pool, model_name, cut_label,
                                        rho_val, L_obs = L) {
  T_obs <- cond$T_obs; out <- list()
  for (h in EVAL_HORIZONS) {
    end_idx <- T_obs + h
    if (end_idx > L_obs) next
    idx <- (T_obs + 1):end_idx
    obs <- cond$I_obs_raw[idx]
    gen <- cond$I_gen[, idx, drop = FALSE]

    stratum  <- .Nstrat(cond$N)
    pool_idx <- which(.Nstrat(sim_pool$N) == stratum)
    if (length(pool_idx) < 5) pool_idx <- seq_len(nrow(sim_pool$I))
    clim <- sim_pool$I[sample(pool_idx, nrow(gen), replace = TRUE), idx, drop = FALSE]

    last <- cond$I_obs_raw[T_obs]
    sd_step <- if (T_obs >= 3) sd(diff(cond$I_obs_raw[1:T_obs])) else 1
    if (!is.finite(sd_step) || sd_step <= 0) sd_step <- 1
    pers <- pmax(matrix(last, nrow(gen), h) + matrix(rnorm(nrow(gen) * h, 0, sd_step), nrow(gen), h), 0)

    lg <- .logt(gen); lo <- .logt(obs)
    es  <- cur_energy_score(lg, lo)
    esc <- cur_energy_score(.logt(clim), lo)
    esp <- cur_energy_score(.logt(pers), lo)
    out[[length(out) + 1]] <- tibble(
      outbreak = cond$label, model = model_name, cut_label = cut_label,
      rho = rho_val, T_obs = T_obs, horizon = h,
      energy_score = es, crps = cur_crps(lg, lo),
      WIS = mean(compute_wis(lg, lo), na.rm = TRUE),
      cov50 = mean(cur_hdr_coverage(gen, obs, 0.50), na.rm = TRUE),
      cov80 = mean(cur_hdr_coverage(gen, obs, 0.80), na.rm = TRUE),
      cov95 = mean(cur_hdr_coverage(gen, obs, 0.95), na.rm = TRUE),
      bd_rank = cur_banddepth_rank(lg, lo),
      skill_clim = .skill(es, esc), skill_pers = .skill(es, esp))
  }
  if (!length(out)) return(NULL)
  do.call(rbind, out)
}

make_outbreak <- function(traj, label, N_val) {
  raw <- as.numeric(traj); raw[!is.finite(raw)] <- 0; raw[raw < 0] <- 0
  list(label = label, I_raw = raw, I_norm = raw / N_val, N = N_val)
}

# Effective length is the last active day plus a small margin, not the zero-padded
# grid: otherwise the declining cut lands in the dead post-epidemic zone.
.effective_L <- function(I_vec, thr = 0.5, margin = 3L) {
  nz <- which(I_vec > thr)
  if (!length(nz)) return(NA_integer_)
  as.integer(min(length(I_vec), max(nz) + margin))
}

.cutpoints <- function(I_vec, L_obs = length(I_vec)) {
  if (is.na(L_obs) || L_obs < 8L) return(integer(0))
  pk <- which.max(I_vec[seq_len(L_obs)])
  cuts <- c(rising = max(3L, round(pk * 0.6)), peak = pk,
            declining = min(L_obs - 5L, round(pk + (L_obs - pk) * 0.4)))
  cuts[cuts >= 3L & cuts < L_obs - 2L]
}

.is_minor <- function(I_vec, N_val) (max(I_vec) / max(N_val, 1)) < EXT_PEAK_FRAC


# Simulator validation

cat("\nsimulator checks\n")
.chk_R0 <- c(1.5, 2.5, 4, 8); .chk_n <- 1L
cat(sprintf("  Erlang extinction at n = 1 vs 1/R0: max deviation %.1e\n",
            max(abs(vapply(.chk_R0, q_erlang, numeric(1), n = .chk_n) - 1 / .chk_R0))))
.mean_TI <- mean(final_size_frac(I_train, N_train_values, rep(1, nrow(I_train))) /
                 pmax(rowSums(I_train > 0) * 0 + 1, 1))
cat(sprintf("  mean infectious period from the renewal identity: %.3f d\n",
            mean(rowSums(I_train) * DT / pmax(apply(I_train, 1, function(r) sum(diff(c(0, cummax(r)))> 0)), 1),
                 na.rm = TRUE)))


# Emulation fidelity

cat("\nemulation fidelity\n")
set.seed(1717)
sim <- list(I = I_train, N = N_train_values, gamma = gamma_train)
n_fid     <- min(N_GEN_FIDELITY, nrow(sim$I))
idx_true  <- sample(nrow(sim$I), n_fid)
N_fid     <- sim$N[idx_true]
logN_fid  <- log(N_fid)
rho_fid   <- .draw_rho(n_fid)
I_ref     <- .apply_observation(sim$I[idx_true, , drop = FALSE], rho_fid)
strat_fid <- .Nstrat(N_fid)

fid <- list(); GEN_FID <- list()
for (mdl in MODEL_NAMES) {
  g <- .sample_chunked(SAMPLER_UNCOND[[mdl]], FITS[[mdl]], logN_fid, rho_fid)
  # Generated paths are continuous and never hit exactly zero, so extinction_time
  # would sit at the end of the window for all of them. Rounding to integer counts
  # makes generated and true directly comparable.
  I_gen <- round(pmax(denormalize_by_N(g, N_fid), 0))
  GEN_FID[[mdl]] <- I_gen
  for (st in c(DECADES, "all")) {
    ii <- if (st == "all") seq_len(n_fid) else which(strat_fid == st)
    if (length(ii) < 50) next
    fid[[length(fid) + 1]] <- compute_distributional_metrics(
      I_true = I_ref[ii, , drop = FALSE], N_true = N_fid[ii],
      I_gen  = I_gen[ii, , drop = FALSE], N_gen  = N_fid[ii],
      model_name = mdl, stratum = st)
  }
  rm(g); invisible(gc())
  cat(sprintf("  %s done\n", mdl))
}

# Null baseline: two disjoint halves of the simulator scored against each other.
# Every metric has a non-zero value at finite sample size, and without this
# reference a C2ST of 0.55 cannot be read.
hA <- sample(n_fid, floor(n_fid / 2)); hB <- setdiff(seq_len(n_fid), hA)
for (st in c(DECADES, "all")) {
  ia <- if (st == "all") hA else hA[strat_fid[hA] == st]
  ib <- if (st == "all") hB else hB[strat_fid[hB] == st]
  if (length(ia) < 50 || length(ib) < 50) next
  fid[[length(fid) + 1]] <- compute_distributional_metrics(
    I_true = I_ref[ia, , drop = FALSE], N_true = N_fid[ia],
    I_gen  = I_ref[ib, , drop = FALSE], N_gen  = N_fid[ib],
    model_name = "NullBaseline", stratum = st)
}
FID <- do.call(rbind, fid)
fwrite(FID, file.path(OUT_DIR, "fidelity.csv"))
FID_CTX <- list(I_ref = I_ref, N = N_fid, strat = strat_fid, gen = GEN_FID)


# Conditional forecasting on held-out simulations

cat("\nforecasting on held-out simulations\n")
set.seed(4242)
n_h   <- min(EVAL_FC_N, nrow(I_held))
sel   <- sample(nrow(I_held), n_h)
rho_h <- .draw_rho(n_h)
I_h_obs <- .apply_observation(I_held[sel, , drop = FALSE], rho_h)

fc <- list()
pb <- progress_bar$new(format = "  [forecast] [:bar] :percent ETA: :eta",
                       total = n_h, clear = FALSE, width = 60)
for (i in seq_len(n_h)) {
  pb$tick()
  ob  <- make_outbreak(I_h_obs[i, ], paste0("heldout-", sel[i]), N_held_values[sel[i]])
  cts <- .cutpoints(ob$I_raw, .effective_L(ob$I_raw))
  for (cl in names(cts)) {
    for (mdl in MODEL_NAMES) {
      cond <- SAMPLER_COND[[mdl]](FITS[[mdl]], ob, cts[[cl]], n_samples = EVAL_ENS, rho = rho_h[i])
      r <- compute_forecasting_metrics(cond, sim, mdl, cl, rho_h[i])
      if (!is.null(r)) {
        # Minor outbreaks are kept but tagged, since forecasting an extinction is
        # trivial and would otherwise inflate the averages.
        r$regime <- if (.is_minor(I_held[sel[i], ], N_held_values[sel[i]])) "minor" else "established"
        fc[[length(fc) + 1]] <- r
      }
    }
  }
}
FC <- do.call(rbind, fc)
fwrite(FC, file.path(OUT_DIR, "forecasting_raw.csv"))
fwrite(aggregate(cbind(energy_score, crps, WIS, cov80, bd_rank, skill_clim, skill_pers) ~
                   model + cut_label + horizon + regime, data = as.data.frame(FC),
                 FUN = mean, na.rm = TRUE, na.action = na.pass),
       file.path(OUT_DIR, "forecasting_summary.csv"))

w <- reshape(as.data.frame(FC[, c("outbreak", "cut_label", "horizon", "model", "energy_score")]),
             idvar = c("outbreak", "cut_label", "horizon"), timevar = "model", direction = "wide")
prs <- list(c("FlowMatching", "Diffusion"),
            c("StochasticInterpolants", "Diffusion"),
            c("FlowMatching", "StochasticInterpolants"))
br <- do.call(rbind, lapply(prs, function(p) {
  ca <- paste0("energy_score.", p[1]); cb <- paste0("energy_score.", p[2])
  stopifnot(ca %in% names(w), cb %in% names(w))
  ci <- cur_bootstrap_ci(w[[ca]] - w[[cb]])
  data.frame(comparison = paste(p[1], "-", p[2]), mean = ci["mean"],
             lo = ci["lo"], hi = ci["hi"], significant = !(ci["lo"] < 0 & ci["hi"] > 0))
}))
fwrite(br, file.path(OUT_DIR, "model_diff_bootstrap.csv"))


# Historical outbreaks

load_flu1978 <- function() {
  d <- outbreaks::influenza_england_1978_school
  list(I_obs = d$in_bed, N = 763, disease = "flu1978", gamma = 1 / 2.5)
}

.hagelloch_prevalence <- function(tI, tR, N_val) {
  ok <- is.finite(tI) & is.finite(tR) & (tR > tI)
  tI <- tI[ok]; tR <- tR[ok]
  if (!length(tI)) stop("hagelloch: no usable (tI, tR) pair")
  g <- floor(min(tI)):ceiling(max(tR))
  list(I_obs = vapply(g, function(t) sum(tI <= t & t < tR), numeric(1)),
       N = N_val, disease = "hagelloch", gamma = 1 / 8)
}

load_hagelloch <- function() {
  df <- tryCatch({
    e <- new.env()
    utils::data("hagelloch", package = "surveillance", envir = e)
    if (exists("hagelloch.df", envir = e, inherits = FALSE)) get("hagelloch.df", envir = e) else NULL
  }, error = function(err) NULL)
  if (!is.null(df) && all(c("tI", "tR") %in% names(df)))
    return(.hagelloch_prevalence(as.numeric(df$tI), as.numeric(df$tR), nrow(df)))
  ll <- outbreaks::measles_hagelloch_1861
  t0 <- min(as.numeric(ll$date_of_prodrome), na.rm = TRUE)
  .hagelloch_prevalence(as.numeric(ll$date_of_prodrome) - t0,
                        as.numeric(ll$date_of_rash) - t0 + 4, nrow(ll))
}

run_real_sweep <- function(diseases = DISEASES) {
  cat("\nhistorical outbreaks\n")
  loaders <- list(flu1978 = load_flu1978, hagelloch = load_hagelloch)
  outs <- list()
  for (dz in diseases) {
    d <- loaders[[dz]]()
    I_raw <- as.numeric(d$I_obs); N_val <- d$N
    if (length(I_raw) > L) {
      pk <- which.max(I_raw); s <- max(1, pk - floor(L / 2))
      I_raw <- I_raw[s:min(length(I_raw), s + L - 1)]
    }
    L_obs <- length(I_raw)
    ob <- make_outbreak(c(I_raw, rep(0, max(0, L - L_obs)))[1:L], dz, N_val)
    cat(sprintf("  %s: N = %d, L_obs = %d, peak = %.0f\n", dz, N_val, L_obs, max(I_raw)))
    rho_grid <- .rho_feasible(I_raw, N_val, d$gamma)
    cuts <- .cutpoints(I_raw, L_obs)
    if (!length(cuts)) next
    for (cl in names(cuts)) for (mdl in MODEL_NAMES) for (rv in rho_grid) {
      cond <- SAMPLER_COND[[mdl]](FITS[[mdl]], ob, cuts[[cl]], n_samples = EVAL_ENS, rho = rv)
      r <- compute_forecasting_metrics(cond, list(I = I_train, N = N_train_values),
                                       mdl, cl, rv, L_obs = L_obs)
      if (!is.null(r)) {
        r$disease <- dz; r$L_obs <- L_obs; r$rho_min <- max(I_raw) / N_val
        outs[[length(outs) + 1]] <- r
      }
    }
  }
  if (!length(outs)) return(NULL)
  df <- do.call(rbind, outs)
  fwrite(df, file.path(OUT_DIR, "realdata_rho_sweep.csv"))
  df
}
SWEEP <- run_real_sweep()


# Figures

# Axis labels only: no title, no subtitle, no legend title.
.cap <- function(x) { x <- as.character(x); paste0(toupper(substring(x, 1, 1)), substring(x, 2)) }

MODEL_LAB  <- c(Diffusion = "Diffusion", FlowMatching = "Flow matching",
                StochasticInterpolants = "Stochastic interpolants")
FUNC_LAB   <- c(peak_I = "Peak height", peak_time = "Peak time",
                auc = "Area under the curve", extinction_time = "Time to extinction",
                extinction = "Extinction indicator")
DECADE_LAB <- c(small = "Small", medium = "Medium", large = "Large")
CUT_LAB    <- c(rising = "Rising", peak = "Peak", declining = "Declining")

theme_fig <- theme_minimal(base_size = 11) +
  theme(plot.title = element_blank(), plot.subtitle = element_blank(),
        plot.caption = element_blank(), legend.title = element_blank(),
        legend.position = "bottom", strip.text = element_text(face = "plain", size = 10),
        panel.grid.minor = element_blank())

.save_fig <- function(name, p, w, h) {
  p <- p + labs(title = NULL, subtitle = NULL, caption = NULL) + theme_fig
  ggsave(file.path(FIG_DIR, paste0(name, ".png")), p, width = w, height = h, dpi = 300)
  ggsave(file.path(FIG_DIR, paste0(name, ".pdf")), p, width = w, height = h)
  cat(sprintf("  %s\n", name))
}

cat("\nfigures\n")

# Figure 1. I_train is the pool before thinning, so I(t) is the right symbol.
set.seed(1)
k  <- min(400L, nrow(I_train)); ii <- sample(nrow(I_train), k)
reg <- P$stratum[match(train_sim_ids, P$sim_id)]
d1 <- do.call(rbind, lapply(seq_along(ii), function(j) {
  i <- ii[j]
  data.frame(t = seq(0, by = DT, length.out = L), y = I_train[i, ] / N_train_values[i],
             id = j, regime = reg[i], decade = .Nstrat(N_train_values[i]))
}))
d1$decade <- factor(unname(DECADE_LAB[d1$decade]), levels = unname(DECADE_LAB[DECADES]))
d1$regime <- factor(.cap(d1$regime), levels = .cap(names(STRATA)))
.save_fig("fig_01_pool_trajectories",
          ggplot(d1, aes(t, y, group = id)) +
            geom_line(alpha = 0.18, linewidth = 0.3) +
            facet_grid(regime ~ decade, scales = "free_y") +
            labs(x = "Day", y = "I(t) / N"),
          w = 10, h = 9)

# Figure 3. The five functionals live on incompatible scales and N is log-uniform
# within each decade, so a natural scale collapses the densities against the axis.
# The +1 keeps extinction_time = 0 finite. The transform compresses the right
# tail, where the extinction discrepancy sits, so read this figure with table 3.
long_one <- function(mat, src) {
  f <- compute_functionals(mat, FID_CTX$N)
  do.call(rbind, lapply(names(FUNC_LAB), function(fn)
    data.frame(functional = FUNC_LAB[[fn]], decade = FID_CTX$strat,
               v = as.numeric(f[[fn]]), source = src)))
}
d3 <- rbind(long_one(FID_CTX$I_ref, "Simulated"),
            do.call(rbind, lapply(names(FID_CTX$gen), function(m)
              long_one(FID_CTX$gen[[m]], unname(MODEL_LAB[m])))))
d3$decade     <- factor(unname(DECADE_LAB[d3$decade]), levels = unname(DECADE_LAB[DECADES]))
d3$functional <- factor(d3$functional, levels = unname(FUNC_LAB))
d3$source     <- factor(d3$source, levels = c("Simulated", unname(MODEL_LAB[MODEL_NAMES])))
d3$lv <- log1p(pmax(d3$v, 0))
.save_fig("fig_03_fidelity_functionals",
          ggplot(d3, aes(lv, colour = source)) +
            geom_density(linewidth = 0.6) +
            facet_grid(functional ~ decade, scales = "free") +
            labs(x = "log(1 + Value)", y = "Density"),
          w = 11, h = 9)

# Figure 4. At rho = 1 the thinning is the identity, so the observed series and
# the true prevalence coincide and I(t) is exact. The x axis is trimmed to the
# active window; the influenza series is 14 days against a grid of 126.
FIG4_RHO <- 1.0; FIG4_NDRAW <- 40L
COL_TRUTH <- "#3B2E7E"; COL_GEN <- "#D9A441"

fig4_one <- function(loader, fname) {
  d <- loader()
  I_raw <- as.numeric(d$I_obs); N_val <- d$N
  if (length(I_raw) > L) {
    pk <- which.max(I_raw); s <- max(1L, pk - floor(L / 2))
    I_raw <- I_raw[s:min(length(I_raw), s + L - 1L)]
  }
  L_obs <- length(I_raw)
  ob <- make_outbreak(c(I_raw, rep(0, max(0, L - L_obs)))[1:L], d$disease, N_val)
  cuts <- .cutpoints(I_raw, L_obs)
  if (!length(cuts)) return(invisible(NULL))
  t_axis <- seq(0, by = DT, length.out = L)

  gen_l <- list(); tr_l <- list(); to_l <- list()
  for (cl in names(cuts)) {
    T_obs <- as.integer(cuts[[cl]])
    for (mdl in MODEL_NAMES) {
      cond <- SAMPLER_COND[[mdl]](FITS[[mdl]], ob, T_obs, n_samples = FIG4_NDRAW, rho = FIG4_RHO)
      G <- cond$I_gen; ns <- nrow(G)
      gen_l[[length(gen_l) + 1L]] <- data.frame(
        model = unname(MODEL_LAB[mdl]), cut_label = cl,
        sample = rep(seq_len(ns), each = L),
        t = rep(t_axis, times = ns), I = as.vector(t(G)))
    }
    tr_l[[length(tr_l) + 1L]] <- data.frame(cut_label = cl, t = t_axis, I = ob$I_raw)
    to_l[[length(to_l) + 1L]] <- data.frame(cut_label = cl, x = (T_obs - 1L) * DT)
  }
  gen <- do.call(rbind, gen_l); truth <- do.call(rbind, tr_l); tobs <- do.call(rbind, to_l)
  lev <- intersect(CUT_LABELS, unique(as.character(gen$cut_label)))
  lab <- unname(CUT_LAB[lev])
  for (nm in c("gen", "truth", "tobs")) {
    x <- get(nm); x$cut_label <- factor(unname(CUT_LAB[as.character(x$cut_label)]), levels = lab)
    assign(nm, x)
  }
  gen$model <- factor(gen$model, levels = unname(MODEL_LAB[MODEL_NAMES]))
  active <- c(truth$t[truth$I > 0.5], gen$t[gen$I > 0.5])
  xmax <- if (length(active)) min((L - 1) * DT, max(active) + 4 * DT) else (L - 1) * DT

  .save_fig(fname,
            ggplot(gen, aes(t, I, group = interaction(model, sample))) +
              geom_line(colour = COL_GEN, alpha = 0.30, linewidth = 0.35) +
              geom_line(data = truth, aes(t, I), inherit.aes = FALSE,
                        colour = COL_TRUTH, linewidth = 1.05) +
              geom_vline(data = tobs, aes(xintercept = x), inherit.aes = FALSE,
                         linetype = "dashed", colour = "#888888", linewidth = 0.5) +
              facet_grid(model ~ cut_label) +
              coord_cartesian(xlim = c(0, xmax)) +
              labs(x = "Day", y = "I(t)"),
            w = 11, h = 8)
}
fig4_one(load_flu1978,   "fig_04a_flu1978")
fig4_one(load_hagelloch, "fig_04b_hagelloch")


# Distribution shift

# How far the two historical outbreaks sit outside the training pool. The peak
# prevalence needs no methodological choice; the growth rate does, so the fitting
# window is swept and the same window is applied to both sides of the comparison.
verify_shift <- function() {
  cat("\ndistribution shift\n")
  r_fit <- function(x, n, floor_ = 1) {
    i <- which(x >= floor_)
    if (!length(i)) return(NA_real_)
    j <- i[1]:min(i[1] + n - 1L, length(x))
    if (length(j) < 3L) return(NA_real_)
    y <- log(pmax(x[j], 1e-9)); tt <- seq_along(j) - 1
    if (length(unique(y)) < 3L) return(NA_real_)
    unname(coef(lm(y ~ tt))[2])
  }

  h <- load_hagelloch()
  pk_hag  <- max(as.numeric(h$I_obs)) / h$N
  pk_pool <- apply(I_train, 1, max) / N_train_values
  cat(sprintf("  Hagelloch peak/N = %.3f, reached by %.1f%% of the pool\n",
              pk_hag, 100 * mean(pk_pool >= pk_hag)))

  flu <- as.numeric(load_flu1978()$I_obs)
  cat("  window   r(flu)   pool reaching it   valid fits\n")
  res <- do.call(rbind, lapply(3:8, function(n) {
    r_flu  <- r_fit(flu, n)
    r_pool <- apply(I_train, 1, r_fit, n = n)
    frac   <- 100 * mean(r_pool >= r_flu, na.rm = TRUE)
    cat(sprintf("  %2d days  %6.3f   %13.1f%%   %5d\n", n, r_flu, frac, sum(!is.na(r_pool))))
    data.frame(window_days = n, r_flu = r_flu, pct_pool = frac,
               n_valid = sum(!is.na(r_pool)), peak_frac_hagelloch = pk_hag,
               pct_pool_peak = 100 * mean(pk_pool >= pk_hag))
  }))
  fwrite(res, file.path(OUT_DIR, "distribution_shift_check.csv"))
  invisible(res)
}
verify_shift()

cat(sprintf("\ndone. outputs in %s\n", normalizePath(OUT_DIR)))

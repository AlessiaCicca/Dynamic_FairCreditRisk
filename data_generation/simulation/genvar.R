# =============================================================================
# Adapted from:
#   "Dynamic Estimation with Random Forests for Discrete-Time Survival Data" (2021)
# Modifications:
#   - Fixed simulation parameters
#   - Added 4 fairness scenarios (fair, direct, proxy, temporal)
# =============================================================================

genvar <- function(nsub = 1000, 
                   matsigma = NULL,
                   scenario= c("fair", "direct", "proxy", "temporal")){

  
  # Generates a simulated longitudinal dataset with:
  # - nsub subjects
  # - 6 covariates (X1–X6)
  # - 12 time points
  # - a sensitive binary group S
  # - different fairness/data-generating mechanism depending on "scenario"
    

  coefficients <- create_coeff(scenario = scenario,  
                          nsub = nsub)
  Coeff <- coefficients$Coeff

  #SET BASIC DIMENSIONS
  ncov <- 6
  ncovfixed <- 2
  nperiod <- 24
  n1seq <- ncovfixed * nsub    #TI variables
  n2seq <- (ncov - ncovfixed) * nsub    #TV variables

  #Binary sensitive attribute S ∈ {0,1}
  #Distribution depends on scenario: biased scenarios → 30% ones / fair scenario → 50% ones
  
  if (scenario %in% c("direct", "proxy","temporal")) {
    S <- rbinom(nsub, size = 1, prob = 0.3) 
    }
  else{
    S <- rbinom(nsub, size = 1, prob = 0.5) 
    }

  #Splits indices into group 0 and group 1
  idx_S0 <- which(S == 0)
  idx_S1 <- which(S == 1)


  # S=0: high covariance inter-feature → stable signal
  # S=1: low covariance inter-feature → noise
  if (scenario %in% c("proxy", "temporal")) {
    mats        <- create_matsigma_by_group(S)
    matsigma_S0 <- mats$S0
    matsigma_S1 <- mats$S1
  }

  # =============================================================================
  #
  # First is to generate a normal VAR process
  # z0 = e_0                                                      
  # z1 = A * e_0 + e_1                                             
  # z2 = A * z1 + e2 = A^2 * e_0 +   A * e_1 +     e_2             
  # z3 = A * z2 + e3 = A^3 * e_0 + A^2 * e_1 + A * e_2 + e_3       
  #The VAR (Vector AutoRegressive) model is used to generate and model multiple variables that evolve over time 
  # and influence each other, meaning to create realistic longitudinal data.
  #
  # =============================================================================

  # Creates a list of length nperiod and initialize each element as a standard normal random values
 
  z <- rep(list(0), nperiod) 
  z[[1]] <- matrix(rnorm(ncov * nsub), nrow = ncov, ncol = nsub)  
  
  for (pp in 2:nperiod) {
    noise <- matrix(c(rep(0, n1seq), rnorm(n2seq)),
                    nrow = ncov, ncol = nsub, byrow = TRUE)
    
    if (scenario %in% c("proxy","temporal")) {
      z[[pp]] <- z[[pp-1]] 
      z[[pp]][, idx_S0] <- matsigma_S0 %*% z[[pp-1]][, idx_S0] + noise[, idx_S0]
      z[[pp]][, idx_S1] <- matsigma_S1 %*% z[[pp-1]][, idx_S1] + noise[, idx_S1]
    } else {
      z[[pp]] <- matsigma %*% z[[pp-1]] + noise
    }
}

  
  Data <- matrix(NA, nrow = nperiod * nsub, ncol = ncov + 1)
  colnames(Data) <- c(paste0("X", 1:ncov), "S")
  rownames(Data) <- rep(1:nsub, each = nperiod)
  
  Data[, "S"] <- rep(S, each = nperiod)
  
  Data[, c("X3","X4","X5","X6")] <- sapply((ncovfixed + 1):ncov, function(jj) 
    as.vector(t(sapply(z, function(zz) zz[jj, ]))))
  Data[, "X1"] <- rep(as.numeric(z[[1]][1, ] > 0), each = nperiod)
  Data[, "X2"] <- rep(pnorm(z[[1]][2, ]), each = nperiod)
  rm(z)
  Data[, "X3"] <- as.numeric(Data[, "X3"] > 0)
  Data[, "X4"] <- pnorm(Data[, "X4"])
  Data[, "X5"] <- (Data[, "X5"] < qnorm(.2)) + 
    2 * (Data[, "X5"] >= qnorm(.2)) * (Data[, "X5"] < qnorm(.4)) + 
    3 * (Data[, "X5"] >= qnorm(.4)) * (Data[, "X5"] < qnorm(.6)) + 
    4 * (Data[, "X5"] >= qnorm(.6)) * (Data[, "X5"] < qnorm(.8)) +
    5 * (Data[, "X5"] >= qnorm(.8))
  Data[, "X6"] <- pnorm(Data[, "X6"]) * 2


  # --- Feature-level noise on X4 e X6 for S=1 (proxy e temporal) ---
  # Corrupts the predictive signal for the disadvantaged group
  # Adding Gaussian noise → increases the variance, not the mean
  if (scenario %in% c("proxy", "temporal")) {
    S_rep      <- rep(S, each = nperiod)
    sigma_noise <- 0.3 
    n_S1        <- sum(S_rep == 1)
    Data[S_rep == 1, "X4"] <- Data[S_rep == 1, "X4"] + rnorm(n_S1, 0, sigma_noise)
    Data[S_rep == 1, "X6"] <- Data[S_rep == 1, "X6"] + rnorm(n_S1, 0, sigma_noise)
  }
                     
  # --- Mean shift ---
  Gamma_vec <- c(0, -1, 0, -1, 0, 1) * Coeff$Gamma  
  
  if (scenario == "proxy") {
    S_rep <- rep(S, each = nperiod)
    for (jj in 1:ncov) {
      if (Gamma_vec[jj] != 0) {
        Data[, jj] <- Data[, jj] + Gamma_vec[jj] * S_rep
      }
    }
    
  } else if (scenario == "temporal") {
    S_rep    <- rep(S, each = nperiod)
    time_idx <- rep(1:nperiod, times = nsub)
    for (jj in 1:ncov) {
      if (Gamma_vec[jj] != 0) {
        Data[, jj] <- Data[, jj] + Gamma_vec[jj] * S_rep * log(time_idx)
      }
    }
  }

  return(Data)
}

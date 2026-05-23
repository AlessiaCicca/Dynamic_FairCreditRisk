######################################################################################################
genvar <- function(nsub = 1000, 
                   matsigma = NULL,
                   scenario= c("fair", "direct", "proxy", "temporal")){
  #######################################################################
  # GOAL #
  # Produce dataset of subject size nsub, with 6 + 1 number of covariates, 
  # The values of covariates are correlated.
  # In addition, the values of time-varying covariates are autoregressive.
  # The coefficients are determined by a autocorrelation matrix 
  #
  # INPUT # 
  # nsub = number of subjects
  # matsigma = matrix for the VAR process, of size ncov x ncov
  #            (Z_t = matsigma Z_{t-1} + epsilon_t; 
  #             where epsilon_t are iid standard multivariate normal)
  # 
  # OUTPUT # 
  # A dataframe with nperiod * nsub rows and 6 columns 
  #######################################################################

  coefficients <- create_coeff(scenario = scenario,  
                          nsub = nsub)
  Coeff <- coefficients$Coeff
  
  ncov <- 6
  ncovfixed <- 2
  nperiod <- 12
  ## First is to generate a normal VAR process
  # z0 = e_0                                                       # Time 0: e is of size ncov times nsub
  # z1 = A * e_0 + e_1                                             # Time 1: e is of size ncov times nsub
  # z2 = A * z1 + e2 = A^2 * e_0 +   A * e_1 +     e_2             # Time 2: e is of size ncov times nsub
  # z3 = A * z2 + e3 = A^3 * e_0 + A^2 * e_1 + A * e_2 + e_3       # Time 3: e is of size ncov times nsub
  
  z <- rep(list(0), nperiod)
  ## Each of size ncov x nsub, for ncov covariates and nsub subjects
  z[[1]] <- matrix(rnorm(ncov * nsub), nrow = ncov, ncol = nsub)   
  n1seq <- ncovfixed * nsub
  n2seq <- (ncov - ncovfixed) * nsub
  
  if (scenario %in% c("direct", "proxy","temporal")) {
    S <- rbinom(nsub, size = 1, prob = 0.3) 
    }
  else{
    S <- rbinom(nsub, size = 1, prob = 0.5) 
    }
  
  idx_S0 <- which(S == 0)
  idx_S1 <- which(S == 1)
  
  
  # matsigma per gruppo solo per direct e proxy
  if (scenario %in% c("direct", "proxy", "temporal")) {
    matsigma_S0 <- matsigma  # quella originale passata da fuori
    matsigma_S1 <- matsigma
    matsigma_S1[4, 4] <- matsigma[4, 4] + 0.15  # X4 più rumorosa per S=1
    matsigma_S1[6, 6] <- matsigma[6, 6] + 0.15  # X6 più rumorosa per S=1
  }
  z <- rep(list(0), nperiod)
  z[[1]] <- matrix(rnorm(ncov * nsub), nrow = ncov, ncol = nsub)
  

  
  
  for (pp in 2:nperiod) {
    noise <- matrix(c(rep(0, n1seq), rnorm(n2seq)),
                    nrow = ncov, ncol = nsub, byrow = TRUE)
    
    if (scenario %in% c("direct", "proxy","temporal")) {
      z[[pp]] <- z[[pp-1]]  # inizializza
      z[[pp]][, idx_S0] <- matsigma_S0 %*% z[[pp-1]][, idx_S0] + noise[, idx_S0]
      z[[pp]][, idx_S1] <- matsigma_S1 %*% z[[pp-1]][, idx_S1] + noise[, idx_S1]
    } else {
      # fair: comportamento originale
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
  
  
  Gamma_vec <- c(0, -1, 0, -1, 0, 1) * Coeff$Gamma  # Coeff$Gamma scala il tutto (0 per fair/direct)
  
  if (scenario == "proxy") {
    S_rep <- rep(S, each = nperiod)
    for (jj in 1:ncov) {
      if (Gamma_vec[jj] != 0) {
        Data[, jj] <- Data[, jj] + Gamma_vec[jj] * S_rep
      }
    }
    # rumore sistematico su X4 e X6
   # Data[, "X4"] <- Data[, "X4"] + rexp(nrow(Data), rate=1/Coeff$NoiseS) * S_rep
    #Data[, "X6"] <- Data[, "X6"] - rexp(nrow(Data), rate=1/Coeff$NoiseS) * S_rep
    
  } else if (scenario == "temporal") {
    S_rep    <- rep(S, each = nperiod)
    time_idx <- rep(1:nperiod, times = nsub)
    for (jj in 1:ncov) {
      if (Gamma_vec[jj] != 0) {
        Data[, jj] <- Data[, jj] + Gamma_vec[jj] * S_rep * log(time_idx)
      }
    }
    # stesso rumore di proxy ma crescente nel tempo — coerente con semantica temporal
    #Data[, "X4"] <- Data[, "X4"] + rexp(nrow(Data), rate=1/Coeff$NoiseS) * S_rep * log(time_idx)
    #Data[, "X6"] <- Data[, "X6"] - rexp(nrow(Data), rate=1/Coeff$NoiseS) * S_rep * log(time_idx)
  }

  return(Data)
}




######################################################################################################

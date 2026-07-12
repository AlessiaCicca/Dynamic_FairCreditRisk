# =============================================================================
# Adapted from:
#   "Dynamic Estimation with Random Forests for Discrete-Time Survival Data" (2021)
# Modifications:
#   - Fixed simulation parameters
#   - Added 4 fairness scenarios (fair, direct, proxy, temporal)
# =============================================================================


create_theta <- function(data, scenario, coeff){
  # Computes the hazard rate theta = exp(f(X, S)) for each subject-period row.
  # Returns survtime, a vector reporting for each subject, the discrete time period in which the event occurs (e.g. loan default in a credit risk context).
  
  X <- data[, -ncol(data)]   # covariates X1-X6
  S <- data[, ncol(data)]    # sensitive attribute
  if (scenario == "direct"){
      # Direct discrimination: S affects theta both via fixed shift (BetaS)
      # and via subject-specific random noise (epsilon_i), applied only to S=1.
      # Formula: theta = exp(X*Beta1 + Beta0 + BetaS*S + epsilon_i*S)
    
      Fstar <- X %*% coeff$Beta1 + coeff$Beta0[1] + coeff$BetaS * S
      theta <- exp(Fstar)
      
      nperiod <- 24
      nsub    <- nrow(data) / nperiod
  
      # Subject-level noise: one value per subject, repeated across their periods
      noise_per_sub <- rnorm(nsub, mean = 0, sd = coeff$NoiseS)
      noise_rep     <- rep(noise_per_sub, each = nperiod)
      S_first       <- S[seq(1, nrow(data), by = nperiod)]   # S value at period 1 per subject
      S_rep         <- rep(S_first, each = nperiod)
      
      theta <- theta * exp(noise_rep * S_rep)
      return(theta)   
  }
  else if (scenario == "fair" | scenario == "proxy" | scenario == "temporal"){
     Fstar <- X %*% coeff$Beta1 + coeff$Beta0[1] 
  } else {
    stop("Wrong model is set.")
  }
  return(exp(Fstar))
}


# Compute the cumulative hazards function
# Uses an Exponential baseline hazard: H(t) = Lambda * theta * t
ExpHfunc <- function(ts1, ts2, theta, coeff){
  coeff$Lambda * theta * (ts1 - ts2)
}


# Compute the continuous survival time within an interval
Exptfunc <- function(tall, theta, coeff, t0, rid){
  t0 / coeff$Lambda / theta[rid] + tall[rid]
}


# Defines discrete time interval boundaries from continuous survival times.
# Ensures that a target proportion (rate) of subjects are censored at the end.
findsurvint <- function(y, nper, rate) {
  int <- quantile(y, probs = seq((1 - rate) / nper, 1 - rate, length.out = nper))
  return(int)
}

# --- MAIN DATA GENERATION FUNCTION ---
# Generates a complete simulated survival dataset with fairness scenarios.
# Combines covariate generation (genvar), hazard computation (create_theta),
# and discrete-time survival time generation via inverse hazard sampling.
tvstimegnrt <- function(nsub = 200, 
                        scenario = c("fair", "direct", "proxy", "temporal"), 
                        matsigma = NULL){

  nperiod <- 24
  
  # Generate covariate matrix
  Data <- matrix(NA, nperiod * nsub, 8)
  colnames(Data) <- c("ID","X1","X2","X3","X4","X5","X6","S")
  Data[, 1] <- rep(1:nsub, each = nperiod)
  Data[, 2:8] <- genvar(nsub = nsub, 
                        matsigma = matsigma,
                        scenario = scenario)
  
  # Set the coefficients and compute the Theta = exp(f(X))
  coeffTS <- create_coeff(scenario = scenario, nsub = nsub)
  Coeff <- coeffTS$Coeff
  TS <- coeffTS$TS
  rm(coeffTS)
  
  Theta <- create_theta(data = Data[, 2:8], coeff = Coeff, scenario = scenario)
                       
  Hfunc <- ExpHfunc
  tfunc <- Exptfunc
  
  tlen <- length(TS)
  seqt2 <- nperiod * c(1:(tlen / nperiod))     # Index of the last period
  seqt1 <- nperiod * c(0:((tlen - 1) / nperiod)) + 1       # Index of the first period
  # --- seqt2 and seqt1 are used to compute intervals and then cumulative risk for each intervals ---
  # Subject 1: (t1_1→t1_2), (t1_2→t1_3), (t1_3→t1_4)
  # Subject  2: (t2_1→t2_2), (t2_2→t2_3), (t2_3→t2_4)

  
  # --- Cumulative hazard increments over each interval ---
  # R[i, j] = H(ts_{j+1}) - H(ts_j) for subject i at period j
  R <- Hfunc(ts1 = TS[-seqt1], 
             ts2 = TS[-seqt2], 
             theta = Theta[-seqt2], 
             coeff = Coeff)
  # each row belongs to a subject
  R <- matrix(R, ncol = nperiod - 1, byrow = TRUE)
  
  # --- Inverse hazard sampling: find discrete survival period per subject ---
  # U ~ Uniform(0,1); event time = smallest t such that H(t) >= -log(U)
  U <- runif(nsub)
  survtime <- rep(0, nsub)
  survnrow <- rep(0, nsub)
  # Data[,"ID"] e' rep(1:nsub, each=nperiod) -> blocchi contigui, indice diretto
  # invece di which() (che scansionava l'intero vettore per ogni soggetto)
  for (Count in 1:nsub) {
    base <- (Count - 1L) * nperiod
    idxC <- (base + 1L):(base + nperiod)
    VEC <- c(0, cumsum(R[Count, ]), Inf)
    R.ID <- findInterval(-log(U[Count]), VEC)
    TT <- -log(U[Count]) - VEC[R.ID] 
    survnrow[Count] <- R.ID
    survtime[Count] <- tfunc(tall = TS[idxC], 
                             theta = Theta[idxC], 
                             coeff = Coeff,
                             t0 = TT, 
                             rid = R.ID)
  }
  rm(R)
  rm(Theta)
  rm(U)
  gc()
  RET = list(survtime = survtime,
             coeff = Coeff)
  return(RET)
}
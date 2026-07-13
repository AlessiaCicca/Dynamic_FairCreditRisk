# =============================================================================
# Adapted from:
#   "Dynamic Estimation with Random Forests for Discrete-Time Survival Data" (2021)
# Modifications:
#   - Fixed simulation parameters
#   - Added 4 fairness scenarios (fair, direct, proxy, temporal)
# 
# Mantain nperiod=8, model=linear and distribution=Exp (easiest), censor.rate = 10% (more realistic) 
# and SNR (How strong is the signal compared to the noise) = "low" (to financial domain)
# =============================================================================

create_matsigma <- function(){
  matsigma <- .7 * diag(6) + matrix(.05, 6, 6)
  # to make the first two variables time-independent
  matsigma[1, ] <- c(1, 0, 0, 0, 0, 0)
  matsigma[2, ] <- c(0, 1, 0, 0, 0, 0) 
  return(matsigma)
}

create_matsigma_by_group <- function(S){
  # S=0: coupling inter-feature forte
  matsigma_S0 <- .7 * diag(6) + matrix(.05, 6, 6)
  matsigma_S0[1,] <- c(1,0,0,0,0,0)
  matsigma_S0[2,] <- c(0,1,0,0,0,0)
  
  # S=1: coupling inter-feature debole
  matsigma_S1 <- .6 * diag(6) + matrix(.025, 6, 6)
  matsigma_S1[1,] <- c(1,0,0,0,0,0)
  matsigma_S1[2,] <- c(0,1,0,0,0,0)
  
  return(list(S0 = matsigma_S0, S1 = matsigma_S1))
}


# It's null -> The only source of censoring in the new script is the natural censoring arising from the compressed chngpt (the rate = 0.80 mechanism discussed before). 
# There is no additional administrative censoring imposed on top. 
# setting Censor.time = Inf is not an arbitrary simplification, but rather the natural result of the calibration process:
# the natural censoring induced by findsurvint with rate = 0.80 is already sufficient to achieve the target of 80% censoring, 
# without the need for any additional administrative censoring.

create_ctime <- function(nsub){
  Censor.time <- rep(Inf, nsub)
  return(Censor.time)
}


# FAIR: No discrimination -> BetaS=0 Gamma=0
# DIRECT: Sensitive variable directly influences risk Theta -> BetaS=!0 Gamma=0
# PROXY: Sensitive variable influences covariates Zt -> BetaS=0 Gamma!=0
# TEMPORAL:  Sensitive variable influences covariates Zt and bias increases over time -> BetaS=0 Gamma!=0 (later * t)

create_coeff <- function(nsub, scenario){
  NoiseS <- 1.5
  nperiod <- 24
  Beta1 <- c(1, -1, 1, -1, -0.25, 0.5) 
  Lambda = 0.5
  Alpha = 0
  V = 0
  Beta0 = -5
  TS <- as.vector(replicate(nsub, 
                            c(0, sort(rtrunc(nperiod - 1, spec = "beta", a = 0.0001, b = 1, shape1 = 0.1, shape2 = 2)) * 900)
  ))
  if(scenario == "fair") {
    Gamma=0 
    BetaS=0
  }
  else if(scenario == "direct"){
    Gamma=0
    BetaS=0.8
  }
  else if(scenario == "proxy" | scenario == "temporal"){
    Gamma= 0.5  
    BetaS=0
  }
  else {
      stop("Wrong scenario is specified.")
  }
  Coeff <- list(Lambda = Lambda, Alpha = Alpha, V = V, NoiseS=NoiseS,
                Beta1 = Beta1, Beta0 = Beta0, BetaS=BetaS, Gamma=Gamma)
  return(list(TS = TS, Coeff = Coeff))
}

# =============================================================================
# Adapted from:
#   "Dynamic Estimation with Random Forests for Discrete-Time Survival Data" (2021)
# Modifications:
#   - Fixed simulation parameters
#   - Added 4 fairness scenarios (fair, direct, proxy, temporal)
# =============================================================================


traindtv_autocorr_gnrt <- function(nsub = 200, 
                                   matsigma = NULL,
                                   scenario = c("fair", "direct", "proxy", "temporal")){
  
  nstime <- 1000
  nperiod <- 24


  # chngpt defines the 12 discrete time interval boundaries always computed on "fair" to
  # ensure breakpoints are stable and comparable
  RET <- tvstimegnrt(nsub = nstime,
                     scenario = "fair",      
                     matsigma = matsigma)
  Coeff <- create_coeff(scenario = scenario, nsub = nsub)$Coeff
  


  # In the credit risk context, survival corresponds to a borrower not defaulting on their loan. Since default is a rare event in real-world credit portfolios,
  # a high censoring rate of 80% is more realistic, meaning that the majority of subjects will not experience the event within the observation window. 
  chngpt <- findsurvint(y = sort(RET$survtime),
                        nper = nperiod,
                        rate = 0.80)
  rm(RET)
  gc()

  # --- Initialize data matrix ---
  Data <- as.data.frame(matrix(NA, nperiod * nsub, 14))
  names(Data) <- c("ID", "X1", "X2", "X3", "X4", "X5", "X6", "S",
                   "StopT", "Time", "Event", "Theta", "Su", "h")
  
  Data$Time <- rep(1:nperiod, nsub)
  Data$ID <- rep(1:nsub, each = nperiod)

  # Generate covariates X1-X6 and sensitive attribute S
  Data[, 2:8] <- genvar(nsub = nsub, 
                        matsigma = matsigma,
                        scenario = scenario)
  Data$StopT <- rep(chngpt, nsub)
  
  # Compute hazard rate theta = exp(f(X, S)) for each subject-period 
  Data$Theta <- create_theta(data = as.matrix(Data[, 2:8]), 
                             coeff = Coeff, 
                             scenario = scenario)
  Hfunc <-  ExpHfunc
  tfunc <- Exptfunc

  # Build interval sequences
  TS <- as.vector(rep(c(0, chngpt), nsub))
  tlen <- length(TS)
  seqt2 <- (nperiod + 1) * c(1:(tlen / (nperiod + 1))) # each : -nperiod
  seqt1 <- (nperiod + 1) * c(0:((tlen - 1) / (nperiod + 1))) + 1 # each : -1
  
  #  Cumulative hazard increments
  R <- Hfunc(ts1 = TS[-seqt1], 
             ts2 = TS[-seqt2], 
             theta = Data$Theta, 
             coeff = Coeff)
  TS <- TS[-seqt2]
  rm(seqt2,seqt1)

  # R: [nsub x nperiod] — each row is a subject, each column is a period
  R <- matrix(R, ncol = nperiod, byrow = TRUE)
  
  # Cumulative sum across periods: [nperiod x nsub]
  R <- apply(R, 1, cumsum)
  
  # --- Survival probability and hazard at each period ---
  # Su[i, t]  = exp(-H(t)) = P(T > t)
  # h[i, t]   = P(event at t | survived to t) = (S(t-1) - S(t)) / S(t-1)
  survprob <- exp(-R)
  Data$h <- as.vector(sapply(1:nsub, function(ni) (c(1, survprob[, ni][-nperiod]) - survprob[, ni]) / c(1, survprob[, ni][-nperiod])))
  Data$Su <- as.vector(survprob)
  rm(survprob)

  # Inverse hazard sampling: assign discrete event time per subject
  U <- runif(nsub)
  
  TS <- as.vector(rep(c(0, chngpt[1:(nperiod - 1)]), nsub))
  # Data$ID e' rep(1:nsub, each = nperiod) -> blocchi contigui, quindi l'indice
  # del soggetto Count si calcola direttamente, niente bisogno di which() (che
  # scansionava l'intero vettore ID per OGNI soggetto: O(nperiod * nsub^2),
  # diventava proibitivo per nsub grandi). Stesso risultato, molto piu' veloce.
  for (Count in 1:nsub) {
    base  <- (Count - 1L) * nperiod
    idxC  <- (base + 1L):(base + nperiod)
    VEC <- c(0, R[, Count], Inf)
    rID <- findInterval(-log(U[Count]), VEC)
    if (rID == 1){
      # event in first period
      Data$Event[idxC[1]] <- 1
    } else if (rID <= nperiod){
      # survived up to rID-1, event at rID
      Data$Event[idxC[1:(rID - 1)]] <- 0
      Data$Event[idxC[rID]] <- 1
    } else {
      # survived all periods → censored
      Data$Event[idxC] <- 0
    }
  }
  
                             
  Data <- Data[!is.na(Data$Event), ]
               
  if (length(unique(Data$ID)) != nsub){
    stop("ID length NOT equal to nsub")
  }
  
  
  Data$I <- 1:nrow(Data)
  Data$Theta <- NULL
  RET <- NULL
  RET$fullData <- Data
  RET$Info = list(DRate = sum(Data$Event) / nsub, Coeff = Coeff)
  
  rm(Data)
  gc()
  return(RET)
}
# =============================================================================
# Adapted from:
#   "Dynamic Estimation with Random Forests for Discrete-Time Survival Data" (2021)
# Modifications:
#   - Fixed simulation parameters
#   - Added 4 fairness scenarios (fair, direct, proxy, temporal)
# =============================================================================


testdtv_gnrt = function(data, ntest, id = NULL, period = NULL, y = NULL){

  # Generates the test data sets, one for each t (given T > t)

  maxt <- max(data[, period])    # number of periods
  allid <- unique(data[, id])    # unique id from the data
  ind <- sample(allid, ntest)    # sample of subject (their id)
  
  inddata <- which(data[, id] %in% ind)
  
  out <- list(data[inddata, ])
  
  for (j in 2:maxt) {
    tempdata <- data[data[, period] == j, ]
    indc <- tempdata[, id]
    # number of subjects in the sample provided (must be large enough to make sure that we will
    # have at least ntest subjects for each j=1, ..., maxt
    if (length(unique(indc)) < ntest) stop(sprintf("Number of subjects in %1.0f-th set from the provided sample: %1.0f < ntest!", 
                                                   j, length(unique(indc))))
    
    ind <- sample(indc, ntest)   
    inddata <- which(data[, id] %in% ind)
    out[[j]] <- data[inddata, ]
  }
  out
}

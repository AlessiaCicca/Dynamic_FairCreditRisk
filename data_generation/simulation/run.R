# Main script to generate simulated survival data under 4 fairness scenarios.

install.packages("openxlsx")
install.packages("truncdist")
library(openxlsx)
library(truncdist)

# --- Source all required scripts ---
source("utils.R")                    # utility functions
source("genvar.R")                   # covariate generation (VAR process)
source("timevarying_gnrt.R")         # continuous survival time generation
source("traindtv_autocorr_gnrt.R")   # discrete-time training data generation

# --- Simulation setup ---
matsigma  <- create_matsigma()
scenarios <- c("fair", "direct", "proxy", "temporal")
wb <- createWorkbook()

# --- Output folder with timestamp ---
run_folder <- paste0("run_", format(
    as.POSIXct(Sys.time(), tz="Europe/Rome"), 
    "%Y%m%d_%H%M%S"
))

dir.create(run_folder)
cat("Output folder:", run_folder, "\n")

for (sc in scenarios) {
  result <- traindtv_autocorr_gnrt(nsub = 24000, matsigma = matsigma, scenario = sc)
  
  # --- Compute event counts and percentages by group S ---
  df        <- result$fullData
  df_sorted <- df[order(df$ID, df$Time), ]
  df_id     <- df_sorted[!duplicated(df_sorted$ID, fromLast = TRUE), ]
  counts    <- table(df_id$S, df_id$Event)
  percent   <- prop.table(counts, margin = 1) * 100

  df_counts  <- as.data.frame.matrix(counts)
  df_percent <- as.data.frame.matrix(round(percent, 2))
  df_counts$S  <- rownames(df_counts);  df_counts  <- df_counts[, c("S", setdiff(names(df_counts), "S"))]
  df_percent$S <- rownames(df_percent); df_percent <- df_percent[, c("S", setdiff(names(df_percent), "S"))] 

  df_info <- data.frame(Metric = "Death Rate", Value = result$Info$DRate)



  # --- Coefficients used in generation ---
  coeff_list <- result$Info$Coeff
  df_coeff <- do.call(rbind, lapply(names(coeff_list), function(nm) {
    vals <- as.vector(coeff_list[[nm]])
    data.frame(
      Coefficient = if (length(vals) == 1) nm else paste0(nm, "_", seq_along(vals)),
      Value       = vals,
      stringsAsFactors = FALSE
    )
  }))
  
  # ---- EXCEL ----
  addWorksheet(wb, sheetName = sc)
  writeData(wb, sc, "SCENARIO INFO",        startRow = 1,  startCol = 1)
  writeData(wb, sc, df_info,                startRow = 2,  startCol = 1)
  writeData(wb, sc, "EVENT COUNTS",         startRow = 6,  startCol = 1)
  writeData(wb, sc, df_counts,              startRow = 7,  startCol = 1)
  row_pct <- 7 + nrow(df_counts) + 2
  writeData(wb, sc, "EVENT PERCENTAGES (%)", startRow = row_pct,     startCol = 1)
  writeData(wb, sc, df_percent,              startRow = row_pct + 1, startCol = 1)
  row_coeff <- row_pct + nrow(df_percent) + 3
  writeData(wb, sc, "COEFFICIENTS",         startRow = row_coeff,     startCol = 1)
  writeData(wb, sc, df_coeff,               startRow = row_coeff + 1, startCol = 1)

  data_file <- file.path(run_folder, paste0("data_", sc, ".csv"))
  write.csv(result$fullData, file = data_file, row.names = FALSE)

  train_sheet <- paste0("train_", sc)
  addWorksheet(wb, train_sheet)
  writeData(wb, train_sheet, result$fullData)
  
}


excel_file <- file.path(run_folder, "simulation_results.xlsx")
saveWorkbook(wb, excel_file, overwrite = TRUE)
cat("Data generation process completed!")

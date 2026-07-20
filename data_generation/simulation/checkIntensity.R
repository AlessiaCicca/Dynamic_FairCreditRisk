# =============================================================================
# check_intensity.R
#
# Confronta l'intensita' di discriminazione iniettata tra due run della
# simulazione (tipicamente LOW e HIGH), scenario per scenario.
#
# Uso:
#   Rscript check_intensity.R <cartella_low> <cartella_high> [cartella_output]
#
# Esempio:
#   Rscript check_intensity.R run_20260720_121240 run_20260720_122308
#
# Produce:
#   - un report a schermo con, per ogni scenario:
#       * tasso di evento per gruppo S e gap fra i gruppi
#       * coefficiente di S in un modello di Cox (stima di BetaS)
#       * differenze di media delle covariate fra gruppi (stima di Gamma)
#       * gap di hazard nel primo e nell'ultimo periodo (proxy vs temporal)
#   - intensity_check.csv   tutte le quantita' in forma tabellare
#   - hazard_gap.csv        gap di hazard per periodo, tutti gli scenari
#   - km_<scenario>.png     curve di Kaplan-Meier per gruppo, low vs high
#   - hazgap_<scenario>.png gap di hazard per periodo, low vs high
# =============================================================================

suppressPackageStartupMessages({
  if (!requireNamespace("survival", quietly = TRUE)) install.packages("survival")
  library(survival)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("Uso: Rscript check_intensity.R <cartella_low> <cartella_high> [cartella_output]")
}
dir_low  <- args[1]
dir_high <- args[2]
out_dir  <- if (length(args) >= 3) args[3] else "."
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

SCENARIOS <- c("fair", "direct", "proxy", "temporal")
COVS      <- paste0("X", 1:6)

# ---------------------------------------------------------------------------
# Carica un CSV e riduce a una riga per soggetto (ultimo periodo osservato)
# ---------------------------------------------------------------------------
load_scenario <- function(dir_path, scenario) {
  f <- file.path(dir_path, paste0("data_", scenario, ".csv"))
  if (!file.exists(f)) {
    warning("File mancante: ", f)
    return(NULL)
  }
  df <- read.csv(f)
  df <- df[order(df$ID, df$Time), ]
  df_id <- df[!duplicated(df$ID, fromLast = TRUE), ]
  list(panel = df, subj = df_id)
}

# ---------------------------------------------------------------------------
# 1. Tasso di evento per gruppo
# ---------------------------------------------------------------------------
event_rates <- function(subj) {
  tab <- table(subj$S, subj$Event)
  if (ncol(tab) < 2 || nrow(tab) < 2) return(c(r0 = NA, r1 = NA, gap = NA))
  pct <- prop.table(tab, margin = 1) * 100
  r0 <- pct["0", "1"]; r1 <- pct["1", "1"]
  c(r0 = as.numeric(r0), r1 = as.numeric(r1), gap = as.numeric(r1 - r0))
}

# ---------------------------------------------------------------------------
# 2. Coefficiente di S in un Cox con tutte le covariate.
#    Per DIRECT stima BetaS. Per PROXY/TEMPORAL e' atteso ~0, perche' S non
#    entra nell'hazard: agisce spostando le covariate (vedi punto 3).
# ---------------------------------------------------------------------------
cox_coef_S <- function(subj) {
  present <- COVS[COVS %in% names(subj)]
  fml <- as.formula(paste("Surv(Time, Event) ~", paste(c(present, "S"), collapse = " + ")))
  fit <- try(coxph(fml, data = subj), silent = TRUE)
  if (inherits(fit, "try-error")) return(c(coef = NA, se = NA, hr = NA))
  cf <- coef(fit)["S"]
  se <- sqrt(diag(vcov(fit)))["S"]
  c(coef = as.numeric(cf), se = as.numeric(se), hr = as.numeric(exp(cf)))
}

# ---------------------------------------------------------------------------
# 3. Differenza di media delle covariate fra i due gruppi (sul panel completo).
#    E' la via diretta per verificare Gamma in PROXY/TEMPORAL.
# ---------------------------------------------------------------------------
cov_mean_gaps <- function(panel) {
  present <- COVS[COVS %in% names(panel)]
  sapply(present, function(v) {
    mean(panel[[v]][panel$S == 1], na.rm = TRUE) -
      mean(panel[[v]][panel$S == 0], na.rm = TRUE)
  })
}

# ---------------------------------------------------------------------------
# 4. Hazard discreto empirico: tasso di evento per periodo e per gruppo.
#    Serve a distinguere PROXY da TEMPORAL: le curve di Kaplan-Meier non
#    bastano, perche' divergono comunque nel tempo anche quando l'hazard
#    ratio e' costante. Il gap per periodo, invece, e' piatto in PROXY e
#    crescente in TEMPORAL.
# ---------------------------------------------------------------------------
haz_by_period <- function(panel) {
  if (!all(c("Time", "S", "Event") %in% names(panel))) return(NULL)
  agg <- aggregate(Event ~ Time + S, data = panel, FUN = mean)
  w   <- reshape(agg, idvar = "Time", timevar = "S", direction = "wide")
  w   <- w[order(w$Time), ]
  if (!all(c("Event.0", "Event.1") %in% names(w))) return(NULL)
  w$gap <- w$Event.1 - w$Event.0
  w[, c("Time", "Event.0", "Event.1", "gap")]
}

# ---------------------------------------------------------------------------
# Analisi di un singolo scenario in una singola run
# ---------------------------------------------------------------------------
analyse <- function(dir_path, scenario, label) {
  d <- load_scenario(dir_path, scenario)
  if (is.null(d)) return(NULL)
  er <- event_rates(d$subj)
  cx <- cox_coef_S(d$subj)
  gp <- cov_mean_gaps(d$panel)
  hz <- haz_by_period(d$panel)
  list(
    row = data.frame(
      scenario      = scenario,
      intensity     = label,
      n_subjects    = nrow(d$subj),
      event_rate_S0 = round(er["r0"], 3),
      event_rate_S1 = round(er["r1"], 3),
      event_gap     = round(er["gap"], 3),
      cox_coef_S    = round(cx["coef"], 4),
      cox_se_S      = round(cx["se"], 4),
      hazard_ratio  = round(cx["hr"], 4),
      hazgap_first  = if (!is.null(hz)) round(head(hz$gap, 1), 4) else NA,
      hazgap_last   = if (!is.null(hz)) round(tail(hz$gap, 1), 4) else NA,
      t(round(gp, 4)),
      row.names = NULL, check.names = FALSE
    ),
    subj = d$subj, gaps = gp, haz = hz
  )
}

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
cat("\n", strrep("=", 78), "\n", sep = "")
cat("  CONTROLLO INTENSITA' DELLA DISCRIMINAZIONE\n")
cat("  LOW :", dir_low, "\n")
cat("  HIGH:", dir_high, "\n")
cat(strrep("=", 78), "\n", sep = "")

all_rows <- list()
all_haz  <- list()

for (sc in SCENARIOS) {
  lo <- analyse(dir_low,  sc, "low")
  hi <- analyse(dir_high, sc, "high")
  if (is.null(lo) || is.null(hi)) next

  all_rows[[length(all_rows) + 1]] <- lo$row
  all_rows[[length(all_rows) + 1]] <- hi$row

  cat("\n", strrep("-", 78), "\n", sep = "")
  cat("SCENARIO: ", toupper(sc), "\n", sep = "")
  cat(strrep("-", 78), "\n", sep = "")

  cat(sprintf("%-28s %12s %12s %12s\n", "", "LOW", "HIGH", "rapporto"))

  f <- function(name, a, b) {
    ratio <- if (!is.na(a) && abs(a) > 1e-8) sprintf("%.2fx", b / a) else "--"
    cat(sprintf("%-28s %12.4f %12.4f %12s\n", name, a, b, ratio))
  }

  f("event rate S=0 (%)", lo$row$event_rate_S0, hi$row$event_rate_S0)
  f("event rate S=1 (%)", lo$row$event_rate_S1, hi$row$event_rate_S1)
  f("event gap (S1-S0, %)", lo$row$event_gap,   hi$row$event_gap)
  f("Cox coef S", lo$row$cox_coef_S, hi$row$cox_coef_S)
  f("hazard ratio S", lo$row$hazard_ratio, hi$row$hazard_ratio)

  cat("\n  differenze di media covariate (S=1 meno S=0):\n")
  for (v in names(lo$gaps)) f(paste0("  ", v), lo$gaps[[v]], hi$gaps[[v]])

  # --- gap di hazard per periodo: primo, ultimo e rapporto ultimo/primo ---
  if (!is.null(lo$haz) && !is.null(hi$haz)) {
    all_haz[[length(all_haz) + 1]] <- cbind(scenario = sc, intensity = "low",  lo$haz)
    all_haz[[length(all_haz) + 1]] <- cbind(scenario = sc, intensity = "high", hi$haz)

    cat("\n  gap di hazard per periodo (S=1 meno S=0):\n")
    f("  primo periodo", head(lo$haz$gap, 1), head(hi$haz$gap, 1))
    f("  ultimo periodo", tail(lo$haz$gap, 1), tail(hi$haz$gap, 1))

    trend <- function(h) {
      n <- nrow(h); k <- max(1, floor(n / 4))
      early <- mean(head(h$gap, k), na.rm = TRUE)
      late  <- mean(tail(h$gap, k), na.rm = TRUE)
      if (abs(early) < 1e-8) return(NA)
      late / early
    }
    cat(sprintf("%-28s %12s %12.2f %12.2f\n",
                "  rapporto tardi/presto", "", trend(lo$haz), trend(hi$haz)))
  }

  # interpretazione minima, per non dover ricordare cosa aspettarsi
  if (sc == "fair") {
    cat("\n  atteso: tutti i valori vicini a zero in entrambe le run.\n")
  } else if (sc == "direct") {
    cat("\n  atteso: Cox coef S positivo, circa 2x in HIGH rispetto a LOW.\n")
  } else if (sc == "proxy") {
    cat("\n  atteso: Cox coef S vicino a zero; differenze di media su X2, X4, X6\n")
    cat("          circa 2x in HIGH; gap di hazard sostanzialmente PIATTO nel\n")
    cat("          tempo (rapporto tardi/presto vicino a 1).\n")
  } else {
    cat("\n  atteso: Cox coef S vicino a zero; differenze di media su X2, X4, X6\n")
    cat("          circa 2x in HIGH; gap di hazard CRESCENTE nel tempo\n")
    cat("          (rapporto tardi/presto nettamente maggiore di 1, e maggiore\n")
    cat("          di quello osservato in PROXY).\n")
  }

  # --- Kaplan-Meier: due pannelli, low e high ---
  png(file.path(out_dir, paste0("km_", sc, ".png")), width = 1000, height = 450)
  par(mfrow = c(1, 2))
  for (p in list(list(d = lo$subj, t = paste0(sc, " - low")),
                 list(d = hi$subj, t = paste0(sc, " - high")))) {
    fit <- survfit(Surv(Time, Event) ~ S, data = p$d)
    plot(fit, col = c("black", "red"), lwd = 2,
         xlab = "Time", ylab = "Survival", main = p$t)
    legend("bottomleft", legend = c("S=0", "S=1"), col = c("black", "red"),
           lwd = 2, bty = "n")
  }
  dev.off()

  # --- gap di hazard nel tempo: piatto in proxy, crescente in temporal ---
  if (!is.null(lo$haz) && !is.null(hi$haz)) {
    png(file.path(out_dir, paste0("hazgap_", sc, ".png")), width = 700, height = 450)
    yl <- range(c(lo$haz$gap, hi$haz$gap), na.rm = TRUE)
    plot(lo$haz$Time, lo$haz$gap, type = "b", pch = 16, col = "steelblue",
         ylim = yl, xlab = "Time", ylab = "hazard gap (S=1 - S=0)",
         main = paste0(sc, ": event-rate gap per period"))
    lines(hi$haz$Time, hi$haz$gap, type = "b", pch = 17, col = "firebrick")
    abline(h = 0, lty = 2)
    legend("topleft", legend = c("low", "high"),
           col = c("steelblue", "firebrick"), pch = c(16, 17), bty = "n")
    dev.off()
  }
}

if (length(all_rows) > 0) {
  res <- do.call(rbind, all_rows)
  out_csv <- file.path(out_dir, "intensity_check.csv")
  write.csv(res, out_csv, row.names = FALSE)

  cat("\n", strrep("=", 78), "\n", sep = "")
  cat("Tabella salvata in: ", out_csv, "\n", sep = "")

  if (length(all_haz) > 0) {
    hz <- do.call(rbind, all_haz)
    hz_csv <- file.path(out_dir, "hazard_gap.csv")
    write.csv(hz, hz_csv, row.names = FALSE)
    cat("Gap di hazard per periodo in: ", hz_csv, "\n", sep = "")
  }
  cat("Grafici salvati in: ", out_dir, "/km_<scenario>.png e hazgap_<scenario>.png\n", sep = "")
} else {
  cat("\nNessuno scenario trovato nelle cartelle indicate.\n")
}
cat(strrep("=", 78), "\n\n", sep = "")
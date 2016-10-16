library(rEDM)
library(jsonlite)
library(multispatialCCM)

showdata <- function(Accm, Bccm) {
  plot(Accm, type="l", col=1, lwd=2, xlim=c(0, 212), ylim=c(0,1), xlab="time step", ylab="Normalized Value", cex.lab = 1.5)
  lines(Bccm, type="l", col=2, lty=2, lwd=2, cex.lab = 1.5)
  legend("topleft", c("Salinity", "Temprature"), cex=1.5, lty=c(1,2), col=c(1,2), lwd=2, bty="n")
}

drawCCM <- function(Accm, Bccm, Em, TAU) {
  Accm_Bccm <- data.frame(Accm=Accm, Bccm=Bccm)
  Accm_xmap_Bccm <- ccm(Accm_Bccm, E = Em, lib_column = "Accm", tau = TAU,
                        target_column = "Bccm", lib_sizes = seq(10, 200, by = 10), random_libs = TRUE)
  Bccm_xmap_Accm <- ccm(Accm_Bccm, E = Em, lib_column = "Bccm", tau = TAU,
                        target_column = "Accm", lib_sizes = seq(10, 200, by = 10), random_libs = TRUE)
  Accm_xmap_Bccm_means <- ccm_means(Accm_xmap_Bccm)
  Bccm_xmap_Accm_means <- ccm_means(Bccm_xmap_Accm)
  par(mar = c(4, 4, 1, 1), mgp = c(2.5, 1, 0))
  plot(Accm_xmap_Bccm_means$lib_size, pmax(0, Accm_xmap_Bccm_means$rho), type = "l", col = "red", xlab = "Library Size", ylab = "Cross Map Skill (rho)", ylim = c(0, 1))
  lines(Bccm_xmap_Accm_means$lib_size, pmax(0, Bccm_xmap_Accm_means$rho), col = "blue")
  legend(x = "topleft", legend = c("Accm xmap Bccm", "Bccm xmap Accm"), col = c("red", "blue"), lwd = 1, inset = 0.02, cex = 0.8)
}

rainfall_csv <- read.csv("../csv/kousuiryou_revised.csv", header=FALSE)
rainfall <- rainfall_csv$V2
plot(rainfall, type = "l")

# Embedding Dimension
lib <- c(1, 90)
pred <- c(95, 110)  
simplex_output <- simplex(rainfall, lib, pred)
plot(simplex_output$E, simplex_output$rho, ylim=c(0,1), type = "l", xlab = "Embedding Dimension (E)", ylab = "Forecast Skill (rho)")
E <- 4

# Prediction Decay
simplex_output <- simplex(rainfall, lib, pred, E = E, tp = 1:10)
par(mar = c(4, 4, 1, 1))
plot(simplex_output$tp, simplex_output$rho, ylim=c(0,1), type = "l", xlab = "Time to Prediction (tp)", ylab = "Forecast Skill (rho)")
TAU = 1

# Identifying Nonlinearity
smap_output <- s_map(rainfall, lib, pred, E=E)
par(mar = c(4, 4, 1, 1), mgp = c(2.5, 1, 0))
plot(smap_output$theta, smap_output$rho, type = "l", xlab = "Nonlinearity (theta)", ylab = "Forecast Skill (rho)")


huusoku_csv <- read.csv("../csv/huusoku_revised.csv", header=FALSE)
huusoku <- huusoku_csv$V2
plot(huusoku, type = "l")

# Embedding Dimension
lib <- c(1, 90)
pred <- c(95, 110)  
simplex_output <- simplex(huusoku, lib, pred)
plot(simplex_output$E, simplex_output$rho, ylim=c(0,1), type = "l", xlab = "Embedding Dimension (E)", ylab = "Forecast Skill (rho)")
E <- 1

# Prediction Decay
simplex_output <- simplex(huusoku, lib, pred, E = E, tp = 1:10)
par(mar = c(4, 4, 1, 1))
plot(simplex_output$tp, simplex_output$rho, ylim=c(0,1), type = "l", xlab = "Time to Prediction (tp)", ylab = "Forecast Skill (rho)")
TAU = 1

# Identifying Nonlinearity
smap_output <- s_map(huusoku, lib, pred, E=E)
par(mar = c(4, 4, 1, 1), mgp = c(2.5, 1, 0))
plot(smap_output$theta, smap_output$rho, type = "l", xlab = "Nonlinearity (theta)", ylab = "Forecast Skill (rho)")

library(rEDM)
library(jsonlite)
library(multispatialCCM)

showdata <- function(Accm, Bccm) {
  plot(Accm, type="l", col=1, lwd=2, xlim=c(0, 212), ylim=c(0,1), xlab="time step", ylab="Normalized Value", cex.lab = 1.5)
  lines(Bccm, type="l", col=2, lty=2, lwd=2, cex.lab = 1.5)
  legend("topleft", c("Salinity", "Temprature"), cex=1.5, lty=c(1,2), col=c(1,2), lwd=2, bty="n")
}

determineEmbeddingDimension <- function(data) {
  lib <- c(1, 50)
  pred <- c(90, 212)  
  simplex_output <- simplex(data, lib, pred)
  plot(simplex_output$E, simplex_output$rho, type = "l", xlab = "Embedding Dimension (E)", ylab = "Forecast Skill (rho)")
  
  max_index <- 1
  max <- simplex_output$rho[1]
  for (i in 2:length(simplex_output$rho)) {
    if (simplex_output$rho[i] > max) {
      max_index <- i
      max <- simplex_output$rho[i]
    }  
  }
  return(max_index)
}

predictionDeacy <- function(data, Em) {
  lib <- c(1, 50)
  pred <- c(90, 212)  
  simplex_output <- simplex(data, lib, pred, E = Em, tp = 1:10)
  par(mar = c(4, 4, 1, 1))
  plot(simplex_output$tp, simplex_output$rho, type = "l", xlab = "Time to Prediction (tp)", ylab = "Forecast Skill (rho)")
}

identifyingNonlinearity <- function(data, Em) {
  lib <- c(1, 50)
  pred <- c(90, 212)  
  smap_output <- s_map(data, lib, pred, E=Em)
  par(mar = c(4, 4, 1, 1), mgp = c(2.5, 1, 0))
  plot(smap_output$theta, smap_output$rho, type = "l", xlab = "Nonlinearity (theta)", ylab = "Forecast Skill (rho)")
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

data = fromJSON("../python/data.json")

index <- 3000
Accm <- as.numeric(unlist(data$data[index,]$s))
Bccm <- as.numeric(unlist(data$data[index,]$t))

# show data
showdata(Accm, Bccm)
# determine Embedding Dimension
determineEmbeddingDimension(Accm)
determineEmbeddingDimension(Bccm)
E <- 2
# Prediction Decay
predictionDeacy(data = Accm, Em = E)
predictionDeacy(data = Bccm, Em = E)
TAU = 1
# Identifying Nonlinearity
identifyingNonlinearity(data = Accm, Em = E)
identifyingNonlinearity(data = Bccm, Em = E)
# draw CCM
drawCCM(Accm = Accm, Bccm = Bccm, E = E, TAU = TAU)

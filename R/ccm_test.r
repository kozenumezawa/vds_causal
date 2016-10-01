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
  smap_output <- s_map(data, lib, pred, E=E_A)
  par(mar = c(4, 4, 1, 1), mgp = c(2.5, 1, 0))
  plot(smap_output$theta, smap_output$rho, type = "l", xlab = "Nonlinearity (theta)", ylab = "Forecast Skill (rho)")
}

drawCCM <- function(Accm, Bccm, E_A, E_B, TAU) {
  Accm_Bccm <- data.frame(Accm=Accm, Bccm=Bccm)
  Accm_xmap_Bccm <- ccm(Accm_Bccm, E = E_A, lib_column = "Accm", tau = TAU,
                        target_column = "Bccm", lib_sizes = seq(10, 200, by = 10), random_libs = TRUE)
  Bccm_xmap_Accm <- ccm(Accm_Bccm, E = E_B, lib_column = "Bccm", tau = TAU,
                        target_column = "Accm", lib_sizes = seq(10, 200, by = 10), random_libs = TRUE)
  Accm_xmap_Bccm_means <- ccm_means(Accm_xmap_Bccm)
  Bccm_xmap_Accm_means <- ccm_means(Bccm_xmap_Accm)
  par(mar = c(4, 4, 1, 1), mgp = c(2.5, 1, 0))
  plot(Accm_xmap_Bccm_means$lib_size, pmax(0, Accm_xmap_Bccm_means$rho), type = "l", col = "red", xlab = "Library Size", ylab = "Cross Map Skill (rho)", ylim = c(0, 1))
  lines(Bccm_xmap_Accm_means$lib_size, pmax(0, Bccm_xmap_Accm_means$rho), col = "blue")
  legend(x = "topleft", legend = c("Accm xmap Bccm", "Bccm xmap Accm"), col = c("red", "blue"), lwd = 1, inset = 0.02, cex = 0.8)
}

data = fromJSON("../python/data.json")

index <- 1
Accm <- as.numeric(unlist(data$data[index,]$s))
Bccm <- as.numeric(unlist(data$data[index,]$t))

# show data
showdata(Accm, Bccm)
# determine Embedding Dimension
E_A <- determineEmbeddingDimension(Accm)
E_B <- determineEmbeddingDimension(Bccm)
# Prediction Decay
predictionDeacy(data = Accm, Em = E_A)
predictionDeacy(data = Bccm, Em = E_B)
TAU = 1
# Identifying Nonlinearity
identifyingNonlinearity(data = Accm, Em = E_A)
identifyingNonlinearity(data = Bccm, Em = E_B)
# draw CCM
drawCCM(Accm = Accm, Bccm = Bccm, E_A = E_A, E_B = E_B, TAU = TAU)

#---twin surrogate---
Heaviside <- function(x) {
  if (x > 0) {
    return (1)
  }
  return (0)
}
# create trajectory vector form attractor
X_DIM <- E_A
BACK_MAX <- (X_DIM - 1) * TAU
X_N <- length(Accm) - BACK_MAX # length of x
x <- array(0, dim=c(X_N, X_DIM))
for (t in 1 : X_N) {
  for(j in 1:X_DIM) {
    x[t,j] <- Accm[(t + BACK_MAX) - (j - 1) * TAU]
  }
}

# cauculate reccurrence matrix
delta <- 0.125
R <- array(0, dim=c(X_N, X_N))
for (i in 1 : X_N) {
  for(j in 1 : X_N) {
    R[i,j] <- Heaviside(delta - max(abs(x[i,] - x[j,])))
  }
}

# create twin surrogates datacheckHavingTwin <- function(i, R) {
  N <- length(R[1,])
  for(j in 1 : N) {
    twin_flag <- TRUE
    for (k in 1 : N) {
      if (R[i,k] != R[j,k]) {
        twin_flag <- FALSE
      }
    }
    if (twin_flag == TRUE) {
      return (TRUE)
    }
  }
  return (FALSE)
}


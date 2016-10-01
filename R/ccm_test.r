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

drawCCM <- function(CCM_boot_A, CCM_boot_B) {
  plotxlimits<-range(c(CCM_boot_A$Lobs, CCM_boot_B$Lobs))
  plot(CCM_boot_A$Lobs, CCM_boot_A$rho, type="l", col=1, lwd=2, xlim=c(plotxlimits[1], plotxlimits[2]), ylim=c(0,1), xlab="Library Size", ylab="Cross Map Skill (rho)", cex.lab = 1.5)
  lines(CCM_boot_B$Lobs, CCM_boot_B$rho, type="l", col=2, lty=2, lwd=2)
  legend("topleft", c("Salinity causes Temprature", "Temprature causes Salinituy"), lty=c(1,2), col=c(1,2), lwd=2, bty="n", cex=1.2)
}

data = fromJSON("../python/data.json")

index <- 1
Accm <- as.numeric(unlist(data$data[index,]$s))
Bccm <- as.numeric(unlist(data$data[index,]$t))

# show data
showdata(Accm, Bccm)
# determine Embedding Dimension
determineEmbeddingDimension(Accm)
E_A = 2
determineEmbeddingDimension(Bccm)
E_B = 5
# Prediction Decay
predictionDeacy(data = Accm, Em = E_A)
predictionDeacy(data = Bccm, Em = E_B)
TAU = 1
# Identifying Nonlinearity
identifyingNonlinearity(data = Accm, Em = E_A)
identifyingNonlinearity(data = Bccm, Em = E_B)
# CCM(use multispatialCCM)
signal_A_out<-SSR_check_signal(A=Accm, E=E_A, tau=TAU, predsteplist=1:10)
signal_B_out<-SSR_check_signal(A=Bccm, E=E_B, tau=TAU, predsteplist=1:10)
CCM_boot_A<-CCM_boot(Accm, Bccm, E_A, tau=TAU, iterations=30)
CCM_boot_B<-CCM_boot(Bccm, Accm, E_B, tau=TAU, iterations=30)
(CCM_significance_test<-ccmtest(CCM_boot_A, CCM_boot_B))
drawCCM(CCM_boot_A = CCM_boot_A, CCM_boot_B = CCM_boot_B)

# create data frame from datasets
Accm_Bccm <- data.frame(Accm=Accm, Bccm=Bccm)
Accm_xmap_Bccm <- ccm(Accm_Bccm, E = E_A, lib_column = "Accm", tau = TAU,
                        target_column = "Bccm", lib_sizes = seq(10, 200, by = 10), random_libs = FALSE)
Bccm_xmap_Accm <- ccm(Accm_Bccm, E = E_B, lib_column = "Bccm", tau = TAU,
                      target_column = "Accm", lib_sizes = seq(10, 200, by = 10), random_libs = FALSE)

Accm_xmap_Bccm_means <- ccm_means(Accm_xmap_Bccm)
Bccm_xmap_Accm_means <- ccm_means(Bccm_xmap_Accm)
par(mar = c(4, 4, 1, 1), mgp = c(2.5, 1, 0))
plot(Accm_xmap_Bccm_means$lib_size, pmax(0, Accm_xmap_Bccm_means$rho), type = "l", col = "red", 
     xlab = "Library Size", ylab = "Cross Map Skill (rho)", ylim = c(0, 1))
lines(Bccm_xmap_Accm_means$lib_size, pmax(0, Bccm_xmap_Accm_means$rho), col = "blue")
legend(x = "topleft", legend = c("Accm xmap Bccm", "Bccm xmap Accm"), col = c("red", 
                                                                                  "blue"), lwd = 1, inset = 0.02, cex = 0.8)


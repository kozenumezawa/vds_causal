library(rEDM)
library(jsonlite)
library(multispatialCCM)

data = fromJSON("data.json")


index <- 9911 # probably have causality
Accm <- as.numeric(unlist(data$data[index,]$s))
Bccm <- as.numeric(unlist(data$data[index,]$t))

# show data
plot(Accm, type="l", col=1, lwd=2, xlim=c(0, 212), ylim=c(0,1), xlab="time step", ylab="Normalized Value", cex.lab = 1.5)
lines(Bccm, type="l", col=2, lty=2, lwd=2, cex.lab = 1.5)
legend("topleft", c("Salinity", "Flow velocity"), cex=1.5, lty=c(1,2), col=c(1,2), lwd=2, bty="n")

# determine Embedding Dimension
lib <- c(1, 50)
pred <- c(90, 212)
simplex_output <- simplex(Accm, lib, pred)
par(mar = c(4, 4, 1, 1), mgp = c(2.5, 1, 0))
plot(simplex_output$E, simplex_output$rho, type = "l", xlab = "Embedding Dimension (E)", ylab = "Forecast Skill (rho)")

simplex_output <- simplex(Bccm, lib, pred)
par(mar = c(4, 4, 1, 1), mgp = c(2.5, 1, 0))
plot(simplex_output$E, simplex_output$rho, type = "l", xlab = "Embedding Dimension (E)", ylab = "Forecast Skill (rho)")

# Prediction Decay
simplex_output <- simplex(Accm, lib, pred, E = 4, tp = 1:10)
par(mar = c(4, 4, 1, 1))
plot(simplex_output$tp, simplex_output$rho, type = "l", xlab = "Time to Prediction (tp)", ylab = "Forecast Skill (rho)")

simplex_output <- simplex(Bccm, lib, pred, E = 4, tp = 1:10)
par(mar = c(4, 4, 1, 1))
plot(simplex_output$tp, simplex_output$rho, type = "l", xlab = "Time to Prediction (tp)", ylab = "Forecast Skill (rho)")

# Identifying Nonlinearity
smap_output <- s_map(Accm, lib, pred, E=4)
par(mar = c(4, 4, 1, 1), mgp = c(2.5, 1, 0))
plot(smap_output$theta, smap_output$rho, type = "l", xlab = "Nonlinearity (theta)", ylab = "Forecast Skill (rho)")

smap_output <- s_map(Bccm, lib, pred, E=4)
par(mar = c(4, 4, 1, 1), mgp = c(2.5, 1, 0))
plot(smap_output$theta, smap_output$rho, type = "l", xlab = "Nonlinearity (theta)", ylab = "Forecast Skill (rho)")

# CCM(use multispatialCCM)
E_A <- 4
E_B <- 4
signal_A_out<-SSR_check_signal(A=Accm, E=E_A, tau=2, predsteplist=1:10)
signal_B_out<-SSR_check_signal(A=Bccm, E=E_B, tau=2, predsteplist=1:10)
CCM_boot_A<-CCM_boot(Accm, Bccm, E_A, tau=2, iterations=10)
CCM_boot_B<-CCM_boot(Bccm, Accm, E_B, tau=2, iterations=10)
(CCM_significance_test<-ccmtest(CCM_boot_A, CCM_boot_B))
plotxlimits<-range(c(CCM_boot_A$Lobs, CCM_boot_B$Lobs))
plot(CCM_boot_A$Lobs, CCM_boot_A$rho, type="l", col=1, lwd=2, xlim=c(plotxlimits[1], plotxlimits[2]), ylim=c(0,1), xlab="Library Size", ylab="Cross Map Skill (rho)", cex.lab = 1.5)
lines(CCM_boot_B$Lobs, CCM_boot_B$rho, type="l", col=2, lty=2, lwd=2)
legend("topleft", c("Salinity causes Flow velocity", "Flow velocity causes Salinity"), lty=c(1,2), col=c(1,2), lwd=2, bty="n", cex=1.2)


# --- do same task again ---

index <- 453 # probably do not have causality
Accm <- as.numeric(unlist(data$data[index,]$s))
Bccm <- as.numeric(unlist(data$data[index,]$t))

# show data
plot(Accm, type="l", col=1, lwd=2, xlim=c(0, 212), ylim=c(0,1), xlab="time step", ylab="Normalized Value", cex.lab = 1.5)
lines(Bccm, type="l", col=2, lty=2, lwd=2, cex.lab = 1.5)

# test of no correlation
cor(Accm, Bccm, method="spearman")
cor.test(Accm, Bccm, method="pearson")

# determine Embedding Dimension
lib <- c(1, 50)
pred <- c(90, 212)
simplex_output <- simplex(Accm, lib, pred)
par(mar = c(4, 4, 1, 1), mgp = c(2.5, 1, 0))
plot(simplex_output$E, simplex_output$rho, type = "l", xlab = "Embedding Dimension (E)", ylab = "Forecast Skill (rho)")

simplex_output <- simplex(Bccm, lib, pred)
par(mar = c(4, 4, 1, 1), mgp = c(2.5, 1, 0))
plot(simplex_output$E, simplex_output$rho, type = "l", xlab = "Embedding Dimension (E)", ylab = "Forecast Skill (rho)")

# Prediction Decay
simplex_output <- simplex(Accm, lib, pred, E = 4, tp = 1:10)
par(mar = c(4, 4, 1, 1))
plot(simplex_output$tp, simplex_output$rho, type = "l", xlab = "Time to Prediction (tp)", ylab = "Forecast Skill (rho)")

simplex_output <- simplex(Bccm, lib, pred, E = 4, tp = 1:10)
par(mar = c(4, 4, 1, 1))
plot(simplex_output$tp, simplex_output$rho, type = "l", xlab = "Time to Prediction (tp)", ylab = "Forecast Skill (rho)")

# Identifying Nonlinearity
smap_output <- s_map(Accm, lib, pred, E=4)
par(mar = c(4, 4, 1, 1), mgp = c(2.5, 1, 0))
plot(smap_output$theta, smap_output$rho, type = "l", xlab = "Nonlinearity (theta)", ylab = "Forecast Skill (rho)")

smap_output <- s_map(Bccm, lib, pred, E=4)
par(mar = c(4, 4, 1, 1), mgp = c(2.5, 1, 0))
plot(smap_output$theta, smap_output$rho, type = "l", xlab = "Nonlinearity (theta)", ylab = "Forecast Skill (rho)")

# CCM(use multispatialCCM)
E_A <- 2
E_B <- 2
signal_A_out<-SSR_check_signal(A=Accm, E=E_A, tau=1, predsteplist=1:10)
signal_B_out<-SSR_check_signal(A=Bccm, E=E_B, tau=1, predsteplist=1:10)
CCM_boot_A<-CCM_boot(Accm, Bccm, E_A, tau=1, iterations=10)
CCM_boot_B<-CCM_boot(Bccm, Accm, E_B, tau=1, iterations=10)
(CCM_significance_test<-ccmtest(CCM_boot_A, CCM_boot_B))
plotxlimits<-range(c(CCM_boot_A$Lobs, CCM_boot_B$Lobs))
plot(CCM_boot_A$Lobs, CCM_boot_A$rho, type="l", col=1, lwd=2, xlim=c(plotxlimits[1], plotxlimits[2]), ylim=c(0,1), xlab="L", ylab="rho")
lines(CCM_boot_B$Lobs, CCM_boot_B$rho, type="l", col=2, lty=2, lwd=2)
legend("topleft", c("Flow velocity cross map Salinity", "Salinity cross map Flow velocity"), lty=c(1,2), col=c(1,2), lwd=2, bty="n")


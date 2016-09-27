library(rEDM)
library(jsonlite)
library(multispatialCCM)

data = fromJSON("../python/data.json")


index <- 1
Accm <- as.numeric(unlist(data$data[index,]$s))
Bccm <- as.numeric(unlist(data$data[index,]$v))

# show data
plot(Accm, type="l", col=1, lwd=2, xlim=c(0, 212), ylim=c(0,1), xlab="time step", ylab="Normalized Value", cex.lab = 1.5)
lines(Bccm, type="l", col=2, lty=2, lwd=2, cex.lab = 1.5)
legend("topleft", c("Salinity", "Velocity"), cex=1.5, lty=c(1,2), col=c(1,2), lwd=2, bty="n")

# determine Embedding Dimension
lib <- c(1, 50)
pred <- c(90, 212)
simplex_output <- simplex(Accm, lib, pred)
par(mar = c(4, 4, 1, 1), mgp = c(2.5, 1, 0))
plot(simplex_output$E, simplex_output$rho, type = "l", xlab = "Embedding Dimension (E)", ylab = "Forecast Skill (rho)")
E_A = 2

simplex_output <- simplex(Bccm, lib, pred)
par(mar = c(4, 4, 1, 1), mgp = c(2.5, 1, 0))
plot(simplex_output$E, simplex_output$rho, type = "l", xlab = "Embedding Dimension (E)", ylab = "Forecast Skill (rho)")
E_B = 5

# Prediction Decay
simplex_output <- simplex(Accm, lib, pred, E = E_A, tp = 1:10)
par(mar = c(4, 4, 1, 1))
plot(simplex_output$tp, simplex_output$rho, type = "l", xlab = "Time to Prediction (tp)", ylab = "Forecast Skill (rho)")

simplex_output <- simplex(Bccm, lib, pred, E = E_B, tp = 1:10)
par(mar = c(4, 4, 1, 1))
plot(simplex_output$tp, simplex_output$rho, type = "l", xlab = "Time to Prediction (tp)", ylab = "Forecast Skill (rho)")
TAU = 1
# Identifying Nonlinearity
smap_output <- s_map(Accm, lib, pred, E=E_A)
par(mar = c(4, 4, 1, 1), mgp = c(2.5, 1, 0))
plot(smap_output$theta, smap_output$rho, type = "l", xlab = "Nonlinearity (theta)", ylab = "Forecast Skill (rho)")

smap_output <- s_map(Bccm, lib, pred, E=E_B)
par(mar = c(4, 4, 1, 1), mgp = c(2.5, 1, 0))
plot(smap_output$theta, smap_output$rho, type = "l", xlab = "Nonlinearity (theta)", ylab = "Forecast Skill (rho)")

# CCM(use multispatialCCM)
signal_A_out<-SSR_check_signal(A=Accm, E=E_A, tau=TAU, predsteplist=1:10)
signal_B_out<-SSR_check_signal(A=Bccm, E=E_B, tau=TAU, predsteplist=1:10)
CCM_boot_A<-CCM_boot(Accm, Bccm, E_A, tau=TAU, iterations=30)
CCM_boot_B<-CCM_boot(Bccm, Accm, E_B, tau=TAU, iterations=30)
(CCM_significance_test<-ccmtest(CCM_boot_A, CCM_boot_B))
plotxlimits<-range(c(CCM_boot_A$Lobs, CCM_boot_B$Lobs))
plot(CCM_boot_A$Lobs, CCM_boot_A$rho, type="l", col=1, lwd=2, xlim=c(plotxlimits[1], plotxlimits[2]), ylim=c(0,1), xlab="Library Size", ylab="Cross Map Skill (rho)", cex.lab = 1.5)
lines(CCM_boot_B$Lobs, CCM_boot_B$rho, type="l", col=2, lty=2, lwd=2)
legend("topleft", c("Salinity causes Velocity", "Velocity causes Salinituy"), lty=c(1,2), col=c(1,2), lwd=2, bty="n", cex=1.2)

# ---------------------------------------------------------------
index <- 10
Accm <- as.numeric(unlist(data$data[index,]$s))
Bccm <- as.numeric(unlist(data$data[index,]$v))

# show data
plot(Accm, type="l", col=1, lwd=2, xlim=c(0, 212), ylim=c(0,1), xlab="time step", ylab="Normalized Value", cex.lab = 1.5)
lines(Bccm, type="l", col=2, lty=2, lwd=2, cex.lab = 1.5)
legend("topleft", c("Salinity", "Velocity"), cex=1.5, lty=c(1,2), col=c(1,2), lwd=2, bty="n")

# determine Embedding Dimension
lib <- c(1, 50)
pred <- c(90, 212)
simplex_output <- simplex(Accm, lib, pred)
par(mar = c(4, 4, 1, 1), mgp = c(2.5, 1, 0))
plot(simplex_output$E, simplex_output$rho, type = "l", xlab = "Embedding Dimension (E)", ylab = "Forecast Skill (rho)")
E_A = 1

simplex_output <- simplex(Bccm, lib, pred)
par(mar = c(4, 4, 1, 1), mgp = c(2.5, 1, 0))
plot(simplex_output$E, simplex_output$rho, type = "l", xlab = "Embedding Dimension (E)", ylab = "Forecast Skill (rho)")
E_B = 1

# Prediction Decay
simplex_output <- simplex(Accm, lib, pred, E = E_A, tp = 1:10)
par(mar = c(4, 4, 1, 1))
plot(simplex_output$tp, simplex_output$rho, type = "l", xlab = "Time to Prediction (tp)", ylab = "Forecast Skill (rho)")

simplex_output <- simplex(Bccm, lib, pred, E = E_B, tp = 1:10)
par(mar = c(4, 4, 1, 1))
plot(simplex_output$tp, simplex_output$rho, type = "l", xlab = "Time to Prediction (tp)", ylab = "Forecast Skill (rho)")
TAU = 1
# Identifying Nonlinearity
smap_output <- s_map(Accm, lib, pred, E=E_A)
par(mar = c(4, 4, 1, 1), mgp = c(2.5, 1, 0))
plot(smap_output$theta, smap_output$rho, type = "l", xlab = "Nonlinearity (theta)", ylab = "Forecast Skill (rho)")

smap_output <- s_map(Bccm, lib, pred, E=E_B)
par(mar = c(4, 4, 1, 1), mgp = c(2.5, 1, 0))
plot(smap_output$theta, smap_output$rho, type = "l", xlab = "Nonlinearity (theta)", ylab = "Forecast Skill (rho)")

# CCM(use multispatialCCM)
signal_A_out<-SSR_check_signal(A=Accm, E=E_A, tau=TAU, predsteplist=1:10)
signal_B_out<-SSR_check_signal(A=Bccm, E=E_B, tau=TAU, predsteplist=1:10)
CCM_boot_A<-CCM_boot(Accm, Bccm, E_A, tau=TAU, iterations=30)
CCM_boot_B<-CCM_boot(Bccm, Accm, E_B, tau=TAU, iterations=30)
(CCM_significance_test<-ccmtest(CCM_boot_A, CCM_boot_B))
plotxlimits<-range(c(CCM_boot_A$Lobs, CCM_boot_B$Lobs))
plot(CCM_boot_A$Lobs, CCM_boot_A$rho, type="l", col=1, lwd=2, xlim=c(plotxlimits[1], plotxlimits[2]), ylim=c(0,1), xlab="Library Size", ylab="Cross Map Skill (rho)", cex.lab = 1.5)
lines(CCM_boot_B$Lobs, CCM_boot_B$rho, type="l", col=2, lty=2, lwd=2)
legend("topleft", c("Salinity causes Velocity", "Velocity causes Salinituy"), lty=c(1,2), col=c(1,2), lwd=2, bty="n", cex=1.2)

# ---------------------------------------------------------------
index <- 100
Accm <- as.numeric(unlist(data$data[index,]$s))
Bccm <- as.numeric(unlist(data$data[index,]$v))

# show data
plot(Accm, type="l", col=1, lwd=2, xlim=c(0, 212), ylim=c(0,1), xlab="time step", ylab="Normalized Value", cex.lab = 1.5)
lines(Bccm, type="l", col=2, lty=2, lwd=2, cex.lab = 1.5)
legend("topleft", c("Salinity", "Velocity"), cex=1.5, lty=c(1,2), col=c(1,2), lwd=2, bty="n")

# determine Embedding Dimension
lib <- c(1, 50)
pred <- c(90, 212)
simplex_output <- simplex(Accm, lib, pred)
par(mar = c(4, 4, 1, 1), mgp = c(2.5, 1, 0))
plot(simplex_output$E, simplex_output$rho, type = "l", xlab = "Embedding Dimension (E)", ylab = "Forecast Skill (rho)")
E_A = 1

simplex_output <- simplex(Bccm, lib, pred)
par(mar = c(4, 4, 1, 1), mgp = c(2.5, 1, 0))
plot(simplex_output$E, simplex_output$rho, type = "l", xlab = "Embedding Dimension (E)", ylab = "Forecast Skill (rho)")
E_B = 1

# Prediction Decay
simplex_output <- simplex(Accm, lib, pred, E = E_A, tp = 1:10)
par(mar = c(4, 4, 1, 1))
plot(simplex_output$tp, simplex_output$rho, type = "l", xlab = "Time to Prediction (tp)", ylab = "Forecast Skill (rho)")

simplex_output <- simplex(Bccm, lib, pred, E = E_B, tp = 1:10)
par(mar = c(4, 4, 1, 1))
plot(simplex_output$tp, simplex_output$rho, type = "l", xlab = "Time to Prediction (tp)", ylab = "Forecast Skill (rho)")
TAU = 1
# Identifying Nonlinearity
smap_output <- s_map(Accm, lib, pred, E=E_A)
par(mar = c(4, 4, 1, 1), mgp = c(2.5, 1, 0))
plot(smap_output$theta, smap_output$rho, type = "l", xlab = "Nonlinearity (theta)", ylab = "Forecast Skill (rho)")

smap_output <- s_map(Bccm, lib, pred, E=E_B)
par(mar = c(4, 4, 1, 1), mgp = c(2.5, 1, 0))
plot(smap_output$theta, smap_output$rho, type = "l", xlab = "Nonlinearity (theta)", ylab = "Forecast Skill (rho)")

# CCM(use multispatialCCM)
signal_A_out<-SSR_check_signal(A=Accm, E=E_A, tau=TAU, predsteplist=1:10)
signal_B_out<-SSR_check_signal(A=Bccm, E=E_B, tau=TAU, predsteplist=1:10)
CCM_boot_A<-CCM_boot(Accm, Bccm, E_A, tau=TAU, iterations=30)
CCM_boot_B<-CCM_boot(Bccm, Accm, E_B, tau=TAU, iterations=30)
(CCM_significance_test<-ccmtest(CCM_boot_A, CCM_boot_B))
plotxlimits<-range(c(CCM_boot_A$Lobs, CCM_boot_B$Lobs))
plot(CCM_boot_A$Lobs, CCM_boot_A$rho, type="l", col=1, lwd=2, xlim=c(plotxlimits[1], plotxlimits[2]), ylim=c(0,1), xlab="Library Size", ylab="Cross Map Skill (rho)", cex.lab = 1.5)
lines(CCM_boot_B$Lobs, CCM_boot_B$rho, type="l", col=2, lty=2, lwd=2)
legend("topleft", c("Salinity causes Velocity", "Velocity causes Salinituy"), lty=c(1,2), col=c(1,2), lwd=2, bty="n", cex=1.2)

# ---------------------------------------------------------------
index <- 200
Accm <- as.numeric(unlist(data$data[index,]$s))
Bccm <- as.numeric(unlist(data$data[index,]$v))

# show data
plot(Accm, type="l", col=1, lwd=2, xlim=c(0, 212), ylim=c(0,1), xlab="time step", ylab="Normalized Value", cex.lab = 1.5)
lines(Bccm, type="l", col=2, lty=2, lwd=2, cex.lab = 1.5)
legend("topleft", c("Salinity", "Velocity"), cex=1.5, lty=c(1,2), col=c(1,2), lwd=2, bty="n")

# determine Embedding Dimension
lib <- c(1, 50)
pred <- c(90, 212)
simplex_output <- simplex(Accm, lib, pred)
par(mar = c(4, 4, 1, 1), mgp = c(2.5, 1, 0))
plot(simplex_output$E, simplex_output$rho, type = "l", xlab = "Embedding Dimension (E)", ylab = "Forecast Skill (rho)")
E_A = 1

simplex_output <- simplex(Bccm, lib, pred)
par(mar = c(4, 4, 1, 1), mgp = c(2.5, 1, 0))
plot(simplex_output$E, simplex_output$rho, type = "l", xlab = "Embedding Dimension (E)", ylab = "Forecast Skill (rho)")
E_B = 1

# Prediction Decay
simplex_output <- simplex(Accm, lib, pred, E = E_A, tp = 1:10)
par(mar = c(4, 4, 1, 1))
plot(simplex_output$tp, simplex_output$rho, type = "l", xlab = "Time to Prediction (tp)", ylab = "Forecast Skill (rho)")

simplex_output <- simplex(Bccm, lib, pred, E = E_B, tp = 1:10)
par(mar = c(4, 4, 1, 1))
plot(simplex_output$tp, simplex_output$rho, type = "l", xlab = "Time to Prediction (tp)", ylab = "Forecast Skill (rho)")
TAU = 1
# Identifying Nonlinearity
smap_output <- s_map(Accm, lib, pred, E=E_A)
par(mar = c(4, 4, 1, 1), mgp = c(2.5, 1, 0))
plot(smap_output$theta, smap_output$rho, type = "l", xlab = "Nonlinearity (theta)", ylab = "Forecast Skill (rho)")

smap_output <- s_map(Bccm, lib, pred, E=E_B)
par(mar = c(4, 4, 1, 1), mgp = c(2.5, 1, 0))
plot(smap_output$theta, smap_output$rho, type = "l", xlab = "Nonlinearity (theta)", ylab = "Forecast Skill (rho)")

# CCM(use multispatialCCM)
signal_A_out<-SSR_check_signal(A=Accm, E=E_A, tau=TAU, predsteplist=1:10)
signal_B_out<-SSR_check_signal(A=Bccm, E=E_B, tau=TAU, predsteplist=1:10)
CCM_boot_A<-CCM_boot(Accm, Bccm, E_A, tau=TAU, iterations=30)
CCM_boot_B<-CCM_boot(Bccm, Accm, E_B, tau=TAU, iterations=30)
(CCM_significance_test<-ccmtest(CCM_boot_A, CCM_boot_B))
plotxlimits<-range(c(CCM_boot_A$Lobs, CCM_boot_B$Lobs))
plot(CCM_boot_A$Lobs, CCM_boot_A$rho, type="l", col=1, lwd=2, xlim=c(plotxlimits[1], plotxlimits[2]), ylim=c(0,1), xlab="Library Size", ylab="Cross Map Skill (rho)", cex.lab = 1.5)
lines(CCM_boot_B$Lobs, CCM_boot_B$rho, type="l", col=2, lty=2, lwd=2)
legend("topleft", c("Salinity causes Velocity", "Velocity causes Salinituy"), lty=c(1,2), col=c(1,2), lwd=2, bty="n", cex=1.2)


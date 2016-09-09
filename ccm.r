library(multispatialCCM)
library(jsonlite)

data = fromJSON("public/data.json")

index <- 13
Accm <- as.numeric(unlist(data$feature[index,]$x))
Bccm <- as.numeric(unlist(data$feature[index,]$y))

maxE <- 5
Emat <- matrix(nrow=maxE-1, ncol=2)
colnames(Emat) <- c("A", "B")
for(E in 2:maxE) {
  Emat[E - 1, "A"] <- SSR_pred_boot(A=Accm, E=E, predstep=1, tau=1)$rho
  Emat[E - 1, "B"] <- SSR_pred_boot(A=Bccm, E=E, predstep=1, tau=1)$rho
}
matplot(2:maxE, Emat, type="l", col=1:2, lty=1:2, xlab="E", ylab="rho", lwd=2)
legend("bottomleft", c("A", "B"), lty=1:2, col=1:2, lwd=2, bty="n")

E_A <- 2
E_B <- 2
signal_A_out<-SSR_check_signal(A=Accm, E=E_A, tau=1, predsteplist=1:10)
signal_B_out<-SSR_check_signal(A=Bccm, E=E_B, tau=1, predsteplist=1:10)
CCM_boot_A<-CCM_boot(Accm, Bccm, E_A, tau=1, iterations=10)
CCM_boot_B<-CCM_boot(Bccm, Accm, E_B, tau=1, iterations=10)
(CCM_significance_test<-ccmtest(CCM_boot_A, CCM_boot_B))
plotxlimits<-range(c(CCM_boot_A$Lobs, CCM_boot_B$Lobs))
plot(CCM_boot_A$Lobs, CCM_boot_A$rho, type="l", col=1, lwd=2, xlim=c(plotxlimits[1], plotxlimits[2]), ylim=c(0,1), xlab="L", ylab="rho")
matlines(CCM_boot_A$Lobs, cbind(CCM_boot_A$rho-CCM_boot_A$sdevrho, CCM_boot_A$rho+CCM_boot_A$sdevrho), lty=3, col=1)
lines(CCM_boot_B$Lobs, CCM_boot_B$rho, type="l", col=2, lty=2, lwd=2)
matlines(CCM_boot_B$Lobs, cbind(CCM_boot_B$rho-CCM_boot_B$sdevrho, CCM_boot_B$rho+CCM_boot_B$sdevrho), lty=3, col=2)
legend("topleft", c("A causes B", "B causes A"), lty=c(1,2), col=c(1,2), lwd=2, bty="n")


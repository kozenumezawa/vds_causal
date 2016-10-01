library(rEDM)
library(jsonlite)
library(multispatialCCM)

data(sardine_anchovy_sst)
anchovy_xmap_sst <- ccm(sardine_anchovy_sst, E = 3, lib_column = "anchovy", 
                        target_column = "np_sst", lib_sizes = seq(10, 80, by = 10), random_libs = FALSE)
sst_xmap_anchovy <- ccm(sardine_anchovy_sst, E = 3, lib_column = "np_sst", target_column = "anchovy", 
                        lib_sizes = seq(10, 80, by = 10), random_libs = FALSE)

a_xmap_t_means <- ccm_means(anchovy_xmap_sst)
t_xmap_a_means <- ccm_means(sst_xmap_anchovy)

par(mar = c(4, 4, 1, 1), mgp = c(2.5, 1, 0))
plot(a_xmap_t_means$lib_size, pmax(0, a_xmap_t_means$rho), type = "l", col = "red", 
     xlab = "Library Size", ylab = "Cross Map Skill (rho)", ylim = c(0, 0.4))
lines(t_xmap_a_means$lib_size, pmax(0, t_xmap_a_means$rho), col = "blue")
legend(x = "topleft", legend = c("anchovy xmap SST", "SST xmap anchovy"), col = c("red", 
                                                                                  "blue"), lwd = 1, inset = 0.02, cex = 0.8)

Accm <- sardine_anchovy_sst$anchovy
Bccm <- sardine_anchovy_sst$np_sst
signal_A_out<-SSR_check_signal(A=Accm, E=1, tau=1, predsteplist=1:10)
signal_B_out<-SSR_check_signal(A=Bccm, E=1, tau=1, predsteplist=1:10)
CCM_boot_A<-CCM_boot(Accm, Bccm, 1, tau=1, iterations=10)
CCM_boot_B<-CCM_boot(Bccm, Accm, 1, tau=1, iterations=10)
(CCM_significance_test<-ccmtest(CCM_boot_A, CCM_boot_B))
plotxlimits<-range(c(CCM_boot_A$Lobs, CCM_boot_B$Lobs))
plot(CCM_boot_A$Lobs, CCM_boot_A$rho, type="l", col=1, lwd=2, xlim=c(plotxlimits[1], plotxlimits[2]), ylim=c(0,0.4), xlab="Library Size", ylab="Cross Map Skill (rho)", cex.lab = 1.5)
lines(CCM_boot_B$Lobs, CCM_boot_B$rho, type="l", col=2, lty=2, lwd=2)
legend("topleft", c("anchovy causes SST", "SST causes anchovy"), lty=c(1,2), col=c(1,2), lwd=2, bty="n", cex=1.2)


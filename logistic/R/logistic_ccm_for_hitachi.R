library(multispatialCCM)
library(rEDM)

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

inputdata <- read.csv("../csv/inputdata_b_0_1.csv", header=FALSE)
TIMESTEP <- length(inputdata[1,])    # the length of X or Y 

t <- 1:TIMESTEP
# show  inputdata
X <- inputdata[1,]
Y <- inputdata[2,]
plot(t, X[1,], type = "l",xlim=c(0, 210), ylim=c(0,1), xlab = "t", ylab = "X(t)", col = "red")
lines(t, Y[1,], type = "l", xlab = "Time step", ylab = "value", col = "blue")

# principle of ccm
t1 <- 4
t2 <- 16
plot(t, X[1,], type = "l",xlim=c(t1, t2), ylim=c(0,1), xlab = "t", ylab = "X(t)", col = "red")
plot(t, Y[1,], type = "l",xlim=c(t1, t2), ylim=c(0,1), xlab = "t", ylab = "Y(t)", col = "blue")


Accm <- as.numeric(X)
Bccm <- as.numeric(Y)

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
Accm_Bccm <- data.frame(Accm=Accm, Bccm=Bccm)
Accm_xmap_Bccm <- ccm(Accm_Bccm, E = E, lib_column = "Accm", tau = TAU,
                      target_column = "Bccm", lib_sizes = seq(10, 200, by = 10), random_libs = TRUE)
Bccm_xmap_Accm <- ccm(Accm_Bccm, E = E, lib_column = "Bccm", tau = TAU,
                      target_column = "Accm", lib_sizes = seq(10, 200, by = 10), random_libs = TRUE)
Accm_xmap_Bccm_means <- ccm_means(Accm_xmap_Bccm)
Bccm_xmap_Accm_means <- ccm_means(Bccm_xmap_Accm)
par(mar = c(4, 4, 1, 1), mgp = c(2.5, 1, 0))
plot(Accm_xmap_Bccm_means$lib_size, pmax(0, Accm_xmap_Bccm_means$rho), type = "l", col = "red", xlab = "Library Size", ylab = "Cross Map Skill (rho)", ylim = c(0, 1))
lines(Bccm_xmap_Accm_means$lib_size, pmax(0, Bccm_xmap_Accm_means$rho), col = "blue")
legend(x = "topleft", legend = c("Accm xmap Bccm", "Bccm xmap Accm"), col = c("red", "blue"), lwd = 1, inset = 0.02, cex = 0.8)


# create trajectory vector for attractor
X_DIM <- E
BACK_MAX <- (X_DIM - 1) * TAU
X_N <- length(Accm) - BACK_MAX # length of x
x <- array(0, dim=c(X_N, X_DIM))
for (t in 1 : X_N) {
  for(j in 1:X_DIM) {
    x[t,j] <- Accm[(t + BACK_MAX) - (j - 1) * TAU]
  }
}

Y_DIM <- E
BACK_MAX <- (Y_DIM - 1) * TAU
Y_N <- length(Bccm) - BACK_MAX # length of y
y <- array(0, dim=c(Y_N, Y_DIM))
for (t in 1 : Y_N) {
  for(j in 1:Y_DIM) {
    y[t,j] <- Bccm[(t + BACK_MAX) - (j - 1) * TAU]
  }
}

# draw attractor
plot(x, xlim=c(0, 1), ylim=c(0,1), xlab = "x(t)", ylab = "x(t-1)")
plot(y, xlim=c(0, 1), ylim=c(0,1), xlab = "y(t)", ylab = "y(t-1)")

# draw attractor (t1-t2)
X_DIM <- E
BACK_MAX <- (X_DIM - 1) * TAU
X_N <- length(Accm) - BACK_MAX # length of x
x <- array(0, dim=c(t2-t1, X_DIM))
for (t in 1 : t2-t1) {
  for(j in 1:X_DIM) {
    x[t,j] <- Accm[(t + BACK_MAX + t1) - (j - 1) * TAU]
  }
}

Y_DIM <- E
BACK_MAX <- (Y_DIM - 1) * TAU
Y_N <- length(Bccm) - BACK_MAX # length of y
y <- array(0, dim=c(t2-t1, Y_DIM))
for (t in 1 : t2-t1) {
  for(j in 1:Y_DIM) {
    y[t,j] <- Bccm[(t + BACK_MAX + t1) - (j - 1) * TAU]
  }
}

plot(x, xlim=c(0.2, 1), ylim=c(0.2,1), xlab = "x(t)", ylab = "x(t-1)")
plot(y, xlim=c(0.2, 1), ylim=c(0.2,1), xlab = "y(t)", ylab = "y(t-1)")


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

# functions for twin surrogates
Heaviside <- function(x) {
  if (x > 0) {
    return (1)
  }
  return (0)
}

checkHavingTwin <- function(i, R) {
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

selectTwinIndex <- function(i, R) {
  twin <- c(i)
  N <- length(R[1,])
  for(j in 1 : N) {
    twin_flag <- TRUE
    for (k in 1 : N) {
      if (R[i,k] != R[j,k]) {
        twin_flag <- FALSE
      }
    }
    if (twin_flag == TRUE) {
      twin <- append(twin, j)
    }
  }
  random_index <- floor(runif(1, min=1, max=length(twin)+0.99999999))
  return (twin[random_index] + 1)
}

calculateCCMrho <- function(Accm, Bccm, Em, TAU) {
  Accm_Bccm <- data.frame(Accm=Accm, Bccm=Bccm)
  Bccm_xmap_Accm <- ccm(Accm_Bccm, E = Em, lib_column = "Bccm", tau = TAU,
                        target_column = "Accm", lib_sizes = seq(10, 200, by = 10), random_libs = TRUE)
  Bccm_xmap_Accm_means <- ccm_means(Bccm_xmap_Accm)
  return (Bccm_xmap_Accm_means$rho[length(Bccm_xmap_Accm_means$rho)])
}

data = fromJSON("../python/data.json")

index <- 1
Accm <- as.numeric(unlist(data$data[index,]$s))
Bccm <- as.numeric(unlist(data$data[index,]$t))

# show data
showdata(Accm, Bccm)
# determine Embedding Dimension
determineEmbeddingDimension(Accm)
determineEmbeddingDimension(Bccm)
E <- 5
# Prediction Decay
predictionDeacy(data = Accm, Em = E)
predictionDeacy(data = Bccm, Em = E)
TAU = 1
# Identifying Nonlinearity
identifyingNonlinearity(data = Accm, Em = E)
identifyingNonlinearity(data = Bccm, Em = E)
# draw CCM
drawCCM(Accm = Accm, Bccm = Bccm, E = E, TAU = TAU)

#---twin surrogate---
# create trajectory vector form attractor
X_DIM <- E
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

# create twin surrogates data
SURROGATE_N <- 1
x_s_bundle <- array(0, dim=c(SURROGATE_N, dim(x)))
for(surrogate_index in 1 : SURROGATE_N) {
  x_s <- array(0, dim=dim(x))
  # サロゲートデータを作るのに失敗することがあるので、100回試す
  for (try_number in 1 : 100) {
    before_index <- floor(runif(1, min=1, max=X_N+0.99999999))
    x_s[1,] <- x[before_index,]
    for (i in 2 : X_N) {
      if(checkHavingTwin(before_index, R)) {
        before_index <- selectTwinIndex(before_index, R)
      } else {
        before_index <- before_index + 1
      }
      if(before_index > X_N) {
        break
      }
      x_s[i,] <- x[before_index]
    }
    if(i == X_N) {
      break # surrogate dataの生成成功
    }  
  }
  x_s_bundle[surrogate_index, , ] <- x_s  
}

# create time series data from surrogate data
test_data_bundle <- array(0, c(SURROGATE_N, length(Accm)))
for(surrogate_index in 1 : SURROGATE_N) {
  surrogate_data <- x_s_bundle[surrogate_index, ,]
  time_series_data <- array(0, c(length(Accm)))
  # 時系列データ→trajectory vectorの逆操作
  for (t in 1 : X_N) {
    for(j in 1:X_DIM) {
      time_series_data[(t + BACK_MAX) - (j - 1) * TAU] <- surrogate_data[t,j]
    }
  }
  test_data_bundle[surrogate_index, ] <- time_series_data
}
# show data
plot(test_data_bundle[1,], type="l", col=1, lwd=2, xlim=c(0, 212), ylim=c(0,1), xlab="time step", ylab="Normalized Value", cex.lab = 1.5)

# conduct a test
data_rho <- calculateCCMrho(Accm = Accm, Bccm = Bccm, Em = E, TAU = TAU)
rho_list <- c()
for(surrogate_index in 1 : SURROGATE_N) {
  rho_list <- c(rho_list, calculateCCMrho(Accm = test_data_bundle[surrogate_index, ], Bccm = Bccm, Em = E, TAU = TAU))
}
max(rho_list)
t.test(rho_list)

write.csv(test_data_bundle, file='test.csv')

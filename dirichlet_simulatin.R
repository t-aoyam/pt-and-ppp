# install.packages('MCMCpack')
library(MCMCpack)

# copying with ctx_size = 100
num_samples = 1000
ctx_size = 100
attention <- c()
for (j in (ctx_size/2):ctx_size){  # tokens
  samples <- rdirichlet(num_samples, rep(1, j))
  attention <- cbind(attention, samples[, j-(ctx_size/2-1)])
}
# rdirichlet()
quantile(apply(attention, 1, mean), 0.999999999)

mean(attention)
sd(attention)/sqrt(50000)
boxplot(apply(attention, 1, mean))
s <- rdirichlet(1000, rep(1, 50))
rdirichlet(1,rep(1, 50))[50]
seq(70, 1, -49)

rep(1, 75)
samples <- rdirichlet(1000, rep(1,75))
samples
quantile(apply(samples, 1, max), 0.9999999999)
quantile(apply(samples, 2, mean), 0.9999999999999)

# SAS with ctx_size = 1024
alpha <- rep(1, (512+1024)/2)
samples <- rdirichlet(floor(251493/1024), alpha)
quantile(samples, 0.999999999999999)

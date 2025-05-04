args <- commandArgs(trailingOnly = TRUE)
# args <- c("EleutherAI-Pythia-160m", 'NULL', 'NULL', T)  # for debugging
# Assign arguments to variables (assuming 3 arguments)
model_name <- args[1]    # First argument: data file path
ablation_mode <- args[2]  # Second argument: output file path
ablation_head <- args[3]  # Third argument: a numeric parameter
if (length(args) == 3) args <- c(args, F)
write_coeff <- args[4]

ROOT_DIR <- getwd()
DATA_DIR <- file.path(ROOT_DIR, 'data')
library(glue)

if (ablation_mode != 'NULL' && ablation_head != 'NULL') {
  model_info <- glue("{model_name}@{ablation_head}-{ablation_mode}")
  DATA_DIR <- file.path(DATA_DIR, 'by_head_ablation')
} else {
  model_info <- model_name
}

fn <- glue("harmonized_results_{model_info}.csv")
input_fp <- file.path(DATA_DIR, 'rt_data', fn)

library(tidyverse)
library(lme4)
library(lmerTest)
library(logging)
library(mvtnorm)
library(mgcv)
# Provides bootstrap resampling tools
library(rsample)

# Compute the log-likelihood of a new dataset using a fit lme4 model.
logLik_test <- function(lm, test_X, test_y) {
  predictions <- predict(lm, test_X, re.form=NA)
  # Get std.dev. of residual, estimated from train data
  stdev <- sigma(lm)
  # For each prediction--observation, get the density p(obs | N(predicted, model_sigma)) and reduce
  density <- sum(dnorm(test_y, predictions, stdev, log=TRUE))
  return(density)
}
# Get per-prediction log-likelihood
logLik_test_per <- function(lm, test_X, test_y) {
  predictions <- predict(lm, test_X, re.form=NA)
  # Get std.dev. of residual, estimated from train data
  stdev <- sigma(lm)
  # For each prediction--observation, get the density p(obs | N(predicted, model_sigma))
  densities <- dnorm(test_y, predictions, stdev, log=TRUE)
  return(densities)
}
# Compute MSE of a new dataset using a fit lme4 model.
mse_test <- function(lm, test_X, test_y) {
  return(mean((predict(lm, test_X, re.form=NA) - test_y) ^ 2))
}
#Sanity checks
#mylm <- gam(psychometric ~  s(surprisal, bs = "cr", k = 20) + s(prev_surp, bs = "cr", k = 20) + te(freq, len, bs = "cr") + te(prev_freq, prev_len, bs = "cr"), data=train_data)
#c(logLik(mylm), logLik_test(mylm, train_data, train_data$psychometric))
#logLik_test(mylm, test_data, test_data$psychometric)

# Data loading and preprocessing

data = read.csv(input_fp)

all_data = data %>%
  mutate(seed = as.factor(seed)) %>%
  group_by(corpus, model, training, seed) %>%
  mutate(prev_surp = lag(surprisal),
         prev_code = lag(code),
         prev_len = lag(len),
         prev_freq = lag(freq),
         prev_surp = lag(surprisal),
         
         prev2_freq = lag(prev_freq),
         prev2_code = lag(prev_code),
         prev2_len = lag(prev_len),
         prev2_surp = lag(prev_surp),
         
         prev3_freq = lag(prev2_freq),
         prev3_code = lag(prev2_code),
         prev3_len = lag(prev2_len),
         prev3_surp = lag(prev2_surp),
         
         prev4_freq = lag(prev3_freq),
         prev4_code = lag(prev3_code),
         prev4_len = lag(prev3_len),
         prev4_surp = lag(prev3_surp)) %>%
  ungroup() %>%
  
  # Filter back two for the dundee corpus. Filter back 1 for all other corpora
  # NB this effectively removes all zero-surprisal rows, since early-sentence tokens don't have contiguous token history
  filter((corpus %in% c("dundee", "meco", "provo") & code == prev2_code + 2) | (!(corpus %in% c("dundee", "meco", "provo")) & code == prev4_code + 4)) %>%
  
  select(-prev_code, -prev2_code, -prev3_code) %>%
  drop_na()

all_data = all_data %>%
  mutate(
    model = as.character(model),
    model = if_else(model == "gpt-2", "gpt2", model),
    model = as.factor(model))

missing_rows = all_data %>% complete(nesting(corpus, code), nesting(model, training, seed)) %>% 
  group_by(corpus, code) %>% 
  filter(sum(is.na(surprisal)) > 0) %>% 
  ungroup() %>% 
  anti_join(all_data, by=c("corpus", "code", "model", "training", "seed"))

missing_rows %>% ggplot(aes(x=corpus, fill=factor(paste(model,training)))) + geom_bar(position=position_dodge(width=0.8))
print(missing_rows %>% group_by(model, training, seed, corpus) %>% summarise(n=n())) %>% arrange(desc(n))

# Compute the ideal number of model--seed--training observations per token.
to_drop = all_data %>%
  group_by(corpus, code) %>% summarise(n = n()) %>% ungroup() %>%
  group_by(corpus) %>% mutate( max_n = max(n)) %>% ungroup() %>%
  filter(max_n != n) %>%
  select(code, corpus)

loginfo(paste("Dropping", nrow(to_drop), "observations corresponding to corpus tokens which are missing observations for some model."))
loginfo(paste("Dropping", to_drop %>% group_by(corpus, code) %>% n_groups(), "tokens which are missing observations for some model."))

all_data = all_data %>% anti_join(to_drop %>% group_by(corpus, code), by=c("corpus", "code"))
loginfo(paste("After drop,", nrow(all_data), "observations (", all_data %>% group_by(corpus, code) %>% n_groups(), " tokens) remain."))


to_drop_zero_surps = all_data %>% group_by(corpus, code) %>% filter(any(surprisal == 0)) %>% ungroup()
loginfo(paste("Dropping", nrow(to_drop_zero_surps), "observations corresponding to corpus tokens which have surprisal zeros for some model."))
loginfo(paste("Dropping", to_drop_zero_surps %>% group_by(corpus, code) %>% n_groups(), "tokens which have surprisal zeros for some model."))

all_data = all_data %>% anti_join(to_drop_zero_surps %>% group_by(corpus, code), by=c("corpus", "code"))
loginfo(paste("After drop,", nrow(all_data), "observations (", all_data %>% group_by(corpus, code) %>% n_groups(), " tokens) remain."))

to_drop_zero_psychs = all_data %>% group_by(corpus, code) %>% filter(any(psychometric == 0)) %>% ungroup()
loginfo(paste("Dropping", nrow(to_drop_zero_psychs), "observations corresponding to corpus tokens which have psychometric zeros for some model."))
loginfo(paste("Dropping", to_drop_zero_psychs %>% group_by(corpus, code) %>% n_groups(), "tokens which have psychometric zeros for some model."))

all_data = all_data %>% anti_join(to_drop_zero_psychs %>% group_by(corpus, code), by=c("corpus", "code"))
loginfo(paste("After drop,", nrow(all_data), "observations (", all_data %>% group_by(corpus, code) %>% n_groups(), " tokens) remain."))


# Train Linear Models which are used to assess Delta Log Lik


# Compute linear model stats for the given training data subset and full test data.
# Automatically subsets the test data to match the relevant group for which we are training a linear model.
get_lm_data <- function(df, test_data, formula, fold, store_env) {
  #this_lm <- gam(formula, data=df);
  this_lm = lm(formula, data=df)
  # if (grepl("surp", as.character(formula)[3])) browser()
  if (grepl("surp", as.character(formula)[3]) && write_coeff==T){
    # print(c(as.character(df$model[1]), df$corpus[1], this_lm$coefficients[c('prev_surp', 'prev2_surp')]))
    write(
      paste(
        as.character(df$model[1]),
        df$corpus[1],
        this_lm$coefficients['prev_surp'],
        this_lm$coefficients['prev2_surp'],
        sep='\t'
        ),
      file="Coeffs_160M.txt", append=TRUE
      )
  }
  this_test_data <- semi_join(test_data, df, by=c("training", "model", "seed", "corpus"));
  
  # Save lm to the global env so that we can access residuals later.
  lm_name = paste(unique(paste(df$model, df$training, df$seed, df$corpus))[1], fold)
  assign(lm_name, this_lm, envir=store_env)
  
  summarise(df,
            log_lik = as.numeric(logLik(this_lm, REML = F)),
            test_lik = logLik_test(this_lm, this_test_data, this_test_data$psychometric),
            test_mse = mse_test(this_lm, this_test_data, this_test_data$psychometric))
}

# For a previously fitted lm stored in store_env, get the residuals on test data of the relevant data subset.
get_lm_residuals <- function(df, fold, store_env) {
  # Retrieve the relevant lm.
  lm_name = paste(unique(paste(df$model, df$training, df$seed, df$corpus))[1], fold)
  this_lm <- get(lm_name, envir=store_env)
  
  mutate(df,
         likelihood = logLik_test_per(this_lm, df, df$psychometric),
         resid = df$psychometric - predict(this_lm, df, re.form=NA))
}
# Compute per-example delta-log-likelihood for the given test fold.
get_lm_delta_log_lik <- function(test_data, fold, baseline_env, full_env) {
  lm_name = paste(unique(paste(test_data$model, test_data$training, test_data$seed, test_data$corpus))[1], fold)
  baseline_lm <- get(lm_name, envir=baseline_env)
  full_lm <- get(lm_name, envir=full_env)
  
  delta_log_lik = logLik_test_per(full_lm, test_data, test_data$psychometric) - logLik_test_per(baseline_lm, test_data, test_data$psychometric)
  return(cbind(test_data, delta_log_lik=delta_log_lik))
}
#####
# Define regression formulae.

# Regression code to fit GAM models.
#baseline_rt_regression = psychometric ~ te(freq, len, bs = "cr") + te(prev_freq, prev_len, bs = "cr") + te(prev2_freq, prev2_len, bs = "cr")
#baselie_sprt_regression = psychometric ~ te(freq, len, bs = "cr") + te(prev_freq, prev_len, bs = "cr") + te(prev2_freq, prev2_len, bs = "cr") + te(prev3_freq, prev3_len, bs = "cr") + te(prev4_freq, prev4_len, bs = "cr")

#full_rt_regression = psychometric ~ s(surprisal, bs = "cr", k = 20) + s(prev_surp, bs = "cr", k = 20) + s(prev2_surp, bs = "cr", k = 20) + te(freq, len, bs = "cr") + te(prev_freq, prev_len, bs = "cr") + te(prev2_freq, prev2_len, bs = "cr")
#full_sprt_regression = psychometric ~ s(surprisal, bs = "cr", k = 20) + s(prev_surp, bs = "cr", k = 20) + s(prev2_surp, bs = "cr", k = 20) + s(prev3_surp, bs = "cr", k = 20) + s(prev4_surp, bs = "cr", k = 20) + te(freq, len, bs = "cr") + te(prev_freq, prev_len, bs = "cr") + te(prev2_freq, prev2_len, bs = "cr") + te(prev3_freq, prev3_len, bs = "cr") + te(prev4_freq, prev4_len, bs = "cr")

# Regression Code to fit linear models
baseline_rt_regression = psychometric ~ freq + prev_freq + prev2_freq + len + prev_len + prev2_len
baseline_sprt_regression = psychometric ~ freq + prev_freq + prev2_freq + prev3_freq + prev4_freq + len + prev_len + prev2_len + prev3_len + prev4_len

full_sprt_regression = psychometric ~ surprisal + prev_surp + prev2_surp + prev3_surp + prev4_surp + freq + prev_freq + prev2_freq + prev3_freq + prev4_freq + len + prev_len + prev2_len + prev3_len + prev4_len
full_rt_regression = psychometric ~ surprisal + prev_surp + prev2_surp + freq + prev_freq + prev2_freq + len + prev_len + prev2_len

#####
# Prepare frames/environments for storing results/objects.
baseline_results = data.frame()
full_model_results = data.frame()
baseline_residuals = data.frame()
full_residuals = data.frame()
log_lik_deltas = data.frame()

#Randomly shuffle the data
all_data<-all_data[sample(nrow(all_data)),]
#Create K equally size folds
K = 10
folds <- cut(seq(1,nrow(all_data)),breaks=K,labels=FALSE)
#Perform 10 fold cross validation

# Fit models for some fold of the data.
baseline_corpus = function(corpus, df, test_data, fold, env) {
  if(corpus %in% c("dundee", "meco", "provo")) {
    get_lm_data(df, test_data, baseline_rt_regression, fold, env)
  } else {
    get_lm_data(df, test_data, baseline_sprt_regression, fold, env)
  }
}

full_model_corpus = function(corpus, df, test_data, fold, env) {
  if(corpus %in% c("dundee", "meco", "provo")) {
    get_lm_data(df, test_data, full_rt_regression, fold, env)
  } else {
    get_lm_data(df, test_data, full_sprt_regression, fold, env)
  }
}

# Prepare a new Environment in which we store fitted LMs, which we'll query later for residuals and other metrics.
baseline_env = new.env()
full_env = new.env()

for(i in 1:K) { 
  #Segement your data by fold using the which() function 
  testIndexes <- which(folds==i, arr.ind=TRUE)
  test_data <- all_data[testIndexes, ]
  train_data <- all_data[-testIndexes, ]
  
  # Compute a baseline linear model for each model--training--seed--RT-corpus combination.
  baselines = train_data %>%
    group_by(model, training, seed, corpus) %>%
    # print(model) %>%
    do(baseline_corpus(unique(.$corpus), ., test_data, i, baseline_env)) %>%
    ungroup() %>%
    mutate(seed = as.factor(seed),
           fold = i)
  
  baseline_results = rbind(baseline_results, baselines)
  
  # Compute a full linear model for each model--training--seed-RT-corpus combination
  full_models = train_data %>%
    group_by(model, training, seed, corpus) %>%
    do(full_model_corpus(unique(.$corpus), ., test_data, i, full_env)) %>%
    ungroup() %>%
    mutate(seed = as.factor(seed),
           fold = i)
  
  full_model_results = rbind(full_model_results, full_models)
  
  # Compute delta-log-likelihoods
  fold_log_lik_deltas = test_data %>%
    group_by(model, training, seed, corpus) %>%
    do(get_lm_delta_log_lik(., i, baseline_env, full_env)) %>%
    ungroup()
  
  log_lik_deltas = rbind(log_lik_deltas, fold_log_lik_deltas)
  
  fold_baseline_residuals = test_data %>%
    group_by(model, training, seed, corpus) %>%
    do(get_lm_residuals(., i, baseline_env)) %>%
    ungroup()
  
  baseline_residuals = rbind(baseline_residuals, fold_baseline_residuals)
  
  fold_full_residuals = test_data %>%
    group_by(model, training, seed, corpus) %>%
    do(get_lm_residuals(., i, full_env)) %>%
    ungroup()
  
  full_residuals = rbind(full_residuals, fold_full_residuals)
}

model_deltas = log_lik_deltas %>%
  group_by(model, training, seed, corpus) %>% 
  summarise(mean_delta_log_lik = mean(delta_log_lik),
            sem_delta_log_lik = sd(delta_log_lik) / sqrt(length(delta_log_lik)))

metric <- "ΔLogLik"
#metric <- "-ΔMSE"

# # Select the relevant metric.
model_deltas = model_deltas %>%
  # Retrieve the current test metric
  mutate(delta_test_mean = mean_delta_log_lik,
         delta_test_sem = sem_delta_log_lik) %>%
  # mutate(delta_test_mean = mean_delta_mse,
  #        delta_test_sem = sem_delta_mse)
  
  # Remove the raw metrics.
  select(-mean_delta_log_lik, -sem_delta_log_lik,
         #-mean_delta_mse, -sem_delta_mse
  )

if (write_coeff == F){
  out_fn <- glue("model_deltas_{model_info}.csv")
  write.csv(model_deltas, file.path(DATA_DIR, 'analysis_checkpoints', out_fn))
}
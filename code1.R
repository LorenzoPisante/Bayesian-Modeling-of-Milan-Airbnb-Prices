

install.packages("devtools")
install.packages("here")
install.packages("dplyr")
install.packages("skimr")
install.packages("stringr")
install.packages("knitr")
install.packages("tidyverse")
install.packages("janitor")
install.packages("ggmap")
install.packages("rgeos")
install.packages("maptools")
install.packages("RColorBrewer")
install.packages("rstan")
install.packages("caret")
install.packages("loo")
install.packages("gridExtra")
install.packages("tidyr")
install.packages("bayesplot")
install.packages("tidybayes")


# load libraries
library(here)
library(dplyr)
library(skimr)
library(stringr)
library(knitr)
library(tidyverse)
library(janitor)
library(ggmap)
library(rgeos)
library(maptools)
library(RColorBrewer)
library(rstan)
library(caret)
library(loo)
library(gridExtra)

library(bayesplot) 

library(tidybayes)

library(tidyr)





# Ottieni la directory di lavoro corrente
current_directory <- getwd()

# Stampa la directory di lavoro corrente
print(current_directory)

# read in data
airbnb_raw <- read.csv(here("Desktop/lore/code/airbnbR.csv"),na.strings=c(""," ","NA"))

#rendo zona categorica
airbnb_raw$zona <- as.factor(airbnb_raw$zona)

# restrict dataframe to certain variables
airbnb_subset <- airbnb_raw %>% dplyr::select(
  host_id,

  host_listings_count,

  neighbourhood,
  room_type,

  bathrooms_text,
  accommodates,
  bedrooms,
  price,
  number_of_reviews,

  review_scores_rating,
  review_scores_accuracy,
  review_scores_cleanliness,
  review_scores_checkin,
  review_scores_communication,
  review_scores_location,
  review_scores_value,
  zona
)

# total number of obs
airbnb_subset %>% count()

# total number of distinct obs
airbnb_subset %>% distinct() %>% count()

# remove duplicates
airbnb_subset <- airbnb_subset %>% distinct()

airbnb_final <-airbnb_subset

# add log price
airbnb_final <- airbnb_final %>% mutate(log_price = log(price))
# PRICE

price_dat <- airbnb_final %>%
  select(price, log_price) %>%
  pivot_longer(cols=price:log_price, names_to = "Type", values_to = "Price")

price_plot <-	ggplot(price_dat, aes(x = Price, fill=Type, facet = Type, ..density..)) +
  geom_histogram(bins = 100, fill = "black", alpha = 0.3) +
  geom_density(fill="lightblue", alpha = 0.8) +
  facet_wrap(~Type, scales = "free") +
  theme_bw() +
  labs(title = "Price and Log Price Density Plots") +
  theme(axis.title.x=element_blank())

price_plot

##raggruppamento per quartieri
  
# Calcolare la media dei prezzi e il conteggio delle case per ogni quartiere
result <- airbnb_final %>%
  group_by(neighbourhood) %>%
  summarise(average_price = mean(log_price, na.rm = TRUE), count = n()) %>%
  arrange(desc(average_price))

# Stampare i primi 10 quartieri con la media dei prezzi più alta
head(result, 10)

# Prendi i primi 10 quartieri con la media dei prezzi più alta
top_10_neighbourhoods <- head(result, 10)

# Creare un grafico a barre
ggplot(top_10_neighbourhoods, aes(x = reorder(neighbourhood, -average_price), y = average_price, fill = count)) +
  geom_bar(stat = "identity") +
  scale_fill_gradient(low = "blue", high = "red") +
  labs(x = "Quartiere", y = "Log-Prezzo medio", fill = "Numero di case") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  ggtitle("Top 10 quartieri con il prezzo medio più alto per case")





airbnb_final <- na.omit(airbnb_final, "zona")

##raggruppamento per zone

result_zone <- airbnb_final %>%
  group_by(zona) %>%
  summarise(
    average_price = mean(log_price, na.rm = TRUE),
    count = n()
  ) %>%
  arrange(desc(average_price))

# Mostra le prime 10 zone con la media del prezzo più alta
top_zones <- head(result_zone, 9)
print(top_zones)

ggplot(top_zones, aes(x = reorder(zona, -average_price), y = average_price, fill = count)) +
  geom_bar(stat = "identity") +
  scale_fill_gradient(low = "blue", high = "red") +
  labs(x = "Zona", y = "Log-Prezzo medio", fill = "Numero di case") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  ggtitle("Top zone con il prezzo medio più alto per case")

# ROOM TYPE
# boxplot
type_boxplot <- ggplot(data = airbnb_final, aes(x = "", y = log_price, fill=room_type)) + 
  geom_boxplot() + 
  theme_bw() +
  labs(title="Log-Price vs. Room Type", y="Log-Price ($)") +
  theme(axis.title.x = element_blank()) 
#guides(fill=guide_legend(nrow=2,byrow=TRUE))

# room type density plot
type_density <- airbnb_final %>% ggplot(aes(x=log_price, colour=room_type)) +
  geom_density() +
  theme_bw() +
  labs(title="Price Densities by Room Type") 
  
type_density

airbnb_final <- rename(airbnb_final, bathrooms = bathrooms_text)

# BED/BATH/ACCOMODATES
bed_bath_dat <- airbnb_final %>% select(log_price, bathrooms, accommodates, bedrooms) %>%
  pivot_longer(-log_price, names_to="Category", values_to="Quantity")

bed_bath_plot <- ggplot(bed_bath_dat,aes(x=Quantity, y=log_price, colour=Category)) +
  geom_point() +
  facet_wrap(~Category, scales="free") +
  geom_smooth(formula = y ~ x, method = "lm", color="black") +
  theme_bw() +
  theme(legend.position = "none") +
  scale_color_brewer(palette="Accent") +
  labs(title="Listing Characteristics vs. Log-Price", y="Log-Price ($)")

bed_bath_plot



#BAYESIAN PART

# lets get the data into the correct format for stan

# make room type a factor
print(levels(as.factor(airbnb_final$room_type)))
airbnb_data <- airbnb_final %>% 
  select(-host_id, -price) %>%
  mutate(room_type = as.numeric(as.factor(room_type)))
airbnb_data$zona <- as.factor(airbnb_data$zona)


# mean center numeric variables
airbnb_data2 <-
  airbnb_data %>% mutate(
    host_listings_count = scale(host_listings_count, center=TRUE, scale=FALSE),
    bathrooms = scale(bathrooms, center=TRUE, scale=FALSE),
    accommodates = scale(accommodates, center=TRUE, scale=FALSE),
    bedrooms = scale(bedrooms, center=TRUE, scale=FALSE),
    number_of_reviews = scale(number_of_reviews, center=TRUE, scale=FALSE),
    review_scores_rating = scale(review_scores_rating, center=TRUE, scale=FALSE),
    review_scores_accuracy = scale(review_scores_accuracy, center=TRUE, scale=FALSE),
    review_scores_cleanliness = scale(review_scores_cleanliness, center=TRUE, scale=FALSE),
    review_scores_checkin = scale(review_scores_checkin, center=TRUE, scale=FALSE),
    review_scores_communication = scale(review_scores_communication, center=TRUE, scale=FALSE),
    review_scores_location = scale(review_scores_location, center=TRUE, scale=FALSE),
    review_scores_value = scale(review_scores_value, center=TRUE, scale=FALSE)
  ) 
set.seed(8)

#Model 1
# create dummy vars for categorical vars
no_district <- airbnb_data2 %>% select(-zona)
mod1_data <- as.data.frame(model.matrix( ~ ., data = no_district))

# select sample
sample <- mod1_data %>% sample_n(2000)
xmatrix <- as.matrix(sample %>% select(-log_price, -room_type, -`(Intercept)`))

# put into stan data
stan_data <- list(N = nrow(xmatrix),
                  J = max(mod1_data$room_type),
                  K = ncol(xmatrix),
                  room_type = sample$room_type,
                  X = xmatrix,
                  y = sample$log_price)

model1 <- stan(data = stan_data, 
               file = here("Desktop/lore/code/airbnb_model.stan"),
               iter = 2000,
               seed = 8)

# save model with everything
saveRDS(model1, "fit1.rds")

# load in the model
model1 <- readRDS("fit1.rds")
max(summary(model1)$summary[,c("Rhat")])

summary(model1)$summary[c(paste0("alpha[", 1:4, "]"), paste0("beta[", 1:6, "]"),
                          paste0("beta[", 146:157, "]"), "mu", "sigma_a", "sigma_y"),
                        c("mean", "se_mean", "n_eff", "Rhat")]

# traceplots - alphas
pars = c(paste0("alpha[", 1:4, "]"))
traceplot(model1, pars=pars)

stan_dens(model1, separate_chains=TRUE, pars=pars)
pairs(model1, pars=pars)

pars = c("mu", "sigma_a", "sigma_y")
#pars = c(paste0("beta[", 1:6, "]"),paste0("beta[", 146:157, "]"))
traceplot(model1, pars=pars)
pairs(model1, pars=pars)

## Model 2 - Districts and all Covariates


#create dummy variables
airbnb_data2$zona <- factor(airbnb_data2$zona, levels = c("9","1","2","3","4","5","6","7","8"))

# create the data
district <- airbnb_data2 %>% select(-neighbourhood)
#dummy_vars <- model.matrix(~zona - 1, data = district) 

# Converte il risultato in un dataframe e aggiunge al dataframe originale
#district <- cbind(district, as.data.frame(dummy_vars))

mod2_data <- as.data.frame(model.matrix( ~ ., data = district))
#mod2_data <- dummy_cols(district, select_columns = "zona", remove_first_dummy = FALSE, remove_selected_columns = TRUE)


# select sample
sample2 <- mod2_data %>% sample_n(2000)
xmatrix2 <- as.matrix(sample2 %>% select(-log_price, -room_type,-`(Intercept)`))
stan_data2 <- list(N = nrow(xmatrix2),
                   J = max(mod2_data$room_type),
                   K = ncol(xmatrix2),
                   room_type = sample2$room_type,
                   X = xmatrix2,
                   y = sample2$log_price)
model2 <- stan(data = stan_data2, 
               file = here("Desktop/lore/code/airbnb_model.stan"),
               iter = 2000,
               seed = 8)

# save model with everything
saveRDS(model2, "fit2.rds")

# load in the model
model2 <- readRDS("fit2.rds")

# look at max rhat val
max(summary(model2)$summary[,c("Rhat")])


#summary(model2)$summary[c(paste0("alpha[", 1:4, "]"), 
                          #paste0("beta[", 1:18, "]"), "mu", "sigma_a", "sigma_y"),
                        #c("mean", "se_mean", "n_eff", "Rhat")]

# traceplots - alphas
pars = c(paste0("alpha[", 1:4, "]"))
traceplot(model2, pars=pars)

# pairs plot - alphas
#stan_dens(model2, separate_chains=TRUE, pars=pars)
pairs(model2, pars=pars)

#overall distributions of the replicated datasets versus the data

set.seed(1856)
y <- sample2$log_price
yrep2 <- rstan::extract(model2)[["log_price_rep"]]
samp100 <- sample(nrow(yrep2), 100)
ppc_dens <- ppc_dens_overlay(y, yrep2[samp100, ])  + ggtitle("Distribution of Observed vs. Predicted Log Prices")
sample2 <- sample2 %>% mutate(room_names = case_when(room_type == 1 ~ "Entire Home/Apartment", 
                                                     room_type == 2 ~ "Hotel Room",
                                                     room_type == 3 ~ "Private Room",
                                                     room_type == 4 ~ "Shared Room"))
# test stats - median
ppc_stat <- ppc_stat_grouped(sample2$log_price, yrep2, group = sample2$room_names, stat = 'median') + ggtitle("Median by Room Type - Model 2")

grid.arrange(ppc_dens,ppc_stat, ncol=1, nrow=2)


#mean room type alphas

# extract results from summary
mod2_summary <- as.data.frame(summary(model2,probs = c(0.025, 0.975))$summary)
colnames(mod2_summary) <- c("mean", "se_mean", "sd", "ci_lower", "ci_upper", "n_eff", "rhat")

mod2_summary$row_name <- rownames(mod2_summary)

# keep alphas
mod2_plot <- mod2_summary %>%
  filter(str_detect(row_name, 'alpha')) 

# add room_type name
mod2_plot <- mod2_plot %>% mutate(room_type = c("Entire Home/Apartment", "Hotel Room", "Private Room", "Shared Room"))

# plot results
ggplot(data = mod2_plot, aes(x = mean, y = room_type)) + 
  geom_point() +
  geom_line() + 
  geom_linerange(aes(xmin = ci_lower, xmax = ci_upper), size=0.25) + 
  theme_minimal() +
  labs(x = "Alpha", y = "Room Type", 
       title = "Room Type Alphas (Intercepts)", subtitle="With 95% CIs")


cols2 <- colnames(sample2 %>% select(-log_price, -room_type, -`(Intercept)`,-room_names))

# keep betas
mod2_plot_beta <- mod2_summary %>%
  filter(str_detect(row_name, 'beta')) 

# add room_type name
mod2_plot_beta <- mod2_plot_beta %>% mutate(variable=cols2)

# plot results
ggplot(data = mod2_plot_beta, aes(x = mean, y = variable)) + 
  geom_point() +
  geom_line() + 
  geom_linerange(aes(xmin = ci_lower, xmax = ci_upper), size=0.25) + 
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 30)) + 
  labs(y = "Variable", x = "Beta", 
       title = "Beta Coefficients", subtitle="With 95% CIs")

## Model 3 - Districts with Selected Covariates


set.seed(8)

# keep only selected vars from EDA
mod3 <- airbnb_data2 %>% select(bedrooms, bathrooms, room_type, zona, log_price, accommodates, review_scores_rating, review_scores_cleanliness, review_scores_location, review_scores_value)
# Crea la nuova colonna 'zona1'
mod3$zona1 <- ifelse(mod3$zona == "1", 1, 0)

# Elimina la colonna 'zona'
mod3$zona <- NULL

# create dummy vars for categorical vars
#mod3_data <- dummy_cols(mod3, select_columns = "zona", remove_first_dummy = FALSE, remove_selected_columns = TRUE)


# Rimuovere le colonne delle variabili di zona esistenti
#mod3_data <- mod3_data %>%
 # select(-c(zona_2, zona_3, zona_4, zona_5, zona_6, zona_7, zona_8, zona_9))
mod3_data <- as.data.frame(model.matrix( ~ ., data = mod3))




# select sample
sample3 <- mod3_data %>% sample_n(2000)
cols <- colnames(sample3 %>% select(-log_price, -room_type, -`(Intercept)`))
xmatrix3 <- as.matrix(sample3 %>% select(-log_price, -room_type, -`(Intercept)`))
stan_data3 <- list(N = nrow(xmatrix3),
                   J = max(mod3_data$room_type),
                   K = ncol(xmatrix3),
                   room_type = sample3$room_type,
                   X = xmatrix3,
                   y = sample3$log_price)

model3 <- stan(data = stan_data3, 
               file = here("Desktop/lore/code/airbnb_model.stan"),
               iter = 2000,
               seed = 8)
# save model with everything
saveRDS(model3, "fit3.rds")

# load in the model
model3 <- readRDS("fit3.rds")
max(summary(model3)$summary[,c("Rhat")])

# traceplots - alphas
pars = c(paste0("alpha[", 1:4, "]"))
traceplot(model3, pars=pars)

stan_dens(model3, separate_chains=TRUE, pars=pars)
pairs(model3, pars=pars)

set.seed(1856)
y3 <- sample3$log_price
yrep3 <- rstan::extract(model3)[["log_price_rep"]]
samp100_3 <- sample(nrow(yrep3), 100)

ppc_dens_3 <- ppc_dens_overlay(y3, yrep3[samp100_3, ])  + ggtitle("Distribution of Observed vs. Predicted Log Prices")
sample3 <- sample3 %>% mutate(room_names = case_when(room_type == 1 ~ "Entire Home/Apartment", 
                                                     room_type == 2 ~ "Hotel Room",
                                                     room_type == 3 ~ "Private Room",
                                                     room_type == 4 ~ "Shared Room"))
# test stats - median
ppc_stat_3 <- ppc_stat_grouped(sample3$log_price, yrep3, group = sample3$room_names, stat = 'median') + ggtitle("Median by Room Type - Model 3")

grid.arrange(ppc_dens_3,ppc_stat_3, ncol=1, nrow=2)

# extract results from summary
mod3_summary <- as.data.frame(summary(model3,probs = c(0.025, 0.975))$summary)
colnames(mod3_summary) <- c("mean", "se_mean", "sd", "ci_lower", "ci_upper", "n_eff", "rhat")

mod3_summary$row_name <- rownames(mod3_summary)

# keep betas
mod3_plot <- mod3_summary %>%
  filter(str_detect(row_name, 'beta')) 

# add room_type name
mod3_plot <- mod3_plot %>% mutate(variable = cols)

# plot results
ggplot(data = mod3_plot, aes(x = mean, y = variable)) + 
  geom_point() +
  geom_line() + 
  geom_linerange(aes(xmin = ci_lower, xmax = ci_upper), size=0.25) + 
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 30)) + 
  labs(y = "Variable", x = "Beta", 
       title = "Beta Coefficients", subtitle="With 95% CIs")

## Model 4 - Baseline Model (Linear Fit, all Covariates)

model4 <- stan(data = stan_data2, 
               file = here("Desktop/lore/code/airbnb_linear.stan"),
               iter = 2000,
               seed = 8)

# save model with everything
saveRDS(model4, "fit4.rds")

# load in the model
model4 <- readRDS("fit4.rds")
#max(summary(model4)$summary[,c("Rhat")])

traceplot(model4, pars="beta")

summary(model4)$summary

## Compare Models

loglik1 <- as.matrix(model1, pars="log_lik")
loglik2 <- as.matrix(model2, pars="log_lik")
loglik3 <- as.matrix(model3, pars="log_lik")
loglik4 <- as.matrix(model4, pars="log_lik")

loo1 <- loo(loglik1, save_psis=TRUE)
loo2 <- loo(loglik2, save_psis=TRUE)
loo3 <- loo(loglik3, save_psis=TRUE)
loo4 <- loo(loglik3, save_psis=TRUE)

### Model 2 vs. Model 3

kable(loo_compare(loo2,loo3))

### Model 2 vs. Model 4

kable(loo_compare(loo2,loo4))

### Model 3 vs. Model 4

kable(loo_compare(loo3,loo4))



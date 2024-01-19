library(ggplot2)
library(dplyr)
library(ggpubr)
library(car)
library(tidyverse)
library(rstatix)

# -----------------------------------------------------------------------------
# DATA PREPROCESSING

megnet_cnn1d_data <- read.table(file = "mixed_sample_accuracies.csv", sep = ",", header = T)
cnn2d_data <- read.table(file = "MEG-CNN-2D-20samp-accuracies.csv", sep = ",", header = T)

# split/collect data
megnet_data <- megnet_cnn1d_data[1]
cnn1d_data <- megnet_cnn1d_data[2]
cnn2d_data <- cnn2d_data[2]
lstm_data <- c(0.7142857142857143, 0.5714285714285714, 0.8571428571428571, 
               0.7142857142857143, 0.8571428571428571, 0.5714285714285714, 
               0.7142857142857143, 0.7142857142857143, 1.0, 
               0.7142857142857143, 1.0, 1.0, 0.7142857142857143, 
               0.8571428571428571, 0.7142857142857143, 0.7142857142857143, 
               0.5714285714285714, 0.42857142857142855, 0.7142857142857143, 
               0.7142857142857143)

# make dataframe
data <- data.frame(
  megnet_cnn1d_data[1],
  megnet_cnn1d_data[2],
  cnn2d_data,
  lstm_data
  )
colnames(data) <- c("MEGNet", "MEG-CNN-1D", "MEG-CNN-2D", "LSTM")

# create df for ANOVA etc.
all_accuracies <- c(data$MEGNet, data$`MEG-CNN-1D`, data$`MEG-CNN-2D`, data$LSTM)
model_type <- rep(c("MEGNet", "MEG-CNN-1D", "MEG-CNN-2D", "LSTM"), each = 20)

anovadata <- data.frame(
  all_accuracies,
  as.factor(model_type)
)
colnames(anovadata) <- c("accuracy", "model")


# -----------------------------------------------------------------------------
# CHECK ANOVA ASSUMPTIONS
anovadata %>%
  group_by(model) %>%
  identify_outliers(accuracy)

# 1. Normality - check normality per variable // non-significant = normal
shapiro.test(data$MEGNet) # p = .057
shapiro.test(data$`MEG-CNN-1D`) # p = .454
shapiro.test(data$`MEG-CNN-2D`) # p = .361
shapiro.test(data$`LSTM`) # p = .027

# 2. Homogeneity of variance // non-significant = good
leveneTest(y = anovadata$accuracy, anovadata$model)

# 3. Independence of samples
# done

# -----------------------------------------------------------------------------
# ANOVA TEST
oneway.test(accuracy ~ model, data = anovadata)

# -----------------------------------------------------------------------------
# POST HOC
games_howell_test(data = anovadata, accuracy ~ model)

# -----------------------------------------------------------------------------
# PLOTS

# boxplots
ggplot(anovadata, aes(x=model, y=accuracy, color=model)) +
  geom_boxplot() +
  ylim(.5, 1)

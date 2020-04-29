# Importando el dataset
dataset = read.csv('Social_Network_Ads.csv')
dataset = dataset[2:5]

# Arreglando campos vacios
dataset$Age = ifelse(is.na(dataset$Age),
                     ave(dataset$Age, FUN = function(x) mean(x, na.rm = TRUE)),
                     dataset$Age)

# Codificando variable Purchased
dataset$Purchased = factor(dataset$Purchased, levels = c(0, 1))

# Codificando variable Gender
dataset$Gender = factor(dataset$Gender,
                        levels = c('Male', 'Female'),
                        labels = c(1, 0))

# Dividiendo dataset en training y test
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Purchased, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Normalizando
training_set[2:3] = scale(training_set[2:3])
test_set[2:3] = scale(test_set[2:3])

# Regresion Logistica set1
classifier = glm(formula = Purchased ~ .,
                 family = binomial,
                 data = training_set)

# Prediciendo resultados con test
prob_pred = predict(classifier, type = 'response', newdata = test_set[1:3])
y_pred = ifelse(prob_pred > 0.5, 1, 0)

# Matriz de cofusion
cm = table(test_set[, 4], y_pred > 0.5)

# Visualizaciones
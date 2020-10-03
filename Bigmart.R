library(dplyr)
library(readr)
library(ggplot2)
library(caret)
library(data.table)
library(cowplot)
library(e1071)
library(ranger)

#read data
train = fread("D:/DSE/R/Datasets/train.csv")

test = fread("D:/DSE/R/Datasets/test.csv")


head(train)
head(test)

dim(train)
dim(test)

#structure of data
str(train)

#summary statistics
summary(train)

#colnames of data
colnames(train)

test[,Item_Outlet_Sales := NA]

df = rbind(train, test) # combining train and test datasets
dim(df)


head(df)


#visualization
#univariate analysis of target variable
qplot(df$Item_Outlet_Sales, geom="histogram",colour=I("red"),fill=I('darkgreen'),binwidth=500, xlab = "Item Outlet Sales", ylab = "Count")
hist(train$Item_Outlet_Sales)

#Independent variable analysis
qplot(df$Item_Weight, geom="histogram", binwidth=0.5, xlab = "Item Weight", ylab = "Count")
qplot(df$Item_Visibility, geom= "histogram",binwidth=0.005, fill=I('blue'), xlab = "Item visibility", ylab = "Count")
qplot(df$Item_MRP, geom = "histogram", fill=I('red'),binwidth=1, xlab = "Item MRP", ylab = "Count")


#Count plot of Item_Fat_Content
ggplot(df %>% group_by(Item_Fat_Content) %>% summarise(Count = n())) + 
  geom_bar(aes(Item_Fat_Content, Count), stat = "identity", fill = "coral2")

df$Item_Fat_Content[df$Item_Fat_Content=="LF"] = "Low Fat"
df$Item_Fat_Content[df$Item_Fat_Content=="low fat"] = "Low Fat"
df$Item_Fat_Content[df$Item_Fat_Content=="reg"] = "Regular"

#new plot after combining data
ggplot(df %>% group_by(Item_Fat_Content) %>% summarise(Count = n())) + 
  geom_bar(aes(Item_Fat_Content, Count), stat = "identity", fill = "coral3")

#null value Analysis
colSums(is.na(df))

#imputing Item_weight values
missing_index = which(is.na(df$Item_Weight))

for(i in missing_index){
  
  item = df$Item_Identifier[i]
  df$Item_Weight[i] = mean(df$Item_Weight[df$Item_Identifier == item],na.rm = T)
}

colSums(is.na(df))

#replacing zero in item variable
zero_index = which(df$Item_Visibility == 0)
for(i in zero_index){
  
  item = df$Item_Identifier[i]
  df$Item_Visibility[i] = mean(df$Item_Visibility[df$Item_Identifier == item], na.rm = T)
  
}

df %>% group_by(Item_Type) %>% summarise(Count = n())


#feature engineering creating new features
perishable = c("Breads","Breakfast","Dairy","Fruits and Vegetables","Meat","Seafood")

non_perishable = c("Baking Goods","Canned","Frozen Foods","Hard Drinks","Health and Hygiene","Household","Soft Drinks")

df[,Item_Type_new := ifelse(Item_Type %in% perishable, "perishable",
                            ifelse(Item_Type %in% non_perishable, "non_perishable", "not_sure"))]


#comapring with item_identifier
table(df$Item_Type, substr(df$Item_Identifier,1,2))

#creating new feature item_category
df[,Item_Category := substr(df$Item_Identifier,1,2)]

head(df)

#changing values of FatContent to item_category="NC"
df$Item_Fat_Content[df$Item_Category=="NC"] = "Non-Edible"

#creating years of operation
df[,Outlet_Years := 2013 - Outlet_Establishment_Year]

df$Outlet_Establishment_Year = as.factor(df$Outlet_Establishment_Year)

head(df)

#creating price_per_unit
df[,Price_per_unit:= Item_MRP/Item_Weight]

#MRP clusters
Item_MRP_clusters = kmeans(df$Item_MRP, centers = 4)
table(Item_MRP_clusters$cluster) # display no. of observations in each cluster


df$Item_MRP_clusters = as.factor(Item_MRP_clusters$cluster)


head(df)


#Encoding Categorical variables
#label encode Outlet_Size and Outlet_Location_Type as these are ordinal variables.
df[,Outlet_Size_num := ifelse(Outlet_Size=="Small",0,
                              ifelse(Outlet_Size=="Medium",1,2))]

df[,Outlet_Location_Type_num := ifelse(Outlet_Location_Type == "Tier 3", 0,
                                       ifelse(Outlet_Location_Type == "Tier 2", 1, 2))]

head(df)
df[,c("Outlet_Size","Outlet_Location_Type"):=NULL]

head(df)

#Onehotencoding
onehot = dummyVars("~.",data = df[,-c("Item_Identifier","Outlet_Establishment_Year",
                                   "Item_Type")],fullRank = T)

onehot_df = data.table(predict(onehot,df[,-c("Item_Identifier","Outlet_Establishment_Year",
                                             "Item_Type")]))

onehot_df

df = cbind(df[,"Item_Identifier"],onehot_df)

head(df)

dim(df)


#data preprocessing
#skew
skewness(df$Item_Visibility)
skewness(df$Price_per_unit)

df[,Item_Visibility := log(Item_Visibility + 1)] # log + 1 to avoid division by zero
df[,price_per_unit_wt := log(Price_per_unit + 1)]


num = which(sapply(df, is.numeric))
num_var_names = names(num)
df_numeric = df[,setdiff(num_var_names,"Item_Outlet_Sales"),with=F]
prep_num = preProcess(df_numeric, method=c("center","scale"))
df_numeric_norm = predict(prep_num,df_numeric)

df[,setdiff(num_var_names, "Item_Outlet_Sales") := NULL] # removing numeric independent variables
df = cbind(df, df_numeric_norm)

train = df[1:nrow(train)]
test = df[(nrow(train) + 1):nrow(df)]
test[,Item_Outlet_Sales := NULL] # removing Item_Outlet_Sales as it contains only NA

head(train)

#corplot
cor_train = cor(train[,-c("Item_Identifier")])
#corrplot(cor_train, method = "pie", type = "lower", tl.cex = 0.9)

#1. Linear Regression
lin_regressor = lm(formula = Item_Outlet_Sales~., data = train[,-c("Item_Identifier")])

summary(lin_regressor)

submission$Item_Outlet_Sales = predict(lin_regressor, test[,-c("Item_Identifier")])

colnames(train) = make.names(colnames(train))
colnames(test) = make.names(colnames(test))

head(train)





# create a list of 80% of the rows in the original dataset we can use for training
validation_index <- createDataPartition(train$Item_Outlet_Sales, p=0.80, list=FALSE)
# select 20% of the data for validation
validation <- train[-validation_index,]
# use the remaining 80% of data to training and testing the models
train <- train[validation_index,]

dim(train)
dim(validation)

model = lm(Item_Outlet_Sales ~ ., data = train[,-c("Item_Identifier","Outlet_TypeSupermarket.Type1","Outlet_TypeSupermarket.Type2","Outlet_TypeSupermarket.Type3","Item_CategoryNC","Outlet_Years","Outlet_Size_num","Outlet_Location_Type_num")])
summary(model)

#prediction
validation_f<-select(validation, -c("Item_Identifier","Outlet_TypeSupermarket.Type1","Outlet_TypeSupermarket.Type2","Outlet_TypeSupermarket.Type3","Item_CategoryNC","Outlet_Years","Outlet_Size_num","Outlet_Location_Type_num"))
pred=predict.lm(model,validation_f)

actuals_preds1 <- data.frame(cbind(actuals=validation_f$Item_Outlet_Sales, predicteds=pred))
actuals_preds1
residual_lm = actuals_preds1$actuals - actuals_preds1$predicteds
RMSE_lm  = sqrt(sum(residual_lm**2)/length(residual_lm))


# plot the results
ggplot(actuals_preds1, aes(x=actuals,y=predicteds)) + 
  geom_point() +
  geom_abline(color = "blue")


(bm_rf <- ranger(Item_Outlet_Sales ~ ., data = train[,-c("Item_Identifier","Outlet_TypeSupermarket.Type1","Outlet_TypeSupermarket.Type2","Outlet_TypeSupermarket.Type3","Item_CategoryNC","Outlet_Years","Outlet_Size_num","Outlet_Location_Type_num")], # data
                         num.trees = 500, 
                         respect.unordered.factors = "order", 
                         seed = set.seed(1234)))

pred_rf = predict(bm_rf,validation_f)
actuals_preds_rf <- data.frame(cbind(actuals=validation_f$Item_Outlet_Sales, predicteds=pred_rf$predictions))
residual_rf = actuals_preds_rf$actuals - actuals_preds_rf$predicteds
RMSE_rf  = sqrt(sum(residual_rf**2)/length(residual_rf)) 

RMSE_rf

plot(varImp(bm_rf))

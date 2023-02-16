library(bruceR)
library(psych)
library(patchwork)
set.wd()
set.wd("../")
emotions = import("Experiment/emotions.txt", header = FALSE)[1:27,1]

## preprocess affective features ----
# load data
rawdata.accept = import("Data/raw_data/emotional_features/affective_features/rawdata.csv")
# filter and export bulk approved list
index.reject = (rawdata.accept[,"Finished"]=="False")|
  (rawdata.accept[,"UserLanguage"]!="EN")|
  (rawdata.accept[,"Consent"]!="I agree")|
  (rawdata.accept[,"28_Arousal_1"]!="9")|
  (rawdata.accept[,"28_Valence_1"]!="9")
rawdata.accept = rawdata.accept[!index.reject,]
export(list(rawdata.accept[,"Prolific ID"]),
       file = "Data/raw_data/emotional_features/affective_features/approve.csv")
rm(index.reject)
# reorganize data
rawdata.accept = rawdata.accept[,4:ncol(rawdata.accept)]
# individual score
ind.affective = data.frame()
n = 0
for (c in 1:nrow(rawdata.accept)){
  for (e in 1:27){
    n = n+1
    ind.affective[n,"subject"] = rawdata.accept$`Prolific ID`[c]
    ind.affective[n,"emotion"] = emotions[e]
    ind.affective[n,"arousal"] = rawdata.accept[c,(e-1)*2+2]
    ind.affective[n,"valence"] = rawdata.accept[c,(e-1)*2+3]
  }
}
ind.affective[,"emotion"] = factor(ind.affective[,"emotion"], levels = emotions)
rm(n,c,e,rawdata.accept)
# average score
avg.affective = ind.affective |> 
  group_by(ind.affective[,"emotion"]) |>
  summarise_at(
    3:ncol(ind.affective),
    list(avg = ~mean(.)))
avg.affective = data.frame(avg.affective[,2:3], row.names = emotions)
colnames(avg.affective) = colnames(ind.affective)[3:4]
for (i in c(1:ncol(avg.affective))) {
  avg.affective[,i] = RESCALE(avg.affective[,i],to=0:1)
}
# distance
temp = as.matrix(dist(avg.affective,method = "euclidean"))
dist.affective = temp[lower.tri(temp)]
rm(temp)

## preprocess physical features----
rawdata.accept = import("Data/raw_data/emotional_features/physical_features/rawdata.csv")
# filter and export bulk approved list
index.reject = (rawdata.accept[,"Finished"]=="False")|
  (rawdata.accept[,"UserLanguage"]!="EN")|
  (rawdata.accept[,"Consent"]!="I agree")|
  (rawdata.accept[,"28_Happy_1"]!=9)|
  (rawdata.accept[,"28_Anger _1"]!=9)|
  (rawdata.accept[,"28_Sad _1"]!=9)|
  (rawdata.accept[,"28_Fear_1"]!=9)|
  (rawdata.accept[,"28_Surprise_1"]!=9)|
  (rawdata.accept[,"28_Disgust_1"]!=9)
rawdata.accept = rawdata.accept[!index.reject,]
export(list(rawdata.accept[,"PROLIFIC_PID"]),
       file = "Data/raw_data/emotional_features/physical_features/approve.csv")
rm(index.reject)
# reorganize data
rawdata.accept = rawdata.accept[,13:(ncol(rawdata.accept)-7)]
# individual score
ind.physical = data.frame()
n = 0
for (c in 1:nrow(rawdata.accept)){
  for (e in 1:27){
    n = n+1
    ind.physical[n,"subject"] = rawdata.accept[c,1]
    ind.physical[n,"emotion"] = emotions[e]
    ind.physical[n,"happy"] = as.numeric(rawdata.accept[c,(e-1)*6+2])
    ind.physical[n,"anger"] = as.numeric(rawdata.accept[c,(e-1)*6+3])
    ind.physical[n,"sad"] = as.numeric(rawdata.accept[c,(e-1)*6+4])
    ind.physical[n,"fear"] = as.numeric(rawdata.accept[c,(e-1)*6+5])
    ind.physical[n,"surprise"] = as.numeric(rawdata.accept[c,(e-1)*6+6])
    ind.physical[n,"disgust"] = as.numeric(rawdata.accept[c,(e-1)*6+7])
  }
}
ind.physical[,"emotion"] = factor(ind.physical[,"emotion"], levels = emotions)
rm(n,c,e,rawdata.accept)
# average score
avg.physical = ind.physical |> 
  group_by(ind.physical[,"emotion"]) |>
  summarise_at(
    3:ncol(ind.physical),
    list(avg = ~mean(.)))
avg.physical = data.frame(avg.physical[,2:7], row.names = emotions)
colnames(avg.physical) = colnames(ind.physical)[3:8]
for (i in c(1:ncol(avg.physical))) {
  avg.physical[,i] = RESCALE(avg.physical[,i],to=0:1)
}
# distance
temp = as.matrix(dist(avg.physical,method = "euclidean"))
dist.physical = temp[lower.tri(temp)]
rm(temp)

## preprocess appraisal features----
rawdata.accept = import("Data/raw_data/emotional_features/appraisal_features/rawdata.csv")
rawdata.accept = rawdata.accept[3:nrow(rawdata.accept),]
# filter and export bulk approved list
index.reject = (rawdata.accept[,"Finished"]=="False")|
  (rawdata.accept[,"UserLanguage"]!="EN")|
  (rawdata.accept[,"Consent"]!="I agree")|
  (rawdata.accept[,"42_Item_1"]!="4")|
  (rawdata.accept[,"43_Item_1"]!="-99")|
  (rawdata.accept[,"Specific Event"]=="n/a")
rawdata.accept = rawdata.accept[!index.reject,]
rm(index.reject)
# reorganize data
item.all = read.table("Experiment/appraisal_features/items_name.txt", header = FALSE, sep = "+", col.names = "name")
item.all[,"description"] = read.table("Experiment/appraisal_features/items_description.txt", header = FALSE, sep = "+")
rawdata.accept = rawdata.accept[,c(1,4,6:(ncol(rawdata.accept)-2))]
colnames(rawdata.accept) = c("duration","id","emotion","recall","specific.event",paste('item',1:38,sep=''))
rownames(rawdata.accept) = c(1:nrow(rawdata.accept))
rawdata.accept[,"emotion"] = tolower(rawdata.accept[,"emotion"])
rawdata.accept[,6:ncol(rawdata.accept)] = as.numeric(unlist(rawdata.accept[,6:ncol(rawdata.accept)]))
rawdata.accept[rawdata.accept==-99] = NA
rawdata.accept[,"emotion"] = factor(rawdata.accept[,"emotion"], levels = emotions)
summary(rawdata.accept[,"emotion"])
# individual score
ind.appraisal = rawdata.accept[,c(3,6:ncol(rawdata.accept))]
ind.appraisal[is.na(ind.appraisal)] = 5
colnames(ind.appraisal)[2:39] = item.all[,"name"]
rm(rawdata.accept)
# PCA (Principal Component Analysis)-- Rotation Method: None
PCA.result = bruceR::PCA(ind.appraisal,vars=colnames(ind.appraisal)[2:39],
                         #rotation="none",
                         #nfactors=38,
                         nfactors="parallel",
                         file="Result/supplement_information/PCA_component_loadings_for_appraisal_features")
PC.loading = PCA.result$loadings[,c(1:(ncol(PCA.result$loadings)-1))]
colnames(PC.loading) = paste("PC",c(1:ncol(PC.loading)),sep = "")
ind.PCs = data.frame(emotion=ind.appraisal[,1], PCA.result[["result"]][["scores"]])
# average PCs score
avg.PCs = ind.PCs |> 
  group_by(ind.PCs[,"emotion"]) |>
  summarise_at(
    2:ncol(ind.PCs),
    list(avg = ~mean(.)))
avg.PCs = data.frame(avg.PCs[,2:ncol(avg.PCs)], row.names = emotions)
colnames(avg.PCs) = paste("PC",c(1:ncol(avg.PCs)),sep = "")
rm(ind.appraisal)
for (i in c(1:ncol(avg.PCs))) {
  avg.PCs[,i] = RESCALE(avg.PCs[,i],to=0:1)
}
# distance
temp = as.matrix(dist(avg.PCs,method = "euclidean"))
dist.appraisal = temp[lower.tri(temp)]
rm(temp)

## exclude features by ICC ----------
# ICC(1,k) for PCs score
icc.results = data.frame("item"=c(colnames(avg.affective),
                                  colnames(avg.physical),
                                  colnames(avg.PCs)),
                         "ICC type"=character(ncol(avg.affective)+ncol(avg.physical)+ncol(avg.PCs)),
                         "ICC coef"=numeric(ncol(avg.affective)+ncol(avg.physical)+ncol(avg.PCs)),
                         "F"=numeric(ncol(avg.affective)+ncol(avg.physical)+ncol(avg.PCs)),
                         "df1"=numeric(ncol(avg.affective)+ncol(avg.physical)+ncol(avg.PCs)),
                         "df2"=numeric(ncol(avg.affective)+ncol(avg.physical)+ncol(avg.PCs)),
                         "p"=numeric(ncol(avg.affective)+ncol(avg.physical)+ncol(avg.PCs)),
                         "lower bound"=numeric(ncol(avg.affective)+ncol(avg.physical)+ncol(avg.PCs)),
                         "upper bound"=numeric(ncol(avg.affective)+ncol(avg.physical)+ncol(avg.PCs)))
colnames(icc.results)[c(2:3,8:9)] = c("ICC type","ICC coef","lower bound","upper bound")
# ICC(2,k) for affective features
for (i in c(1:2)) {
  iccdata = ICC(matrix(ind.affective[,i+2], nrow = 27), lmer = F)
  icc.results[i,"ICC type"] = "ICC(2,k)"
  icc.results[i,"ICC coef"] = iccdata[["results"]][["ICC"]][5]
  icc.results[i,"F"] = iccdata[["results"]][["F"]][5]
  icc.results[i,"df1"] = iccdata[["results"]][["df1"]][5]
  icc.results[i,"df2"] = iccdata[["results"]][["df2"]][5]
  icc.results[i,"p"] = iccdata[["results"]][["p"]][5]
  icc.results[i,"lower bound"] = iccdata[["results"]][["lower bound"]][5]
  icc.results[i,"upper bound"] = iccdata[["results"]][["upper bound"]][5]
}
# ICC(2,k) for physical features
for (i in c(1:6)) {
  iccdata = ICC(matrix(data = ind.physical[,2+i], nrow = 27), lmer = F)
  icc.results[2+i,"ICC type"] = "ICC(2,k)"
  icc.results[2+i,"ICC coef"] = iccdata[["results"]][["ICC"]][5]
  icc.results[2+i,"F"] = iccdata[["results"]][["F"]][5]
  icc.results[2+i,"df1"] = iccdata[["results"]][["df1"]][5]
  icc.results[2+i,"df2"] = iccdata[["results"]][["df2"]][5]
  icc.results[2+i,"p"] = iccdata[["results"]][["p"]][5]
  icc.results[2+i,"lower bound"] = iccdata[["results"]][["lower bound"]][5]
  icc.results[2+i,"upper bound"] = iccdata[["results"]][["upper bound"]][5]
}
# ICC(1,k) for PCs of appraisal
for (i in 1:ncol(avg.PCs)) {
  iccdata = matrix(nrow = 27,ncol = 12)
  for (e in 1:27){
    iccdata[e,1:sum(ind.PCs$emotion==emotions[e])] = ind.PCs[ind.PCs$emotion==emotions[e],i+1]
  }
  iccdata = ICC(iccdata, missing = F, lmer = F)
  icc.results[i+8,"ICC type"] = "ICC(1,k)"
  icc.results[i+8,"ICC coef"] = iccdata[["results"]][["ICC"]][4]
  icc.results[i+8,"F"] = iccdata[["results"]][["F"]][4]
  icc.results[i+8,"df1"] = iccdata[["results"]][["df1"]][4]
  icc.results[i+8,"df2"] = iccdata[["results"]][["df2"]][4]
  icc.results[i+8,"p"] = iccdata[["results"]][["p"]][4]
  icc.results[i+8,"lower bound"] = iccdata[["results"]][["lower bound"]][4]
  icc.results[i+8,"upper bound"] = iccdata[["results"]][["upper bound"]][4]
}
rm(iccdata,e,i)
# FDR correct
icc.results$p = icc.results$p * nrow(icc.results) / rank(icc.results$p)
significant = vector(length = nrow(icc.results))
significant[icc.results$p<0.025] = TRUE
index.significant = as.numeric(rownames(icc.results))[significant]

## save results -----
index.dist = unlist(import("Data/preprocessed/reorder_distance.csv"))
# combine three spaces
avg.all = cbind(avg.affective,avg.physical,avg.PCs)
temp = as.matrix(dist(avg.all,method = "euclidean"))
dist.all = temp[lower.tri(temp)]
rm(temp)
# save distance
dist.space = data.table(affective=dist.affective,
                        basic=dist.physical,
                        appraisal=dist.appraisal,
                        all=dist.all)
colnames(dist.space) = c("Affective Space","Basic Emotions Space","Appraisal Space","Combined Space")
dist.space = dist.space[index.dist,]
export(dist.space,"Data/preprocessed/emotional_features/dist_emotion_space.csv")
# save dist correlation
corr.dist = cor(dist.space)

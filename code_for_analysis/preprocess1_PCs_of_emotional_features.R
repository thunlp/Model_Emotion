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
# distance
dist.affective = matrix(nrow = 351, ncol = ncol(avg.affective))
for (Ncol in 1:ncol(avg.affective)) {
  temp = as.matrix(dist(avg.affective[,Ncol],method = "euclidean"))
  dist.affective[,Ncol] = temp[lower.tri(temp)]
}
colnames(dist.affective) = colnames(avg.affective)
rm(temp,Ncol)

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
# distance
dist.physical = matrix(nrow = 351, ncol = ncol(avg.physical))
for (Ncol in 1:ncol(avg.physical)) {
  temp = as.matrix(dist(avg.physical[,Ncol],method = "euclidean"))
  dist.physical[,Ncol] = temp[lower.tri(temp)]
}
colnames(dist.physical) = colnames(avg.physical)
rm(temp,Ncol)

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
# distance
dist.PCs = matrix(nrow = 351, ncol = ncol(avg.PCs))
for (Ncol in 1:ncol(avg.PCs)) {
  temp = as.matrix(dist(avg.PCs[,Ncol],method = "euclidean"))
  dist.PCs[,Ncol] = temp[lower.tri(temp)]
}
colnames(dist.PCs) = colnames(avg.PCs)
rm(temp,Ncol)

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
# rename valid PCs
index.significant = sort(index.significant[9:length(index.significant)]-8) #before sort: 1,5,3,4,7,2,22,6,11
valide.PCs = c("control",
               "fairness",
               "self-related",
               "other-related",
               "expectedness",
               "non-novelty")
icc.results[9:nrow(icc.results),"item"] = valide.PCs
icc.results$item = factor(icc.results$item, levels = icc.results$item)
# valide.PCs = c("control",
#                "self emergency",
#                "others dominate",#caused by agent, intentional action
#                "novelty",
#                "others affair",#caused by agent
#                "suddenness",
#                "danger",
#                "prospection",#future
#                "assistance")#coping, -remembering, -danger, bodily disease, -morality, relationship influence, self esteem,
colnames(PC.loading)[index.significant] = valide.PCs#paste(colnames(PC.loading)[index.significant],"(",valide.PCs,")")
colnames(avg.PCs)[index.significant] = valide.PCs
colnames(dist.PCs)[index.significant] = valide.PCs

# save score
index.emotions = unlist(import("Data/preprocessed/reorder_emotions.csv"))
emotions = emotions[index.emotions]
avg.PCs = avg.PCs[,index.significant]
emotion.score = cbind(avg.affective,avg.physical,avg.PCs)
emotion.score = emotion.score[index.emotions,]
for (i in c(1:ncol(emotion.score))) {
  emotion.score[,i] = RESCALE(emotion.score[,i],to=0:1)
}
export(emotion.score,"Data/preprocessed/emotional_features/score_emotional_PCs.csv")
plotdata = expand.grid(emotion=rownames(emotion.score),features=colnames(emotion.score))
plotdata[,"score"] = unlist(emotion.score)
plotdata[,"type"] = factor(c(rep("Affective Dimensions",2*27),
                             rep("Basic Emotions",6*27),
                             rep("Appraisal Dimensions",6*27)),
                           levels = c("Affective Dimensions",
                                      "Basic Emotions",
                                      "Appraisal Dimensions"))
p1 = ggplot(plotdata, aes(x=features,y=emotion,fill=score,group=type))+
  geom_tile()+
  scale_fill_continuous(type = "viridis",name="average\nscore",breaks=c(0,1),labels=c("min","max"))+
  scale_x_discrete(name="",position="top",guide=guide_axis(angle = 90))+
  scale_y_discrete(name="",limits=rev)+
  facet_grid(~type,scales = "free_x",space = "free_x")+
  theme(panel.background=element_blank(),
        panel.grid = element_blank(),
        axis.ticks.length=unit(0,units = "mm"),
        strip.background = element_blank(),
        strip.text.x = element_blank())
ggsave("Result/supplement_information/averaged_emotional_PCs.pdf", plot = p1, width = 180, height = 210, units = "mm")
rm(plotdata,i)

# save ICC results
p2 = ggplot(cbind(icc.results,significant))+
  geom_point(aes(x=item, y=`ICC coef`, color=significant), stat = "identity")+
  geom_errorbar(aes(x=item, ymin=`lower bound`, ymax=`upper bound`), width = 0.3)+
  scale_x_discrete(name = "",guide=guide_axis(angle = 90))+
  scale_y_continuous(name = "intraclass correlation coefficient", breaks = c(seq(from=-1,to=1,by=0.2)))+
  scale_color_manual(breaks = c(TRUE,FALSE), labels = c("significant","not significant"), values = c("#5ec962", "#794B48"))+
  theme_classic()+
  theme(legend.position='none')
ggsave("Result/supplement_information/ICC_for_emotional_PCs.pdf", plot = p2, width = 180, height =185, units = "mm")
rm(ind.PCs,ind.affective,ind.physical)

# save PC loadings
plotdata = expand.grid(appraisal=rownames(PC.loading),PC=colnames(PC.loading))
plotdata[["loading"]] = unlist(PC.loading)
p3 = ggplot(plotdata, aes(x=PC,y=appraisal,fill=loading))+
  geom_tile()+
  scale_fill_gradient2()+
  scale_x_discrete(name="",position="top",guide=guide_axis(angle = 90))+
  scale_y_discrete(name="",limits=rev)+
  theme(panel.background=element_blank(), panel.grid = element_blank(), axis.ticks.length=unit(0,units = "mm"))
ggsave("Result/supplement_information/PC_loadings_of_appraisal_features.pdf",plot = p3, width = 180, height = 185, units = "mm")
rm(plotdata)

# save score correlation
corr.score = cor(emotion.score)
plotdata = expand.grid(emotion1=rownames(corr.score),emotion2=colnames(corr.score))
plotdata[["r"]] = as.vector(corr.score)
library(corrplot)
corrplot(corr.score,method = "color",col = rev(COL2("RdBu",200)),tl.col = "black")
p4 = ggplot(plotdata, aes(x=emotion1,y=emotion2,fill=r))+
  geom_tile()+
  scale_fill_gradient2(low = "blue", high = "red", mid = "grey")+
  scale_x_discrete(name="",position = "top",guide = guide_axis(angle=90))+
  scale_y_discrete(name="",limits=rev)+
  theme(panel.background=element_blank(), panel.grid = element_blank(), axis.ticks.length=unit(0,units = "mm"))

# save distance
index.dist = unlist(import("Data/preprocessed/reorder_distance.csv"))
dist.PCs = dist.PCs[,index.significant]
dist.features = cbind(dist.affective,dist.physical,dist.PCs)
dist.features = dist.features[index.dist,]
export(dist.features,"Data/preprocessed/emotional_features/dist_emotional_PCs.csv")

# merge plot
#p3 = p3+
#  scale_x_discrete(name="",position="bottom",guide=guide_axis(angle = 90))
design = "
2
2
2
1
1"
p = p1+p3+plot_layout(design = design)
ggsave("Result/supplement_information/emotion_score.pdf", plot=p, width=180, height=230, units="mm")

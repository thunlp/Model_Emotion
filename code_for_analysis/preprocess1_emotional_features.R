library(bruceR)
library(psych)
library(patchwork)
library(boot)
set.wd()
set.wd("../")
emotions = import("Experiment/emotions.txt", header = FALSE)[1:27,1]

## preprocess affective features
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

## preprocess physical features
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

## preprocess appraisal features
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
export(list(rawdata.accept[,"PROLIFIC_PID"]),
       file = "Data/raw_data/emotional_features/appraisal_features/approve.csv")
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
# average score
avg.appraisal = ind.appraisal |> 
  group_by(ind.appraisal[,"emotion"]) |>
  summarise_at(
    2:ncol(ind.appraisal),
    list(avg = ~mean(.)))
avg.appraisal = data.frame(avg.appraisal[,2:39], row.names = emotions)
colnames(avg.appraisal) = item.all[,"name"]
# distance
dist.appraisal = matrix(nrow = 351, ncol = ncol(avg.appraisal))
for (Ncol in 1:ncol(avg.appraisal)) {
  temp = as.matrix(dist(avg.appraisal[,Ncol],method = "euclidean"))
  dist.appraisal[,Ncol] = temp[lower.tri(temp)]
}
colnames(dist.appraisal) = colnames(avg.appraisal)
rm(temp,Ncol)

## exclude features by ICC
icc.results = data.frame("item"=c(colnames(ind.affective)[3:4],
                                colnames(ind.physical)[3:8],
                                colnames(ind.appraisal)[2:39]),
                       "ICC type"=character(46),
                       "ICC coef"=numeric(46),
                       "F"=numeric(46),
                       "df1"=numeric(46),
                       "df2"=numeric(46),
                       "p"=numeric(46),
                       "lower bound"=numeric(46),
                       "upper bound"=numeric(46))
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
# ICC(1,k) for appraisal features
for (i in 1:38) {
  iccdata = matrix(nrow = 27,ncol = 12)
  for (e in 1:27){
    iccdata[e,1:sum(ind.appraisal$emotion==emotions[e])] = ind.appraisal[ind.appraisal$emotion==emotions[e],i+1]
  }
  iccdata = ICC(iccdata, missing = F, lmer = F)
  icc.results[8+i,"ICC type"] = "ICC(1,k)"
  icc.results[8+i,"ICC coef"] = iccdata[["results"]][["ICC"]][4]
  icc.results[8+i,"F"] = iccdata[["results"]][["F"]][4]
  icc.results[8+i,"df1"] = iccdata[["results"]][["df1"]][4]
  icc.results[8+i,"df2"] = iccdata[["results"]][["df2"]][4]
  icc.results[8+i,"p"] = iccdata[["results"]][["p"]][4]
  icc.results[8+i,"lower bound"] = iccdata[["results"]][["lower bound"]][4]
  icc.results[8+i,"upper bound"] = iccdata[["results"]][["upper bound"]][4]
}
rm(iccdata,e,i)
# FDR correct
icc.results$p = icc.results$p * nrow(icc.results) / rank(icc.results$p)
icc.results = icc.results[c(1:8,order(icc.results$`ICC coef`,decreasing = T)[which(order(icc.results$`ICC coef`,decreasing = T)>8)]),]
significant = vector(length = nrow(icc.results))
significant[icc.results$p<0.025] = TRUE
icc.results$item = factor(icc.results$item, levels = icc.results$item)
index.significant = as.numeric(rownames(icc.results))[significant]
# save preprocessed data
avg.features = cbind(avg.affective,avg.physical,avg.appraisal)[index.significant]
index.emotions = hclust(as.dist(1-cor(t(avg.features), method = "pearson")), method = "complete")[["order"]]
emotions = emotions[index.emotions]
avg.features = avg.features[index.emotions,]
dist.features = cbind(dist.affective,dist.physical,dist.appraisal)
index.dist = matrix(0,nrow = 27,ncol = 27)
index.dist[lower.tri(index.dist)] = c(1:351)
index.dist = index.dist+t(index.dist)
index.dist = index.dist[index.emotions,index.emotions]
index.dist = index.dist[lower.tri(index.dist)]
dist.features = dist.features[index.dist,index.significant]
item.all = rbind(c("arousal","To what extend does EMOTION make you feel...(Very Calming/Very Arousing)"),
                 c("valence","To what extend does EMOTION make you feel...(Very Unpleasant/Very Pleasant)"),
                 c("happy","To what extend is consistent with the physiological response shown in the figures: (Very Inconsistent/Very Consistent)"),
                 c("anger","To what extend is consistent with the physiological response shown in the figures: (Very Inconsistent/Very Consistent)"),
                 c("sad","To what extend is consistent with the physiological response shown in the figures: (Very Inconsistent/Very Consistent)"),
                 c("fear","To what extend is consistent with the physiological response shown in the figures: (Very Inconsistent/Very Consistent)"),
                 c("surprise","To what extend is consistent with the physiological response shown in the figures: (Very Inconsistent/Very Consistent)"),
                 c("disgust","To what extend is consistent with the physiological response shown in the figures: (Very Inconsistent/Very Consistent)"),
                 item.all)[index.significant,]
export(list(index.emotions),"Data/preprocessed/reorder_emotions.csv")
export(list(index.dist),"Data/preprocessed/reorder_distance.csv")
export(list(emotions),"Data/preprocessed/reordered_emotions.csv")
export(avg.features,"Data/preprocessed/emotional_features/score_emotional_features.csv")
export(dist.features,"Data/preprocessed/emotional_features/dist_emotional_features.csv")
export(item.all,"Data/preprocessed/emotional_features/features_description.csv")
rm(avg.affective,avg.physical,avg.appraisal,dist.affective,dist.physical,dist.appraisal)

## additional statistic and visualization
# ICC result
p1 = ggplot(cbind(icc.results,significant))+
  geom_point(aes(x=item, y=`ICC coef`, color=significant), stat = "identity")+
  geom_errorbar(aes(x=item, ymin=`lower bound`, ymax=`upper bound`), width = 0.3)+
  scale_x_discrete(name = "",guide=guide_axis(angle = 90))+
  scale_y_continuous(name = "intraclass correlation coefficient", breaks = c(seq(from=-1,to=1,by=0.2)))+
  scale_color_manual(breaks = c(TRUE,FALSE), labels = c("significant","not significant"), values = c("#5ec962", "#794B48"))+
  theme_classic()+
  theme(legend.position='none')
ggsave("Result/supplement_information/ICC_for_emotional_features.pdf", plot = p1, width = 180, height =185, units = "mm")
rm(icc.results,significant)
# averaged score
plotdata = expand.grid(emotion=rownames(avg.features),features=colnames(avg.features))
plotdata[,"score"] = unlist(avg.features)
p2 = ggplot(plotdata, aes(x=features,y=emotion,fill=score))+
  geom_tile()+
  scale_fill_continuous(type = "viridis",breaks=c(1,3,5,7,9),limits=c(1,9))+
  scale_x_discrete(name="",position="top",guide=guide_axis(angle = 90))+
  scale_y_discrete(name="",limits=rev)+
  theme_minimal()+
  theme(legend.position='none')
ggsave("Result/supplement_information/averaged_emotional_features.pdf", plot = p2, width = 180, height = 185, units = "mm")
rm(p1,p2,plotdata)
# correlation between distances based on different number of subjects
boot.subject = function(subject.pool, index.s, dist.compared, subject.n) {
  if (subject.n==1) {
    score.temp = subject.pool[index.s[1],]
  }else{
    score.temp = colMeans(subject.pool[index.s[1:subject.n],],na.rm = T)
  }
  dist.temp = as.matrix(dist(score.temp, method = "euclidean"))
  dist.temp = dist.temp[lower.tri(dist.temp)]
  r = cor(dist.temp,dist.compared,method = "spearman")
  return(r)
}
boot.results = list()
for (feature.index in c(1:ncol(dist.features))) {
  feature.name = colnames(dist.features)[feature.index]
  dist.compared = dist.features[,feature.name]
  if (feature.index<=2) {
    subject.pool = t(matrix(ind.affective[,feature.name], nrow = 27))
    subject.pool = subject.pool[,index.emotions]
    boot.results[[feature.index]] = data.frame(n.subject=numeric(nrow(subject.pool)-1),
                                               r=numeric(nrow(subject.pool)-1),
                                               se=numeric(nrow(subject.pool)-1),
                                               ttest.p=numeric(nrow(subject.pool)-1))
    for (subject.n in c(1:(nrow(subject.pool)-1))) {
      set.seed(6)
      compare.temp = boot(subject.pool, boot.subject, R=100, sim = "permutation", dist.compared=dist.compared, subject.n=subject.n)
      boot.results[[feature.index]]$n.subject[subject.n] = subject.n
      boot.results[[feature.index]]$r[subject.n] = compare.temp$t0
      boot.results[[feature.index]]$se[subject.n] = sd(compare.temp$t)
      compare.last = compare.temp$t
      if (subject.n > 1) {
        boot.results[[feature.index]]$ttest.p[subject.n] = t.test(compare.last, compare.next, alternative = "greater", paired = F)$p.value
      }
      compare.next = compare.temp$t
      print(paste("feature.index = ",feature.index," subject.n = ",subject.n,sep = ""))
    } 
  }else if(feature.index<=8){
    subject.pool = t(matrix(ind.physical[,feature.name], nrow = 27))
    subject.pool = subject.pool[,index.emotions]
    boot.results[[feature.index]] = data.frame(n.subject=numeric(nrow(subject.pool)-1),
                                               r=numeric(nrow(subject.pool)-1),
                                               se=numeric(nrow(subject.pool)-1),
                                               ttest.p=numeric(nrow(subject.pool)-1))
    for (subject.n in c(1:(nrow(subject.pool)-1))) {
      set.seed(6)
      compare.temp = boot(subject.pool, boot.subject, R=100, sim = "permutation", dist.compared=dist.compared, subject.n=subject.n)
      boot.results[[feature.index]]$n.subject[subject.n] = subject.n
      boot.results[[feature.index]]$r[subject.n] = compare.temp$t0
      boot.results[[feature.index]]$se[subject.n] = sd(compare.temp$t)
      compare.last = compare.temp$t
      if (subject.n > 1) {
        boot.results[[feature.index]]$ttest.p[subject.n] = t.test(compare.last, compare.next, alternative = "greater", paired = F)$p.value
      }
      compare.next = compare.temp$t
      print(paste("feature.index = ",feature.index," subject.n = ",subject.n,sep = ""))
    } 
  }else{
    subject.pool = NULL
    for (e in c(1:27)) {
      add.emotion = ind.appraisal[ind.appraisal$emotion==emotions[e],feature.name]
      subject.pool = cbind(subject.pool, add.emotion[1:11])
    }
    boot.results[[feature.index]] = data.frame(n.subject=numeric(nrow(subject.pool)-1),
                                               r=numeric(nrow(subject.pool)-1),
                                               se=numeric(nrow(subject.pool)-1),
                                               ttest.p=numeric(nrow(subject.pool)-1))
    for (subject.n in c(1:(nrow(subject.pool)-1))) {
      set.seed(6)
      compare.temp = boot(subject.pool, boot.subject, R=100, sim = "permutation", dist.compared=dist.compared, subject.n=subject.n)
      boot.results[[feature.index]]$n.subject[subject.n] = subject.n
      boot.results[[feature.index]]$r[subject.n] = compare.temp$t0
      boot.results[[feature.index]]$se[subject.n] = sd(compare.temp$t)
      compare.last = compare.temp$t
      if (subject.n > 1) {
        boot.results[[feature.index]]$ttest.p[subject.n] = t.test(compare.last, compare.next, alternative = "greater", paired = F)$p.value
      }
      compare.next = compare.temp$t
      print(paste("feature.index = ",feature.index," subject.n = ",subject.n,sep = ""))
    } 
  }
}
for (feature.index in c(1:ncol(dist.features))) {
  if (feature.index==1){
    p = ggplot(boot.results[[feature.index]], aes(x=n.subject, y=r, ymin=r-se, ymax=r+se))+
      geom_line()+
      geom_ribbon(alpha=0.5)+
      scale_x_continuous(n.breaks = 4,expand = expansion(mult = c(0,0)))+
      scale_y_continuous(expand = expansion(mult = c(0,0)))+
      labs(title = colnames(dist.features)[feature.index], x="", y="")+
      theme_classic()
  }else{
    p = p + ggplot(boot.results[[feature.index]], aes(x=n.subject, y=r, ymin=r-se, ymax=r+se))+
      geom_line()+
      geom_ribbon(alpha=0.5)+
      scale_x_continuous(n.breaks = 4,expand = expansion(mult = c(0,0)))+
      scale_y_continuous(expand = expansion(mult = c(0,0)))+
      labs(title = colnames(dist.features)[feature.index], x="", y="")+
      theme_classic()
  }
}
p = p + plot_layout(ncol = 6)
ggsave("Result/supplement_information/correlation_between_different_numbers_of_participant.pdf", plot = p, width = 360, height = 370, units = "mm")




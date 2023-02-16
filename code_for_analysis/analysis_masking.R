library(bruceR)
library(ggsci)
library(patchwork)
library(rjson)
library(ggsignif)
set.wd()
set.wd("../")

# load data ----
analysis.type = "14property"
# analysis.type = "3space"
analysis.method = "RSA_tau"
# analysis.method = "linear_beta"
fpath = paste0("Result/masking/0521_result/",analysis.method,"_",analysis.type,"/")
# index data
index.emotions = unlist(import("Data/preprocessed/reordered_emotions.csv",header=F))
if (analysis.type=="14property") {
  index.features = colnames(import("Data/preprocessed/emotional_features/score_emotional_PCs.csv"))
  rank.feature = order(-(import("Result/RSA_whole_model/RSA_with_emotional_PCs_tau.csv")$`Subjective Comparison`))
  
} else {
  index.features = c("Affective Space","Basic Emotions Space","Appraisal Space")
  rank.feature = c(1,2,3)
}
index.seed = c(1,3,5,7,9,11,13,15,17,19,42,100)
# origin condition
origin.accuracy = import("Data/raw_data/PLM/prompt_tuning_RoBERTa/accuracy.csv",as="data.table")
colnames(origin.accuracy) = c("Sentiment","PromptID","Accuracy")
origin.accuracy[,Sentiment:=factor(Sentiment,levels = index.emotions)]
origin.accuracy[,PromptID:=paste0(Sentiment,PromptID)]
origin.accuracy[,PromptID:=factor(PromptID,levels = unique(PromptID))]
origin.accuracy[,"Type"] = "non-masked"
origin.accuracy[,Type:=factor(Type,levels = c("selective","random","non-masked"))]
# random masking condition
random.accuracy = list()
i = 0
for (e in index.emotions) {
  for (s in index.seed){
    i = i+1
    temp = fromJSON(file = paste0("Result/masking/0521_result/random/Acc_perSeed_JSON/",e,"-",s,".json"))
    # temp = fromJSON(file = paste0("Result/masking/0521_result/RSA_bottom/",e,"-",s,".json"))
    temp[[1]]=NULL
    temp[[1]]=NULL
    temp[[2]]=NULL
    random.accuracy[[i]] = data.table(Sentiment = rep(e,9*14),
                                      PromptID = rep(paste0(e,s),9*14),
                                      Fake.Feature = rep(c(1:14),each=9),
                                      Criteria = rep(c(500,1000,1500,2000,2500,
                                                       3000,4000,5000,6000),14),
                                      Type = rep("random",9*14),
                                      Accuracy = as.numeric(temp))
  }
}
random.accuracy = rbindlist(random.accuracy) %>%
  group_by(Sentiment, PromptID, Criteria, Type) %>%
  summarise_at("Accuracy",
               list(Accuracy = ~mean(.))) %>%
  as.data.table()
random.accuracy[,Sentiment:=factor(Sentiment,levels = index.emotions)]
random.accuracy[,PromptID:=factor(PromptID,levels = unique(PromptID))]
random.accuracy[,Type:=factor(Type,levels = c("selective","random","non-masking"))]
# selective masking condition
masked.accuracy = list()
i = 0
for (e in index.emotions) {
  for (s in index.seed){
    i = i+1
    temp = fromJSON(file = paste0(fpath,"Acc_perSeed_JSON/",e,"-",s,".json"))
    temp[[1]]=NULL
    temp[[1]]=NULL
    temp[[2]]=NULL
    if ((analysis.method=="linear_beta")&(analysis.type=="14property")) {
      masked.accuracy[[i]] = data.table(Sentiment = rep(e,9*length(index.features)),
                                        PromptID = rep(paste0(e,s),9*length(index.features)),
                                        Feature = c(rep(index.features,each=2),rep(index.features, each=7)),
                                        Criteria = c(rep(c(500,1000),length(index.features)),
                                                     rep(c(1500,2000,2500,3000,
                                                           4000,5000,6000),length(index.features))),
                                        Type = rep("selective",9*length(index.features)),
                                        Accuracy = as.numeric(temp))
    } else if ((analysis.method=="linear_beta")&(analysis.type=="3space")) {
      temp=temp[1:27]
      masked.accuracy[[i]] = data.table(Sentiment = rep(e,9*length(index.features)),
                                        PromptID = rep(paste0(e,s),9*length(index.features)),
                                        Feature = rep(index.features, each=9),
                                        Criteria = rep(c(500,1000,1500,2000,2500,
                                                         3000,4000,5000,6000),length(index.features)),
                                        Type = rep("selective",9*length(index.features)),
                                        Accuracy = as.numeric(temp))
    } else if ((analysis.method=="RSA_tau")&(analysis.type=="14property")) {
      masked.accuracy[[i]] = data.table(Sentiment = rep(e,7*length(index.features)),
                                        PromptID = rep(paste0(e,s),7*length(index.features)),
                                        Feature = rep(index.features, each=7),
                                        Criteria = rep(c(1500,2000,2500,3000,
                                                         4000,5000,6000),length(index.features)),
                                        Type = rep("selective",7*length(index.features)),
                                        Accuracy = as.numeric(temp))
    } else if ((analysis.method=="RSA_tau")&(analysis.type=="3space")) {
      temp=temp[1:27]
      masked.accuracy[[i]] = data.table(Sentiment = rep(e,9*length(index.features)),
                                        PromptID = rep(paste0(e,s),9*length(index.features)),
                                        Feature = rep(index.features, each=9),
                                        Criteria = rep(c(500,1000,1500,2000,2500,
                                                         3000,4000,5000,6000),length(index.features)),
                                        Type = rep("selective",9*length(index.features)),
                                        Accuracy = as.numeric(temp))
    }
  }
}
masked.accuracy = rbindlist(masked.accuracy)
masked.accuracy[,Sentiment:=factor(Sentiment,levels = index.emotions)]
masked.accuracy[,PromptID:=factor(PromptID,levels = unique(PromptID))]
masked.accuracy[,Feature:=factor(Feature,levels = index.features)]
masked.accuracy[,Type:=factor(Type,levels = c("selective","random","non-masking"))]
rm(e,s,i,temp)



# 若Criteria不一致，需进行数据对齐 ----
criteria = intersect(unique(random.accuracy$Criteria),unique(masked.accuracy$Criteria))
masked.accuracy = masked.accuracy[Criteria %in% criteria,]
random.accuracy = random.accuracy[Criteria %in% criteria,]



# visualize: ACC ----
plotdata = masked.accuracy %>%
  group_by(Feature, Criteria) %>%
  summarise_at("Accuracy",
               list(ACC = ~list(.),
                    ACC.mean = ~mean(.))) %>%
  as.data.table()
plotdata[,Feature:=factor(Feature,levels = index.features[rank.feature])]
p.ACC = ggplot(plotdata, aes(Feature,factor(Criteria),fill=ACC.mean))+
  geom_tile()+
  scale_fill_continuous(name="Averaged\nAccuracy",type = "viridis",direction=-1)+
  scale_x_discrete(name="Attributes",guide=guide_axis(angle = 90))+
  scale_y_discrete(name="Number of Masked Nuerons",limits=rev)+
  theme_minimal()
p.ACC
ggsave(paste0(fpath,"ACC.pdf"), plot = p.ACC, width = 240, height = 135, units = "mm")
# when N = 4000
masked.accuracy[Criteria==4000,] %>%
  group_by(Feature,Type) %>%
  summarise_at("Accuracy",
               list(ACC = ~mean(.),
                    SE = ~sqrt(var(.)/length(.)))) %>%
  as.data.table -> plotdata
if (analysis.type=="14property") {
  random.accuracy[Criteria==4000,] %>%
    cbind(Feature="random condition") %>%
    group_by(Feature,Type) %>%
    summarise_at("Accuracy",
                 list(ACC = ~mean(.),
                      SE = ~sqrt(var(.)/length(.)))) %>%
    as.data.table %>%
    rbind(plotdata) %>%
    merge(., data.table(Feature=factor(c("random condition",index.features),
                                       levels=c("random condition",index.features)),
                        Space=factor(c("",
                                       rep("Affective Space",2),
                                       rep("Basic Emotions Space",6),
                                       rep("Appraisal Space",6)),
                                     levels = c("Affective Space",
                                                "Basic Emotions Space",
                                                "Appraisal Space"))),
          by="Feature") -> plotdata
  plotdata[,Feature:=factor(Feature,levels = c("random condition",index.features[rank.feature]))]
}else{
  random.accuracy[Criteria==4000,] %>%
    cbind(Feature="random condition") %>%
    group_by(Feature,Type) %>%
    summarise_at("Accuracy",
                 list(ACC = ~mean(.),
                      SE = ~sqrt(var(.)/length(.)))) %>%
    as.data.table %>%
    rbind(plotdata) %>%
    merge(., data.table(Feature=factor(c("random condition",index.features),
                                       levels=c("random condition",index.features)),
                        Space=factor(c("",index.features),
                                     levels = c("Affective Space",
                                                "Basic Emotions Space",
                                                "Appraisal Space"))),
          by="Feature") -> plotdata
}
p.4000 = ggplot(plotdata, aes(Feature, ACC))+
  geom_col(aes(fill=Space))+
  geom_errorbar(aes(ymin=ACC-SE/2,ymax=ACC+SE/2), width = 0.1)+
  scale_x_discrete(name="Attributes",guide=guide_axis(angle = 90))+
  scale_y_continuous(name = "Acc after Masking Correspond Neurons",
                     expand = expansion(mult = c(0, .1)))+
  scale_fill_npg(name="",na.value="#595959",limits=c("Affective Space",
                                                     "Basic Emotions Space",
                                                     "Appraisal Space"))+
  coord_cartesian(ylim = c(0.5,NA))+
  theme_classic()+
  theme(legend.position="top")
ggsave(paste0(fpath,"Acc_top4000.pdf"),plot = p.4000, width = 240, height =135, units = "mm")



# statistic: t-test ----
ttest.versus = masked.accuracy %>%
  group_by(Feature, Criteria) %>%
  summarise_at("Accuracy",
               list(ACC = ~list(.),
                    ACC.mean = ~mean(.))) %>%
  as.data.table()
ttest.random = random.accuracy %>%
  group_by(Criteria) %>%
  summarise_at("Accuracy",
               list(ACC.random = ~list(.))) %>%
  as.data.table()
ttest.versus = merge(ttest.versus,ttest.random,by=c("Criteria"))
for (i in 1:nrow(ttest.versus)){
  ttest.temp = t.test(ttest.versus$ACC[[i]],
                      ttest.versus$ACC.random[[i]],
                      paired = T,
                      alternative = "less")
  ttest.versus[["p.value"]][i] = ttest.temp$p.value
  ttest.versus[["t.value"]][i] = ttest.temp$statistic
  ttest.versus[["diff.estimate"]][i] = ttest.temp$estimate
}
rm(ttest.random,ttest.temp,i)
# correct p-value by FDR
ttest.versus[["corrected.p.value"]] = ttest.versus[,p.value*.N/rank(p.value)]
ttest.versus[["significance"]] = ttest.versus[,corrected.p.value<0.05]
ttest.versus[["signif.ACC"]] = ttest.versus[,ACC.mean]
ttest.versus[["signif.ACC"]][!ttest.versus$significance] = NA
ttest.versus[["signif.t"]] = ttest.versus[,t.value]
ttest.versus[["signif.t"]][!ttest.versus$significance] = NA
ttest.versus[["signif.diff"]] = ttest.versus[,diff.estimate]
ttest.versus[["signif.diff"]][!ttest.versus$significance] = NA
ttest.versus[,Feature:=factor(Feature,levels = index.features[rank.feature])]



# visualization: t-test ----
# visualization: ACC after t-test
p.ACC = ggplot(ttest.versus, aes(Feature,factor(Criteria),fill=signif.ACC))+
  geom_tile()+
  scale_fill_continuous(type = "viridis",direction=-1)+
  scale_x_discrete(name="",guide=guide_axis(angle = 90))+
  scale_y_discrete(name="",limits=rev)+
  theme_minimal()
p.ACC
ggsave(paste0(fpath,"ttest_signif_ACC.pdf"), plot = p.ACC, width = 240, height = 135, units = "mm")
# visualization: ACC difference after t-test
p.diff = ggplot(ttest.versus, aes(Feature,factor(Criteria),fill=signif.diff))+
  geom_tile()+
  scale_fill_continuous(type = "viridis",direction=-1)+
  scale_x_discrete(name="",guide=guide_axis(angle = 90))+
  scale_y_discrete(name="",limits=rev)+
  theme_minimal()
p.diff
ggsave(paste0(fpath,"ttest_signif_ACC_difference.pdf"), plot = p.diff, width = 240, height = 135, units = "mm")
# visualization: t-value after t-test
p.t = ggplot(ttest.versus, aes(Feature,factor(Criteria),fill=signif.t))+
  geom_tile()+
  scale_fill_continuous(type = "viridis",direction=-1)+
  scale_x_discrete(name="",guide=guide_axis(angle = 90))+
  scale_y_discrete(name="",limits=rev)+
  theme_minimal()
p.t
ggsave(paste0(fpath,"ttest_signif_t_value.pdf"), plot = p.t, width = 240, height = 135, units = "mm")



# statistic: ANOVA ----
if (analysis.type=="14property") {
  anova.masked = MANOVA(masked.accuracy[,Feature:=gsub("-",".",Feature,fixed = T)],
                        subID = "PromptID", dv = "Accuracy", sph.correction = "GG",
                        between = c("Sentiment"), within = c("Feature","Criteria"))
  masked.accuracy[,Feature:=gsub(".","-",Feature,fixed = T)]
  masked.accuracy[,Feature:=factor(Feature,levels = index.features)]
} else {
  anova.masked = MANOVA(masked.accuracy[,Feature:=gsub(" ",".",Feature,fixed = T)],
                        subID = "PromptID", dv = "Accuracy", sph.correction = "GG",
                        between = c("Sentiment"), within = c("Feature","Criteria"))
  masked.accuracy[,Feature:=gsub("."," ",Feature,fixed = T)]
  masked.accuracy[,Feature:=factor(Feature,levels = index.features)]
}



# visualization: ANOVA ----
# Main Effect 1: [Criteria]
plotdata1 = masked.accuracy %>%
  group_by(Criteria,Type) %>%
  summarise_at("Accuracy",
               list(ACC = ~mean(.),
                    SE = ~sqrt(var(.)/length(.)))) %>%
  as.data.table
plotdata2 = random.accuracy %>%
  group_by(Criteria,Type) %>%
  summarise_at("Accuracy",
               list(ACC = ~mean(.),
                    SE = ~sqrt(var(.)/length(.)))) %>%
  as.data.table
plotdata = rbind(plotdata1,plotdata2)
p.main.1 = ggplot(plotdata, aes(Criteria, ACC, fill=Type, colour=Type))+
  geom_hline(yintercept = mean(origin.accuracy$Accuracy), linetype="dashed")+
  geom_line(aes(group = Type), position = position_dodge(width = 0.15))+
  geom_errorbar(aes(ymin=ACC-SE/2,ymax=ACC+SE/2), width = 0.3, alpha=0.8,)+
  geom_point(shape=21,colour="black", position = position_dodge(width = 0.15))+
  scale_x_continuous(name="", guide = guide_axis(angle=90),breaks = unique(plotdata$Criteria))+
  scale_y_continuous(name = "Acc after Masking Correspond Neurons")+
  scale_fill_npg(name="Masked Neurons")+
  scale_color_npg(name="Masked Neurons")+
  theme_classic()
p.main.1
ggsave(paste0(fpath,"/MainEffect1_Criteria.pdf"),plot = p.main.1, width = 240, height = 135, units = "mm")
rm(plotdata1,plotdata2,plotdata)

# Main Effect 2: [Sentiment]
plotdata1 = masked.accuracy %>%
  group_by(Sentiment,Type) %>%
  summarise_at("Accuracy",
               list(ACC = ~mean(.),
                    SE = ~sqrt(var(.)/length(.)))) %>%
  as.data.table
plotdata2 = random.accuracy %>%
  group_by(Sentiment,Type) %>%
  summarise_at("Accuracy",
               list(ACC = ~mean(.),
                    SE = ~sqrt(var(.)/length(.)))) %>%
  as.data.table
plotdata3 = origin.accuracy %>%
  group_by(Sentiment,Type) %>%
  summarise_at("Accuracy",
               list(ACC = ~mean(.),
                    SE = ~sqrt(var(.)/length(.)))) %>%
  as.data.table
plotdata = rbind(plotdata1,plotdata2,plotdata3)
p.main.2 = ggplot(plotdata, aes(reorder(Sentiment,-ACC), ACC, group=Type, fill=Type, colour=Type))+
  geom_line(position = position_dodge(width = 0.15))+
  geom_errorbar(aes(ymin=ACC-SE/2,ymax=ACC+SE/2), width = 0.3, alpha=0.8,)+
  geom_point(shape=21,colour="black", position = position_dodge(width = 0.15))+
  scale_x_discrete(name="", guide = guide_axis(angle=90))+
  scale_y_continuous(name = "Acc after Masking Correspond Neurons")+
  scale_fill_npg(name="Masked Neurons")+
  scale_color_npg(name="Masked Neurons")+
  theme_classic()
p.main.2
ggsave(paste0(fpath,"/MainEffect2_Sentiment.pdf"),plot = p.main.2, width = 240, height =135, units = "mm")
rm(plotdata,plotdata1,plotdata2,plotdata3)

# Main Effect 3: [Feature]
plotdata = masked.accuracy %>%
  group_by(Feature,Type) %>%
  summarise_at("Accuracy",
               list(ACC = ~mean(.),
                    SE = ~sqrt(var(.)/length(.)))) %>%
  as.data.table
plotdata[,Feature:=factor(Feature,levels = index.features[rank.feature])]
p.main.3 = ggplot(plotdata, aes(Feature, ACC, fill=Type, colour=Type))+
  geom_hline(yintercept = mean(origin.accuracy$Accuracy), linetype="dashed")+
  geom_hline(yintercept = mean(random.accuracy$Accuracy), linetype="dotdash")+
  geom_line(aes(group = Type), position = position_dodge(width = 0.15))+
  geom_errorbar(aes(ymin=ACC-SE/2,ymax=ACC+SE/2), width = 0.3, alpha=0.8,)+
  geom_point(shape=21,colour="black", position = position_dodge(width = 0.15))+
  scale_x_discrete(name="", guide = guide_axis(angle=90))+
  scale_y_continuous(name = "Acc after Masking Correspond Neurons")+
  scale_fill_npg(name="Masked Neurons")+
  scale_color_npg(name="Masked Neurons")+
  theme_classic()
p.main.3
ggsave(paste0(fpath,"MainEffect3_Feature.pdf"),plot = p.main.3, width = 240, height =135, units = "mm")
rm(plotdata)

# Interactive Effect 1: [Criteria * Feature]
plotdata1 = masked.accuracy %>%
  group_by(Feature,Criteria,Type) %>%
  summarise_at("Accuracy",
               list(ACC=~mean(.),
                    SE=~sqrt(var(.)/length(.)))) %>%
  as.data.table
plotdata2 = random.accuracy %>%
  group_by(Criteria,Type) %>%
  summarise_at("Accuracy",
               list(ACC=~mean(.),
                    SE=~sqrt(var(.)/length(.)))) %>%
  as.data.table
plotdata1[,Feature:=factor(Feature,levels = index.features[rank.feature])]
p.inter.1 = ggplot()+
  geom_hline(yintercept = mean(origin.accuracy$Accuracy), linetype="dashed")+
  geom_line(data = plotdata2, aes(Criteria, ACC, colour=Type),
            position = position_dodge(width = 0.15))+
  geom_errorbar(data = plotdata2, aes(Criteria, ymin=ACC-SE/2, ymax=ACC+SE/2, colour=Type),
                width = 0.3, alpha=0.8)+
  geom_point(data = plotdata2, aes(Criteria, ACC, fill=Type),
             shape=21,colour="black", position = position_dodge(width = 0.15))+
  geom_line(data = plotdata1, aes(Criteria, ACC, colour=Type),
            position = position_dodge(width = 0.15))+
  geom_errorbar(data = plotdata1, aes(Criteria, ymin=ACC-SE/2,ymax=ACC+SE/2, colour=Type),
                width = 0.3, alpha=0.8)+
  geom_point(data = plotdata1, aes(Criteria, ACC, fill=Type),
             shape=21, colour="black", position = position_dodge(width = 0.15))+
  scale_x_continuous(name="Number of Masked Neurons", guide = guide_axis(angle=90))+
  scale_y_continuous(name = "Acc after Masking Correspond Neurons")+
  scale_fill_npg(name="Masked Neurons",limits=c("selective","random"))+
  scale_color_npg(name="Masked Neurons",limits=c("selective","random"))+
  theme_bw()+
  facet_wrap(~Feature)
p.inter.1
ggsave(paste0(fpath,"InterEffect1_Criteria&Feature.pdf"),plot = p.inter.1, width = 240, height = 135, units = "mm")

# Interactive Effect 2: [Criteria * Sentiment]
plotdata1 = masked.accuracy %>%
  group_by(Sentiment,Criteria,Type) %>%
  summarise_at("Accuracy",
               list(ACC=~mean(.),
                    SE=~sqrt(var(.)/length(.)))) %>%
  as.data.table
plotdata2 = random.accuracy %>%
  group_by(Sentiment,Criteria,Type) %>%
  summarise_at("Accuracy",
               list(ACC=~mean(.),
                    SE=~sqrt(var(.)/length(.)))) %>%
  as.data.table
plotdata3 = origin.accuracy %>%
  group_by(Sentiment,Type) %>%
  summarise_at("Accuracy",
               list(ACC=~mean(.),
                    SE=~sqrt(var(.)/length(.)))) %>%
  as.data.table
plotdata3 = do.call("rbind", replicate(length(criteria), plotdata3, simplify = FALSE))
plotdata3[,"Criteria"]=rep(criteria,each=27)
plotdata = rbind(plotdata1,plotdata2)
p.inter.2 = ggplot(plotdata, aes(Criteria, ACC, group=Type, colour=Type, fill=Type))+
  geom_line(data=plotdata3, aes(y=ACC, colour=Type),linetype="dashed")+
  geom_line(position = position_dodge(width = 0.15))+
  geom_errorbar(aes(ymin=ACC-SE/2, ymax=ACC+SE/2),
                width = 0.3, alpha=0.8)+
  geom_point(shape=21,colour="black", position = position_dodge(width = 0.15))+
  scale_x_continuous(name="", guide = guide_axis(angle=90))+
  scale_y_continuous(name = "Acc after Masking Correspond Neurons")+
  scale_fill_npg(name="Masked Neurons",limits=c("selective","random","non-masked"))+
  scale_color_npg(name="Masked Neurons",limits=c("selective","random","non-masked"))+
  facet_wrap(~Sentiment)
p.inter.2
ggsave(paste0(fpath,"InterEffect2_Criteria&Sentiment.pdf"),plot = p.inter.2, width = 240, height = 135, units = "mm")
rm(plotdata,plotdata1,plotdata2,plotdata3)

# Interactive Effect 3: Sentiment * Feature
plotdata1 = masked.accuracy[Criteria==4000,] %>%
  group_by(Sentiment, Feature, Type) %>%
  summarise_at("Accuracy",
               list(ACC = ~mean(.),
                    SE = ~sqrt(var(.)/length(.)))) %>%
  as.data.table
plotdata1[,Feature:=factor(Feature,levels = index.features[rank.feature])]
plotdata2 = random.accuracy[Criteria==4000,] %>%
  group_by(Sentiment, Type) %>%
  summarise_at("Accuracy",
               list(ACC = ~mean(.),
                    SE = ~sqrt(var(.)/length(.)))) %>%
  as.data.table
plotdata2 = do.call("rbind", replicate(length(index.features), plotdata2, simplify = FALSE))
plotdata2[,"Feature"] = rep(index.features,each=27)
plotdata2[,Feature:=factor(Feature,levels = index.features[rank.feature])]
plotdata3 = origin.accuracy %>%
  group_by(Sentiment,Type) %>%
  summarise_at("Accuracy",
               list(ACC=~mean(.),
                    SE=~sqrt(var(.)/length(.)))) %>%
  as.data.table
plotdata3 = do.call("rbind", replicate(length(index.features), plotdata3, simplify = FALSE))
plotdata3[,"Feature"]=rep(index.features,each=27)
plotdata3[,Feature:=factor(Feature,levels = index.features[rank.feature])]
p.inter.3a = ggplot(plotdata1, aes(Feature, ACC, group=Type, colour=Type, fill=Type))+
  geom_line(data=plotdata2, aes(y=ACC, colour=Type),linetype="dashed")+
  geom_line(data=plotdata3, aes(y=ACC, colour=Type),linetype="dashed")+
  geom_line(position = position_dodge(width = 0.15))+
  geom_errorbar(aes(ymin=ACC-SE/2, ymax=ACC+SE/2),
                width = 0.3, alpha=0.8)+
  geom_point(shape=21,colour="black", position = position_dodge(width = 0.15))+
  scale_x_discrete(name="Attributes", guide = guide_axis(angle=90))+
  scale_y_continuous(name = "Acc after Masking Correspond Neurons")+
  scale_fill_npg(name="Masked Neurons",limits=c("selective","random","non-masked"))+
  scale_color_npg(name="Masked Neurons",limits=c("selective","random","non-masked"))+
  facet_wrap(~Sentiment)+
  theme_bw()+
  theme(axis.text.x = element_text(size=5))
p.inter.3a
ggsave(paste0(fpath,"InterEffect3a_Feature&Sentiment.pdf"),plot = p.inter.3a, width = 240, height = 135, units = "mm")
p.inter.3b = ggplot(plotdata1, aes(reorder(Sentiment,-ACC), ACC, group=Type, colour=Type, fill=Type))+
  geom_line(data=plotdata2, aes(y=ACC, colour=Type),linetype="dashed")+
  geom_line(data=plotdata3, aes(y=ACC, colour=Type),linetype="dashed")+
  geom_line(position = position_dodge(width = 0.15))+
  geom_errorbar(aes(ymin=ACC-SE/2, ymax=ACC+SE/2),
                width = 0.3, alpha=0.8)+
  geom_point(shape=21,colour="black", position = position_dodge(width = 0.15))+
  scale_x_discrete(name="Emotion Categories", guide = guide_axis(angle=90),)+
  scale_y_continuous(name = "Acc after Masking Correspond Neurons")+
  scale_fill_npg(name="Masked Neurons",limits=c("selective","random","non-masked"))+
  scale_color_npg(name="Masked Neurons",limits=c("selective","random","non-masked"))+
  facet_wrap(~Feature)+
  theme_bw()+
  theme(axis.text.x = element_text(size=5))
p.inter.3b
ggsave(paste0(fpath,"InterEffect3b_Feature&Sentiment.pdf"),plot = p.inter.3b, width = 240, height = 135, units = "mm")
rm(plotdata1,plotdata2,plotdata3)



# statistic: compare to human behavior ----
human.PLM.compare = list()
# load data
if (analysis.type=="14property") {
  human.PLM.compare$human.behavior = import("Result/RSA_whole_model/RSA_with_emotional_PCs_tau.csv")$`Subjective Comparison`
}else{
  human.PLM.compare$human.behavior = import("Result/RSA_whole_model/RSA_with_emotion_space_tau.csv")$`Subjective Comparison`[1:3]
}
names(human.PLM.compare$human.behavior) = index.features
# merge masked ACC
human.PLM.compare$PLM.behavior = list()
human.PLM.compare$PLM.behavior$ACC = masked.accuracy %>%
  group_by(Feature, Criteria) %>%
  summarise_at("Accuracy",
               list(ACC = ~mean(.))) %>%
  as.data.table() %>%
  dcast(Feature ~ Criteria, value.var = "ACC")
human.PLM.compare$PLM.behavior$mean.ACC = rowMeans(human.PLM.compare$PLM.behavior$ACC[,-1])
# compare
human.PLM.compare$mean.r = cor.test(human.PLM.compare$human.behavior[c(1,2,9:14)],
                                    human.PLM.compare$PLM.behavior$mean.ACC[c(1,2,9:14)],
                                    alternative = "less")
human.PLM.compare$mean.rho = cor.test(human.PLM.compare$human.behavior[c(1,2,9:14)],
                                      human.PLM.compare$PLM.behavior$mean.ACC[c(1,2,9:14)],
                                      method = "spearman",
                                      alternative = "less")
human.PLM.compare$pearson.r = lapply(human.PLM.compare$PLM.behavior$ACC[c(1,2,9:14),2:length(human.PLM.compare$PLM.behavior$ACC)],
                                     cor.test,
                                     y = human.PLM.compare$human.behavior[c(1,2,9:14)],
                                     alternative = "less")
human.PLM.compare$spearman.rho = lapply(human.PLM.compare$PLM.behavior$ACC[c(1,2,9:14),2:length(human.PLM.compare$PLM.behavior$ACC)],
                                        cor.test,
                                        y = human.PLM.compare$human.behavior[c(1,2,9:14)],
                                        method = "spearman",
                                        alternative = "less")



# visualization: compare to Human behavior ----
# 14 attributes
masked.accuracy %>%
  group_by(Feature, Criteria) %>%
  summarise_at("Accuracy",
               list(ACC = ~mean(.))) %>%
  as.data.table() %>%
  merge(., data.table(association=human.PLM.compare$human.behavior,
                      Feature=factor(index.features,levels=index.features)),
        by="Feature") -> plotdata
random.accuracy %>%
  group_by(Criteria) %>%
  summarise_at("Accuracy",
               list(control = ~mean(.))) %>%
  as.data.table() %>%
  merge(., plotdata, by="Criteria") %>%
  merge(., data.table(Feature=factor(index.features,levels=index.features),
                      Space=factor(c(rep("Affective Space",2),
                                     rep("Basic Emotions Space",6),
                                     rep("Appraisal Space",6)),
                                   levels = c("Affective Space",
                                              "Basic Emotions Space",
                                              "Appraisal Space"))),
        by="Feature") -> plotdata
plotdata[,Feature:=factor(Feature,levels = index.features[rank.feature])]
p.compare = ggplot(plotdata,
                   aes(association, ACC, group=Criteria))+
  geom_line(aes(y=control),colour="black",linetype="dashed")+
  geom_point(aes(color=Feature, shape=Space),size=2.5,alpha=1)+
  geom_smooth(method = "lm",se=T)+
  scale_color_discrete(name = "Attributes", position = "top")+
  scale_shape_discrete()+
  scale_x_continuous(name = "Association with Human Behavior", guide = guide_axis(angle=90))+
  scale_y_continuous(name = "Acc after Masking Correspond Neurons")+
  facet_grid(~Criteria)+
  theme_classic()
ggsave(paste0(fpath,"compare_to_human_association_14property.pdf"), plot = p.compare, width = 240, height = 135, units = "mm")
# 12 attributes
p.compare = ggplot(plotdata[(Feature!="arousal")&(Feature!="other-related"),],
                   aes(association, ACC, group=Criteria))+
  geom_line(aes(y=control),colour="black",linetype="dashed")+
  geom_point(aes(color=Feature, shape=Space),size=2.5,alpha=1)+
  geom_smooth(method = "lm",se=T)+
  scale_color_discrete(name = "Attributes", position = "top")+
  scale_shape_discrete()+
  scale_x_continuous(name = "Association with Human Behavior", guide = guide_axis(angle=90))+
  scale_y_continuous(name = "Acc after Masking Correspond Neurons")+
  facet_grid(~Criteria)+
  theme_classic()
ggsave(paste0(fpath,"compare_to_human_association_12property.pdf"), plot = p.compare, width = 240, height = 135, units = "mm")
# N = 4000
p.compare = ggplot(plotdata[Criteria==4000,],
                   aes(association, ACC))+
  geom_line(aes(y=control),colour="black",linetype="dashed")+
  geom_point(aes(color=Feature, shape=Space),size=2.5,alpha=1)+
  geom_smooth(method = "lm",se=T)+
  scale_color_discrete(name = "Attributes", position = "top")+
  scale_shape_discrete()+
  scale_x_continuous(name = "Association with Human Behavior", guide = guide_axis(angle=90))+
  scale_y_continuous(name = "Acc after Masking Correspond Neurons")+
  theme_classic()
ggsave(paste0(fpath,"compare_to_human_association_top4000.pdf"), plot = p.compare, width = 240, height = 135, units = "mm")



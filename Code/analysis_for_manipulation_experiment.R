library(bruceR)
library(ggsci)
library(patchwork)
library(rjson)
library(ggsignif)
library(ggrepel)
library(RColorBrewer)
library(geneRal)
library(DescTools)
library(corrplot)
library(colorspace)
library(jsonlite)
library(DescTools)
set.wd()
set.wd("../")



# load data ----
experiment.type = "14property"
experiment.method = "RSA_tau"
fpath = paste0("Result/masking/RoBERTa/",experiment.method,"_",experiment.type,"/")
# index data
index.emotions = unlist(import("Data/preprocessed/reordered_emotions.csv",header=F))
index.features = colnames(import("Data/preprocessed/emotional_features/score_emotional_PCs.csv"))
index.seed = c(1,3,5,7,9,11,13,15,17,19,42,100)
# origin condition
origin.accuracy = import("Data/raw_data/PLM/prompt_tuning_RoBERTa/accuracy.csv",as="data.table")
colnames(origin.accuracy) = c("Sentiment","PromptID","Accuracy")
origin.accuracy[,Sentiment:=factor(Sentiment,levels = index.emotions)]
origin.accuracy[,PromptID:=paste0(Sentiment,PromptID)]
origin.accuracy[,PromptID:=factor(PromptID,levels = unique(PromptID))]
origin.accuracy[,"Type"] = "raw"
origin.accuracy[,Type:=factor(Type,levels = c("selective","random","raw"))]
# random masking condition
random.accuracy = list()
i = 0
for (e in index.emotions) {
  for (s in index.seed){
    i = i+1
    temp = fromJSON(txt = paste0("Result/masking/RoBERTa/random/Acc_perSeed_JSON/",e,"-",s,".json"))
    temp[[1]]=NULL
    temp[[1]]=NULL
    temp[[2]]=NULL
    random.accuracy[[i]] = data.table(Sentiment = rep(e,9*14),
                                      PromptID = rep(paste0(e,s),9*14),
                                      Feature = rep(index.features,each=9),
                                      Criteria = rep(c(500,1000,1500,2000,2500,
                                                       3000,4000,5000,6000),14),
                                      Type = rep("random",9*14),
                                      Accuracy = as.numeric(temp))
  }
}
random.accuracy = rbindlist(random.accuracy) %>%
  as.data.table()
random.accuracy[,Sentiment:=factor(Sentiment,levels = index.emotions)]
random.accuracy[,PromptID:=factor(PromptID,levels = unique(PromptID))]
random.accuracy[,Feature:=factor(Feature,levels = index.features)]
random.accuracy[,Type:=factor(Type,levels = c("selective","random","non-masking"))]
# selective masking condition
masked.accuracy = list()
i = 0
for (e in index.emotions) {
  for (s in index.seed){
    i = i+1
    temp = fromJSON(txt = paste0(fpath,"Acc_perSeed_JSON/",e,"-",s,".json"))
    temp[[1]]=NULL
    temp[[1]]=NULL
    temp[[2]]=NULL
    masked.accuracy[[i]] = data.table(Sentiment = rep(e,7*length(index.features)),
                                      PromptID = rep(paste0(e,s),7*length(index.features)),
                                      Feature = rep(index.features, each=7),
                                      Criteria = rep(c(1500,2000,2500,3000,
                                                       4000,5000,6000),length(index.features)),
                                      Type = rep("selective",7*length(index.features)),
                                      Accuracy = as.numeric(temp))
  }
}
masked.accuracy = rbindlist(masked.accuracy)
masked.accuracy[,Sentiment:=factor(Sentiment,levels = index.emotions)]
masked.accuracy[,PromptID:=factor(PromptID,levels = unique(PromptID))]
masked.accuracy[,Feature:=factor(Feature,levels = index.features)]
masked.accuracy[,Type:=factor(Type,levels = c("selective","random","raw"))]
rm(e,s,i,temp)
# human conceptual structure of emotions
similarity.judgment = import("Result/RSA_whole_model/similarity_judgment_RSA_with_emotional_PCs.csv",as="data.table")
similarity.judgment[,Feature:=index.features]
similarity.judgment[q<0.001,signif:="*"]
setorder(similarity.judgment,tau)
index.features = similarity.judgment$Feature



# align criteria -----
criteria = intersect(unique(random.accuracy$Criteria),unique(masked.accuracy$Criteria))
masked.accuracy = masked.accuracy[Criteria %in% criteria,]
random.accuracy = random.accuracy[Criteria %in% criteria,]



# reorganize data  ----

# raw ACC for N = 4000
random.accuracy %>%
  group_by(Criteria,Sentiment) %>%
  summarise_at("Accuracy",
               list(ACC = ~mean(.),
                    SE = ~t.test(.)$stderr)) %>%
  as.data.table %>%
  .[,type:="random"] -> plot.data1
masked.accuracy %>%
  group_by(Criteria,Sentiment) %>%
  summarise_at("Accuracy",
               list(ACC = ~mean(.),
                    SE = ~t.test(.)$stderr)) %>%
  as.data.table %>%
  .[,type:="attribute-specific"]-> plot.data2
origin.accuracy %>%
  group_by(Sentiment) %>%
  summarise_at("Accuracy",
               list(ACC = ~mean(.),
                    SE = ~t.test(.)$stderr)) %>%
  as.data.table %>%
  .[,type:="w/o"]-> plot.data3
rbind(plot.data3,plot.data3,plot.data3,plot.data3,plot.data3,plot.data3,plot.data3) %>%
  .[,Criteria:=rep(criteria,each=length(index.emotions))] -> plot.data3
rbind(plot.data1,plot.data2,plot.data3) %>%
  .[,type := factor(type,levels = c("w/o","random","attribute-specific"))] -> ACC.all
rm(plot.data1,plot.data2,plot.data3)



# ACC drop for all combination
masked.accuracy %>%
  merge(random.accuracy,
        by=c("Sentiment","PromptID","Feature","Criteria")) %>%
  group_by(Feature,Sentiment,Criteria) %>%
  summarise(.groups = "keep",
            ACC.diff = t.test(Accuracy.x, Accuracy.y, paired=T,alternative="less")$estimate[1],
            t.value = t.test(Accuracy.x, Accuracy.y,  paired=T,alternative="less")$statistic[1],
            df = t.test(Accuracy.x, Accuracy.y, paired=T,alternative="less")$parameter[1],
            SE = t.test(Accuracy.x, Accuracy.y,paired=T,alternative="less")$stderr,
            CI.lower = t.test(Accuracy.x, Accuracy.y, paired=T)$conf.int[1],
            CI.upper = t.test(Accuracy.x, Accuracy.y, paired=T)$conf.int[2],
            p.value = t.test(Accuracy.x, Accuracy.y, alternative="less", paired=T)$p.value) %>%
  as.data.table %>%
  .[,q.value:=p.value*.N/rank(p.value)] %>%
  .[,signif:=""] %>%
  .[q.value<0.05,signif:="*"] %>%
  # .[q.value<0.01,signif:="**"] %>%
  # .[q.value<0.001,signif:="***"] %>%
  merge(similarity.judgment[,.(Feature,tau)],
        by=c("Feature")) %>%
  .[,Feature:=factor(Feature,levels = index.features)]-> AccDrop.all.combination

# ACC drop for all sentiments
masked.accuracy %>%
  merge(random.accuracy,
        by=c("Sentiment","PromptID","Feature","Criteria")) %>%
  group_by(Sentiment,Criteria) %>%
  summarise(.groups = "keep",
            ACC.diff = t.test(Accuracy.x, Accuracy.y, paired=T,alternative="less")$estimate[1],
            t.value = t.test(Accuracy.x, Accuracy.y,  paired=T,alternative="less")$statistic[1],
            df = t.test(Accuracy.x, Accuracy.y, paired=T,alternative="less")$parameter[1],
            SE = t.test(Accuracy.x, Accuracy.y,paired=T,alternative="less")$stderr,
            CI.lower = t.test(Accuracy.x, Accuracy.y, paired=T)$conf.int[1],
            CI.upper = t.test(Accuracy.x, Accuracy.y, paired=T)$conf.int[2],
            p.value = t.test(Accuracy.x, Accuracy.y, alternative="less", paired=T)$p.value) %>%
  as.data.table %>%
  .[,q.value:=p.value*.N/rank(p.value)] %>%
  .[,signif:=""] %>%
  .[q.value<0.05,signif:="*"] -> AccDrop.by.sentiment
  # .[q.value<0.01,signif:="**"] %>%
  # .[q.value<0.001,signif:="***"] 

# ACC drop for all features
masked.accuracy %>%
  merge(random.accuracy,
        by=c("Sentiment","PromptID","Feature","Criteria")) %>%
  group_by(Feature,Criteria) %>%
  summarise(.groups = "keep",
            ACC.diff = t.test(Accuracy.x, Accuracy.y, paired=T,alternative="less")$estimate[1],
            t.value = t.test(Accuracy.x, Accuracy.y,  paired=T,alternative="less")$statistic[1],
            df = t.test(Accuracy.x, Accuracy.y, paired=T,alternative="less")$parameter[1],
            SE = t.test(Accuracy.x, Accuracy.y,paired=T,alternative="less")$stderr,
            CI.lower = t.test(Accuracy.x, Accuracy.y, paired=T)$conf.int[1],
            CI.upper = t.test(Accuracy.x, Accuracy.y, paired=T)$conf.int[2],
            p.value = t.test(Accuracy.x, Accuracy.y, alternative="less", paired=T)$p.value) %>%
  as.data.table %>%
  .[,q.value:=p.value*.N/rank(p.value)] %>%
  .[,signif:=""] %>%
  .[q.value<0.05,signif:="*"] %>%
  # .[q.value<0.01,signif:="**"] %>%
  # .[q.value<0.001,signif:="***"] %>%
  merge(similarity.judgment[,.(Feature,tau,se,CI.lower,CI.upper,q,signif)],
        by=c("Feature")) %>%
  .[,Feature:=factor(Feature,levels = index.features)] -> AccDrop.by.feature






# raw ACC ----
# rater agreement
import("Data/emotion_agreements.csv",as="data.table") %>%
  .[1:27,c(1:4)] %>%
  .[,Sentiment:=Emotion] -> rater.agreement
rater.agreement %>%
  merge(ACC.all[(type=="w/o")&(Criteria==1500)]) %>%
  .[,`:=`(Sentiment=NULL,Criteria=NULL,type=NULL)] -> rater.agreement
p.agreement = ggplot(rater.agreement,aes(`Cohen's kappa`,ACC,label=Emotion))+
  geom_point()+
  # geom_text_repel()+
  scale_x_continuous(name = "Rater Agreement (Cohen's Kappa)")+
  scale_y_continuous(name="Inference Accuracy",labels = scales::percent)+
  # coord_cartesian(ylim = c(0.5,1))+
  theme_classic()+
  theme(title = element_text(family = "Helvetica",size = 6),
        text = element_text(family = "Helvetica",size = 6))
p.agreement
ggsave(paste0(fpath,"agreement.pdf"), plot = p.agreement, width = 90, height = 60, units = "mm")
# for N = 4000 
p.Acc4000 = ggplot(ACC.all[Criteria==4000],aes(reorder(Sentiment,-ACC),ACC,fill=type))+
  geom_col(position=position_dodge())+
  geom_linerange(aes(ymin=ACC-0.5*SE,ymax=ACC+0.5*SE),position=position_dodge(0.9),color="red")+
  scale_fill_viridis_d(name="Neuron type",option = "magma",begin = 0.1,end = 0.9)+
  scale_x_discrete(name="Emotions",guide=guide_axis(angle = 90))+
  scale_y_continuous(name="Accuracy",labels = scales::percent)+
  labs(title = "Inference performance after neuron manipulation")+
  coord_cartesian(ylim = c(0.5,1))+
  theme_classic()+
  theme(legend.position = c(0.9,0.9),
        # legend.direction = "horizontal",
        legend.key.size = unit(2, 'mm'),
        legend.title = element_text(color = "red"),
        plot.title = element_text(hjust = 0.5),
        title = element_text(family = "Helvetica",size = 6),
        text = element_text(family = "Helvetica",size = 6),
        axis.ticks.length.x=unit(0,units = "mm"))
p.Acc4000
ggsave(paste0(fpath,"4000_Acc.pdf"), plot = p.Acc4000, width = 108, height = 50, units = "mm")
# for different N
p.Acc = ggplot(ACC.all,aes(Criteria,ACC,color=type))+
  geom_line()+
  geom_linerange(aes(ymin=ACC-0.5*SE,ymax=ACC+0.5*SE),color="red")+
  # geom_point()+
  scale_color_viridis_d(name="Neuron type",option = "magma",begin = 0.1,end = 0.9)+
  scale_x_continuous(name="Number of Manipulated Nuerons",guide=guide_axis(angle = 90))+
  scale_y_continuous(name="Accuracy",labels = scales::percent)+
  labs(title = "Inference performance after neuron manipulation")+
  # coord_cartesian(ylim = c(0.5,1))+
  facet_wrap(vars(reorder(Sentiment,-ACC)),scales = "free_y")+
  theme_classic()+
  theme(legend.position = c(0.8,0.05),
        # legend.direction = "horizontal",
        # legend.key.size = unit(2, 'mm'),
        plot.title = element_text(hjust = 0.5),
        title = element_text(family = "Helvetica",size = 6),
        text = element_text(family = "Helvetica",size = 6),
        axis.ticks.length.x=unit(0,units = "mm"))
p.Acc
ggsave(paste0(fpath,"All_Acc.pdf"), plot = p.Acc, width = 180, height = 220, units = "mm")



# ACC drop by sentiment ----
# for N = 4000
p.AccDrop.4000.sentiment = ggplot(AccDrop.by.sentiment[Criteria==4000,],
                                  aes(reorder(Sentiment,ACC.diff),
                                      -ACC.diff,
                                      fill="whatever"))+
  geom_col()+
  geom_errorbar(aes(ymin=-CI.lower,
                    ymax=-CI.upper),color="black",width=0.3)+
  geom_text(aes(y=-CI.lower+0.01, label=signif),colour="black", size = 2)+ #, position = position_jitter(height=0.008,width =0)
  scale_fill_viridis_d(name=NULL,option = "magma",direction = -1,begin = 0.5,end = 0.9,guide="none")+
  scale_x_discrete(name="Emotions",
                   guide=guide_axis(angle = 90))+
  scale_y_continuous(name="Accuracy Drop",
                     labels = scales::percent)+
  labs(title = "Deterioration of inference performance\n(compared to random manipulation)")+
  theme_classic()+
  theme(legend.position = c(0.9,0.7),
        panel.background = element_blank(),
        plot.title = element_text(hjust = 0.5),
        title = element_text(family = "Helvetica",size = 6),
        text = element_text(family = "Helvetica",size = 6),
        axis.ticks.length.x=unit(0,units = "mm"))
p.AccDrop.4000.sentiment
ggsave(paste0(fpath,"4000_AccDrop_sentiment.pdf"), plot = p.AccDrop.4000.sentiment, width = 72, height = 50, units = "mm")
# for difference N
p.AccDrop.sentiment = ggplot(AccDrop.by.sentiment,
                                  aes(reorder(Sentiment,ACC.diff),
                                      -ACC.diff,
                                      fill="whatever"))+
  geom_col()+
  geom_errorbar(aes(ymin=-CI.lower,
                    ymax=-CI.upper),color="black",width=0.3)+
  geom_text(aes(y=-CI.lower+0.01, label=signif),colour="black", size = 2)+ #, position = position_jitter(height=0.008,width =0)
  scale_fill_viridis_d(name=NULL,option = "magma",direction = -1,begin = 0.5,end = 0.9,guide="none")+
  scale_x_discrete(name="Emotions",
                   guide=guide_axis(angle = 90))+
  scale_y_continuous(name="Accuracy Drop",
                     labels = scales::percent)+
  labs(title = "Deterioration of inference performance\n(compared to random manipulation)")+
  facet_grid(rows=vars(Criteria),scales = "free_y")+
  theme_classic()+
  theme(legend.position = c(0.9,0.7),
        panel.background = element_blank(),
        plot.title = element_text(hjust = 0.5),
        title = element_text(family = "Helvetica",size = 6),
        text = element_text(family = "Helvetica",size = 6),
        axis.ticks.length.x=unit(0,units = "mm"))
p.AccDrop.sentiment
ggsave(paste0(fpath,"All_AccDrop_sentiment.pdf"), plot = p.AccDrop.sentiment, width = 180, height =220, units = "mm")
# test the unimodality of causal effect of conceptual attributes on emotions
AccDrop.all.combination %>%
  group_by(Criteria,Feature) %>%
  summarise_at("ACC.diff",
               list(D = ~diptest::dip.test(.)$statistic[1],
                    p.value = ~diptest::dip.test(.)$p.value,
                    N = ~diptest::dip.test(.)$nobs)) %>%
  as.data.table %>%
  .[,"Number of Manipulated Neurons":=Criteria] %>%
  .[,"Conceptual Attribute":=Feature] %>%
  .[,D:=as.character(signif(D,3))] %>%
  .[,p.value:=as.character(signif(p.value,3))] -> Unimodality.AccDiff
export(Unimodality.AccDiff, paste0(fpath,"Unimodality_of_causal_effect.csv"),header=T)




# ACC drop by feature ----
# for N = 4000
p.AccDrop.4000.feature = ggplot(AccDrop.by.feature[Criteria==4000],
                                aes(y=reorder(Feature,ACC.diff),
                                    x=-ACC.diff))+
  geom_col(aes(fill=rank(ACC.diff)),color="black")+
  geom_errorbar(aes(xmin=-CI.lower.x,
                    xmax=-CI.upper.x),
                color="black",width=0.2)+
  geom_text(aes(x=-0.05, label=signif.x), colour="black", size = 2)+
  scale_fill_viridis_c(name=NULL,option = "magma",direction = -1,
                       breaks=c(1,14),labels=c("1st","14th"),guide=guide_colorbar(reverse = TRUE))+
  scale_y_discrete(name="Attribute-Specific Neurons",
                   position = "left",
                   limits=rev)+
  scale_x_continuous(name="Accuracy Drop",
                     labels = scales::percent,
                     position="bottom")+
  labs(title = "Language-based causal contribution\nto emotion inference")+
  theme_classic()+
  theme(legend.position = c(0.8,0.2),
        legend.key.size = unit(2, 'mm'),
        panel.background = element_blank(),
        plot.title = element_text(family = "Helvetica",size = 6,hjust = 0.5),
        title = element_text(family = "Helvetica",size = 6),
        text = element_text(family = "Helvetica",size = 6),
        axis.ticks.length.y=unit(0,units = "mm"))
p.AccDrop.4000.feature
ggsave(paste0(fpath,"4000_AccDrop_feature.pdf"), plot = p.AccDrop.4000.feature, width = 52, height = 60, units = "mm")
# for different N
p.AccDrop.feature = ggplot(AccDrop.by.feature,
                                aes(x=reorder(Feature,ACC.diff),
                                    y=-ACC.diff))+
  geom_col(aes(fill=ACC.diff),color="black")+
  geom_errorbar(aes(ymin=-CI.lower.x,
                    ymax=-CI.upper.x),
                color="black",width=0.2)+
  geom_text(aes(y=-0.05, label=signif.x), colour="black", size = 2)+
  scale_fill_viridis_c(name=NULL,option = "magma",direction = -1,
                       guide=guide_colorbar(reverse = TRUE))+
  scale_y_continuous(name="Accuracy Drop",
                     labels = scales::percent,
                     position = "left")+
  scale_x_discrete(name="Attribute-Specific Neurons",
                   position="bottom",
                   guide=guide_axis(angle = 90))+
  labs(title = "Language-based causal contribution to emotion inference")+
  facet_grid(vars(Criteria),scales = "free_y")+
  theme_classic()+
  theme(legend.position = "none",
        legend.key.size = unit(2, 'mm'),
        panel.background = element_blank(),
        plot.title = element_text(family = "Helvetica",size = 6,hjust = 0.5),
        title = element_text(family = "Helvetica",size = 6),
        text = element_text(family = "Helvetica",size = 6),
        axis.ticks.length.y=unit(0,units = "mm"))
p.AccDrop.feature
ggsave(paste0(fpath,"All_AccDrop_feature.pdf"), plot = p.AccDrop.feature, width = 180, height = 220, units = "mm")



# representational weights in human emotion-concept knowledge ----
p.feature.weight = ggplot(AccDrop.by.feature[Criteria==4000],
                          aes(y=reorder(Feature,tau),
                              ))+
  geom_col(aes(x=tau,fill=rank(ACC.diff)),color="black")+
  geom_errorbar(aes(xmin=CI.lower.y,xmax=CI.upper.y),color="black",width=0.2)+
  geom_text(aes(x=CI.upper.y+0.05, label=signif.y), colour="black", size = 2)+
  scale_y_discrete(name="Conceptual Attributes")+
  scale_x_continuous(name="Kendall's tau")+
  scale_fill_viridis_c(name=NULL,option = "magma",direction = -1,guide="none")+
  labs(title = "Weight in the mental representation\nof emotion concepts")+
  theme_classic()+
  theme(panel.background = element_blank(),
        plot.title = element_text(family = "Helvetica",size = 6,hjust = 0.5),
        axis.ticks.length.y=unit(0,units = "mm"),
        title = element_text(family = "Helvetica",size = 6),
        text = element_text(family = "Helvetica",size = 6))
p.feature.weight
ggsave(paste0(fpath,"Feature_weights_in_Human.pdf"), plot = p.feature.weight, width = 52, height = 60, units = "mm")



# compare with human ----
# for N = 4000 & sentiment = remorse
p.compare.remorse = ggplot(AccDrop.all.combination[(Criteria==4000)&(Sentiment=="remorse")],
                   aes(tau, -ACC.diff))+
  geom_smooth(color="#F9E855",method = "lm",se=T)+
  geom_point(aes(color=rank(ACC.diff)),size=0.8,shape=15)+
  scale_color_viridis_c(name = "Emotions",option = "magma",direction = -1,begin = 0.1,end = 0.9)+
  scale_x_continuous(name = NULL,n.breaks = 2)+
  scale_y_continuous(name = NULL,breaks = c(0,0.35),labels = scales::percent)+
  ggthemes::theme_few()+
  theme(legend.position="none",
        panel.background=element_blank(),
        plot.background=element_blank(),
        axis.ticks.length=unit(0,units = "mm"),
        title = element_text(family = "Helvetica",size=6),
        text = element_text(family = "Helvetica",size=6))
p.compare.remorse
ggsave(paste0(fpath,"Compare_to_human_4000remorse.pdf"), plot = p.compare.remorse, width = 25, height = 22, units = "mm")

# for N = 4000
p.compare.4000 = ggplot(AccDrop.all.combination[Criteria==4000],
                        aes(tau, -ACC.diff, group=Sentiment, color=Sentiment))+
  geom_smooth(method = "lm",se=F, linewidth=1.5)+
  scale_color_viridis_d(name = "Emotions",option = "cividis",direction = 1)+
  scale_x_continuous(name = "Kendall's tau")+
  scale_y_continuous(name = "Accuracy Drop",labels = scales::percent)+
  theme_classic()+
  theme(legend.position="none",
        title = element_text(family = "Helvetica",size=6),
        text = element_text(family = "Helvetica",size=6))
p.compare.4000
ggsave(paste0(fpath,"Compare_to_human_4000.pdf"), plot = p.compare.4000, width = 65, height = 60, units = "mm")

# For all N: average correlation coefficient and t-test
AccDrop.all.combination %>%
  group_by(Criteria,Sentiment) %>%
  summarise(.groups="keep",
            cor.r=cor.test(ACC.diff,tau)$estimate,
            cor.N=14,
            cor.t=cor.test(ACC.diff,tau)$statistic,
            cor.df=cor.test(ACC.diff,tau)$parameter,
            cor.p=cor.test(ACC.diff,tau)$p.value,
            cor.Z=FisherZ(cor.test(ACC.diff,tau)$estimate)) %>%
  as.data.table %>%
  .[,"Type":="individual"] -> a
a %>%
  group_by(Criteria) %>%
  summarise(Sentiment="average",
            cor.r=CorCI(FisherZInv(mean(cor.Z)),14)[1],
            cor.N=NA,
            cor.t=NA,
            cor.df=NA,
            cor.p=NA,
            cor.Z=mean(cor.Z),
            Type="average") %>%
  as.data.table -> b
a %>%
  group_by(Criteria) %>%
  summarise(mean.corZ = t.test(cor.Z)$estimate[1],
            t.value = t.test(cor.Z)$statistic[1],
            df = t.test(cor.Z)$parameter[1],
            SE = t.test(cor.Z)$stderr,
            p.value = t.test(cor.Z)$p.value,
            mean.rho = FisherZInv(t.test(cor.Z)$estimate[1]),
            lwr.rho = FisherZInv(t.test(cor.Z)$conf.int[1]),
            upr.rho = FisherZInv(t.test(cor.Z)$conf.int[2])) %>%
  as.data.table -> c
list(individual.rs = a,
     average.rs = b,
     t.test = c) -> FisherCor
rm(a,b,c)
data.table("Criteria"=0,mean.corZ=0,t.value=0,df=26,SE=0,p.value=1,mean.rho=0,lwr.rho=0,upr.rho=0) %>%
  list(FisherCor[[3]],.) %>%
  rbindlist() %>%
  .[,signif:=""] %>%
  .[p.value<.01,signif:="**"] %>%
  .[p.value<.001,signif:="***"] -> Compare.to.humans
p.compare.overall = ggplot(Compare.to.humans, aes(factor(Criteria), -mean.rho))+
  geom_col(fill="lightgrey", color="black")+ #show mean
  geom_jitter(data=FisherCor[["individual.rs"]],
              aes(x=factor(Criteria),y=-cor.r,color=Sentiment),
              alpha=0.7,width = 0.15,size=1)+ #show samples
  # geom_text(aes(y=0.75, label=signif), colour="black", size = 2)+
  geom_errorbar(aes(ymin=-lwr.rho, ymax=-upr.rho), color="black",width=0.2)+ #show CI
  scale_color_viridis_d(name = "Emotions",option = "cividis")+
  scale_x_discrete(name="Number of Manipulated Neurons")+
  scale_y_continuous(name="Correlation Coefficient")+#)+#limits = c(-0.75,NA),
  guides(color=guide_legend(override.aes=list(shape=15,alpha=1)))+
  theme_classic()+
  theme(legend.position="none",
        title = element_text(family = "Helvetica",size=6),
        text = element_text(family = "Helvetica",size=6))
p.compare.overall
ggsave(paste0(fpath,"Compare_to_human_overall.pdf"), plot = p.compare.overall, width = 65, height = 60, units = "mm")



  
  
  






# SI Appendix (unfinished) ----
# raw ACC
masked.accuracy %>%
  group_by(Sentiment,Feature,Criteria) %>%
  summarise_at("Accuracy",
               list(ACC = ~mean(.))) %>%
  as.data.table -> ACC.raw
random.accuracy %>%
  group_by(Sentiment,Criteria) %>%
  summarise_at("Accuracy",
               list(ACC = ~mean(.))) %>%
  as.data.table %>%
  .[,Feature:="RANDOM"] %>%
  rbind(ACC.raw) -> ACC.raw
origin.accuracy %>%
  group_by(Sentiment) %>%
  summarise_at("Accuracy",
               list(ACC = ~mean(.))) %>%
  as.data.table %>%
  .[,Feature:="RAW"] %>%
  slice(rep(1:n(), length(criteria))) ->a
a[["Criteria"]] = rep(criteria,each=27)
ACC.raw = rbind(a,ACC.raw)
rm(a)
# decrease in ACC (minus origin ACC)
random.accuracy %>%
  merge(origin.accuracy,by=c("Sentiment","PromptID")) %>%
  .[,ACC.decrease:=Accuracy.x-Accuracy.y] %>%
  .[,c("Type.x","Accuracy.x","Type.y","Accuracy.y"):=NULL] %>%
  group_by(Sentiment,Criteria) %>%
  summarise_at("ACC.decrease",
               list(ACC.decrease = ~mean(.))) %>%
  as.data.table %>%
  .[,Feature:="RANDOM"] -> a
masked.accuracy %>%
  merge(origin.accuracy,by=c("Sentiment","PromptID")) %>%
  .[,ACC.decrease:=Accuracy.x-Accuracy.y] %>%
  .[,c("Type.x","Accuracy.x","Type.y","Accuracy.y"):=NULL] %>%
  group_by(Sentiment,Feature,Criteria) %>%
  summarise_at("ACC.decrease",
               list(ACC.decrease = ~mean(.))) %>%
  as.data.table %>%
  rbind(a) %>%
  .[,Feature:=factor(Feature,levels = c("RANDOM",index.features))] -> ACC.decrease
rm(a)
# difference in ACC (minus random ACC)
masked.accuracy %>%
  merge(random.accuracy,
        by=c("Sentiment","PromptID","Feature","Criteria")) %>%
  group_by(Sentiment,Feature,Criteria) %>%
  summarise(.groups = "keep",
            ACC.diff = t.test(Accuracy.x, Accuracy.y, alternative="less", paired=T)$estimate[1],
            t.value = t.test(Accuracy.x, Accuracy.y, alternative="less", paired=T)$statistic[1],
            df = t.test(Accuracy.x, Accuracy.y, alternative="less", paired=T)$parameter[1],
            SE = t.test(Accuracy.x, Accuracy.y, alternative="less", paired=T)$stderr,
            p.value = t.test(Accuracy.x, Accuracy.y, alternative="less", paired=T)$p.value) %>%
  as.data.table %>%
  .[,q.value:=p.value*.N/rank(p.value)] %>%
  .[q.value<0.05,signif:="*"]  -> ACC.difference
export(ACC.difference,file = paste0(fpath,"ACC.difference.csv"))
# raw ACC
p.ACC.raw = ggplot(ACC.raw, aes(Feature,factor(Criteria),fill=ACC), group=Sentiment)+
  geom_tile(color="white")+
  scale_x_discrete(name="Manipulation Type / Conceptual Attributes",guide=guide_axis(angle = 90))+
  scale_y_discrete(name = "Number of Manipulated Neurons",limits=rev)+
  scale_fill_viridis_c(name = "Accuracy",option = "magma",direction = -1,labels = scales::percent)+
  facet_wrap(~Sentiment,ncol = 3,shrink = T)+
  theme_minimal()+
  theme(title = element_text(family = "Helvetica",size = 7),
        text = element_text(family = "Helvetica",size=6))
ggsave(paste0(fpath,"ACC_raw.pdf"),plot = p.ACC.raw, width = 180, height =225, units = "mm")
# decrease in ACC
p.ACC.decrease = ggplot(ACC.decrease, aes(Feature,factor(Criteria),fill=ACC.decrease), group=Sentiment)+
  geom_tile(color="white")+
  scale_x_discrete(name="Manipulation Type / Conceptual Attributes",guide=guide_axis(angle = 90))+
  scale_y_discrete(name = "Number of Manipulated Neurons",limits=rev)+
  scale_fill_viridis_c(name = "Decrease\nin Accuracy",option = "magma",direction = -1,labels = scales::percent)+
  facet_wrap(~Sentiment,ncol = 3,shrink = T)+
  theme_minimal()+
  theme(title = element_text(family = "Helvetica",size = 7),
        text = element_text(family = "Helvetica",size=6))
ggsave(paste0(fpath,"ACC_decrease.pdf"),plot = p.ACC.decrease, width = 180, height =225, units = "mm")
# difference in ACC
p.ACC.difference = ggplot(ACC.difference, aes(Sentiment,Feature,fill=ACC.diff), group=factor(Criteria))+
  geom_tile(color="black")+
  geom_text(aes(label=signif), colour="black", stat = "identity",vjust=0.8)+
  scale_x_discrete(name="Emotion Inference Tasks",guide=guide_axis(angle = 90))+
  scale_y_discrete(name = "Conceptual Attributes",limits=rev)+
  scale_fill_gradient2(name = "Difference\nin Accuracy",labels = scales::percent)+
  facet_wrap(~Criteria,ncol = 2,shrink = T)+
  theme_minimal()+
  theme(title = element_text(family = "Helvetica",size = 7),
        text = element_text(family = "Helvetica",size=6))
ggsave(paste0(fpath,"ACC_difference.pdf"),plot = p.ACC.difference, width = 180, height =225, units = "mm")




# Draw plots for all N (no new insight)----
# ACC drop by sentiment (no new insight)
p.AccDrop.AllN.sentiment = ggplot(AccDrop.by.sentiment,
                                  aes(reorder(Sentiment,ACC.diff),
                                      factor(Criteria),
                                      fill=-ACC.diff))+
  geom_tile(color="black")+
  geom_text(aes(label=signif),position="identity",colour="black", size = 2)+
  scale_fill_viridis_c(name="Accuracy Drop",option = "cividis",direction = 1,
                       labels = scales::percent)+
  scale_x_discrete(name="Semantic Emotions",
                   guide=guide_axis(angle = 90))+
  scale_y_discrete(name="Number of Manipulated Neurons")+
  labs(title = "Deterioration of inference performance\n(compared to random manipulation)")+
  theme_minimal()+
  theme(panel.background = element_blank(),
        plot.title = element_text(hjust = 0.5),
        title = element_text(family = "Helvetica",size = 6),
        text = element_text(family = "Helvetica",size = 6),
        axis.ticks.length=unit(0,units = "mm"))
p.AccDrop.AllN.sentiment
# ggsave(paste0(fpath,"AllN_AccDrop_sentiment.pdf"), plot = p.AccDrop.AllN.sentiment, width = 72, height = 50, units = "mm")

# Acc drop by feature (no new insight)
p.AccDrop.AllN.feature = ggplot(AccDrop.by.feature,
                                aes(Feature,
                                    factor(Criteria),
                                    fill=rank(-ACC.diff)))+
  geom_tile(color="black")+
  geom_text(aes(label=signif),position="identity",colour="black", size = 2)+
  scale_fill_viridis_c(name="Accuracy Drop",option = "magma",direction = 1,
                       labels = scales::percent)+
  scale_y_discrete(name="Number of Manipulated Neurons")+
  scale_x_discrete(name="Attribute-Specific Neurons",
                   guide=guide_axis(angle = 90))+
  labs(title = "Causal contribution\nto inferring semantic emotions")+
  theme_minimal()+
  theme(panel.background = element_blank(),
        plot.title = element_text(family = "Helvetica",size = 6,hjust = 0.5),
        title = element_text(family = "Helvetica",size = 6),
        text = element_text(family = "Helvetica",size = 6),
        axis.ticks.length.y=unit(0,units = "mm"))
p.AccDrop.AllN.feature = ggplot(AccDrop.by.feature,
                                aes(Feature,
                                    -ACC.diff,
                                    group=factor(Criteria),
                                    color=factor(Criteria)))+
  geom_point()+
  geom_line()+
  scale_color_viridis_d(name="Number of Manipulated Neuronsp",option = "viridis",
                        direction = 1)+
  scale_y_continuous(name="Accuracy Drop")+
  scale_x_discrete(name="Attribute-Specific Neurons",
                   guide=guide_axis(angle = 90))+
  labs(title = "Causal contribution\nto inferring semantic emotions")+
  theme_classic()+
  theme(panel.background = element_blank(),
        plot.title = element_text(family = "Helvetica",size = 6,hjust = 0.5),
        title = element_text(family = "Helvetica",size = 6),
        text = element_text(family = "Helvetica",size = 6),
        axis.ticks.length.y=unit(0,units = "mm"))
p.AccDrop.AllN.feature = ggplot(AccDrop.by.feature,
                                aes(tau,
                                    -ACC.diff,
                                    group=factor(Criteria),
                                    color=factor(Criteria)))+
  geom_point()+
  geom_line()+
  scale_color_viridis_d(name="Number of Manipulated Neuronsp",option = "viridis",
                        direction = 1)+
  scale_y_continuous(name="Accuracy Drop")+
  scale_x_continuous(name="Kendall's tau")+
  # labs(title = "Causal contribution\nto inferring semantic emotions")+
  theme_classic()+
  theme(panel.background = element_blank(),
        plot.title = element_text(family = "Helvetica",size = 6,hjust = 0.5),
        title = element_text(family = "Helvetica",size = 6),
        text = element_text(family = "Helvetica",size = 6),
        axis.ticks.length.y=unit(0,units = "mm"))
p.AccDrop.AllN.feature
# ggsave(paste0(fpath,"AllN_AccDrop_feature.pdf"), plot = p.AccDrop.AllN.feature, width = 45, height = 60, units = "mm")

# p=ggplot(AccDrop.all.combination,
#          aes(x=reorder(Feature,ACC.diff),y=-ACC.diff,group=reorder(Sentiment,ACC.diff),color=reorder(Sentiment,ACC.diff)))+
#   geom_line(alpha=.8,color="grey")+
#   geom_point(alpha=.8)+
#   scale_color_viridis_d(name="Semantic Emotions",
#                         option = "cividis",
#                         direction = 1,begin = .1,end = .9)+
#   scale_x_discrete(name="Attribute-Specific Neurons",guide=guide_axis(angle = 90))+
#   scale_y_continuous(name="Accuracy Drop",labels = scales::percent)+
#   facet_wrap(~Criteria)+
#   theme_bw()
# p
# AccDrop.all.combination[,Effect:="no-change"]
# AccDrop.all.combination[CI.lower>0,Effect:="raise"]
# AccDrop.all.combination[CI.upper<0,Effect:="drop"]
# p=ggplot(AccDrop.all.combination,
#          aes(x=Criteria,y=-ACC.diff,
#              group=reorder(Sentiment,ACC.diff),
#              color=Effect))+
#   geom_line(alpha=.5,color="grey")+
#   geom_point(alpha=.8)+
#   scale_color_viridis_d(name="Performance",
#                         option = "magma",
#                         direction = -1,begin = .1,end = .9)+
#   scale_x_continuous(name="Number of Manipulated Neurons",guide=guide_axis(angle = 90))+
#   scale_y_continuous(name="Accuracy Drop",labels = scales::percent)+
#   facet_wrap(~reorder(Feature,ACC.diff))+
#   theme_bw()
# p


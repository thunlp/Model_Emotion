library(bruceR)
library(patchwork)
library(boot)
library(Rtsne)
library(gstat)
set.wd()
set.wd("../")

## load data --------
# load RDMs of subjective comparison
subjective.comparison = list(avg=import("Data/preprocessed/subjective_comparison/avg_dist_subjective_comparison.csv"),
                             ind=import("Data/preprocessed/subjective_comparison/ind_dist_subjective_comparison.csv"))
# load RDMs of Pre-trained Language Model
word.embedding = list(avg=import("Data/preprocessed/PLM/word_embedding_GloVe/avg_dist_reddit768d.csv"),
                      ind=import("Data/preprocessed/PLM/word_embedding_GloVe/ind_dist_reddit768d.csv",header = T))
prompt.embedding = list(avg=import("Data/preprocessed/PLM/prompt_tuning_RoBERTa/prompt_embedding/avg_dist_prompt_embedding.csv"),
                        ind=import("Data/preprocessed/PLM/prompt_tuning_RoBERTa/prompt_embedding/ind_dist_prompt_embedding.csv",header = T))
neurons.activation = list(avg=import("Data/preprocessed/PLM/prompt_tuning_RoBERTa/neurons_activation/whole_model/avg_dist_neurons_activation.csv"),
                        ind=import("Data/preprocessed/PLM/prompt_tuning_RoBERTa/neurons_activation/whole_model/ind_dist_neurons_activation.csv",header = T))
# load control RDM
corpus.annotation = import("Data/preprocessed/control/dist_GoEmotions_annotation.csv")
# load RDMs of emotional features
emotional.features = import("Data/preprocessed/emotional_features/dist_emotional_features.csv")
PCs.emotion = import("Data/preprocessed/emotional_features/dist_emotional_PCs.csv")
emotion.space = import("Data/preprocessed/emotional_features/dist_emotion_space.csv")
affective = data.frame(emotion.space$`Affective Space`)
colnames(affective) = "Affective Space"
basic = data.frame(emotion.space$`Basic Emotions Space`)
colnames(basic) = "Basic Emotions Space"
appraisal = data.frame(emotion.space$`Appraisal Space`)
colnames(appraisal) = "Appraisal Space"
combined = data.frame(emotion.space$`Combined Space`)
colnames(combined) = "Combined Space"
# load emotion categories
emotions = unlist(import("Data/preprocessed/reordered_emotions.csv",header = F))

## visualize RDM of representational emotion space -----
RDM.visualize = list(subjective.comparison$avg,
                     # corpus.annotation,
                     word.embedding$avg,
                     prompt.embedding$avg,
                     neurons.activation$avg)
k=length(RDM.visualize)
for (index in c(1:k)){
  RDM.matrix = matrix(0, nrow = 27, ncol = 27, dimnames = list(emotions,emotions))
  RDM.matrix[lower.tri(RDM.matrix)] = unlist(RDM.visualize[[index]])
  RDM.matrix = RDM.matrix + t(RDM.matrix)
  RDM.matrix = EBImage::equalize(RDM.matrix, range = c(0, max(RDM.matrix)), levels = 351)
  plot.data = expand.grid(x=emotions,y=emotions)
  plot.data[,"dist"] = as.vector(RDM.matrix)
  if(index==1){
    p.RDM = ggplot(plot.data, aes(x=x,y=y,fill=dist))+
      geom_tile()+
      scale_fill_viridis_c(option = "magma",name="Percentile\nof distance",breaks=c(min(plot.data$dist),max(plot.data$dist)),labels=c("Min","Max"))+
      scale_x_discrete(name="",guide=guide_axis(angle = 90))+
      scale_y_discrete(name="")+
      labs(title = colnames(RDM.visualize[[index]]))+
      coord_fixed(ratio = 1) + 
      theme(panel.background=element_blank(), panel.grid = element_blank(), axis.ticks.length=unit(0,units = "mm"))
  }else if(index<=k){
    p.RDM = p.RDM + ggplot(plot.data, aes(x=x,y=y,fill=dist))+
      geom_tile()+
      scale_fill_viridis_c(option = "magma")+
      scale_x_discrete(name="",labels=NULL)+
      scale_y_discrete(name="",labels=NULL)+
      labs(title = sub('[(]','\n(',colnames(RDM.visualize[[index]])))+
      coord_fixed(ratio = 1) + 
      theme(panel.background=element_blank(), panel.grid = element_blank(), axis.ticks.length=unit(0,units = "mm"),legend.position='none')
  }
}
design="
AAAB
AAAC
AAAD
"
p.RDM = p.RDM + plot_layout(design = design, guides = "collect")
ggsave("Result/RSA_whole_model/RDMs_of_Human_Behavior_and_PLM.pdf",plot = p.RDM, width = 240, height = 230, units = "mm")
rm(design,index,plot.data,RDM.matrix,RDM.visualize,k,affective,basic,appraisal)

## RSA (representational similarity analysis) with emotional features as candidates ----
# inferential statistic
RDMs.inference = list(subjective.comparison,
                      corpus.annotation,
                      word.embedding,
                      prompt.embedding,
                      neurons.activation)
RSA = list(RDM.inferenced=c(colnames(subjective.comparison$avg),
                            colnames(corpus.annotation),
                            colnames(word.embedding$avg),
                            colnames(prompt.embedding$avg),
                            colnames(neurons.activation$avg)),
           tau=matrix(nrow = length(RDMs.inference),ncol = ncol(emotional.features)),
           p=matrix(nrow = length(RDMs.inference),ncol = ncol(emotional.features)))
boot.emotion <- function(RDM.inference, index.e, RDM.candidate) {
  RDM.inference <- RDM.inference[index.e,index.e]
  RDM.inference = RDM.inference[lower.tri(RDM.inference)]
  RDM.candidate <- RDM.candidate[index.e,index.e]
  RDM.candidate = RDM.candidate[lower.tri(RDM.candidate)]
  r <- cor(RDM.inference,RDM.candidate,method = "kendall")
  return(r)
}
boot.subject <- function(RDMs.inference, index.s, RDM.candidate) {
  new.RDM <- colMeans(RDMs.inference[index.s,])
  RDM.matrix = matrix(0,nrow = 27,ncol = 27)
  RDM.matrix[lower.tri(RDM.matrix)] = new.RDM
  RDM.inference = RDM.matrix+t(RDM.matrix)
  RDM.matrix = matrix(0,nrow = 27,ncol = 27)
  RDM.matrix[lower.tri(RDM.matrix)] = RDM.candidate
  RDM.candidate = RDM.matrix+t(RDM.matrix)
  r <- boot(RDM.inference, boot.emotion, R=1, RDM.candidate=RDM.candidate)
  r <- r$t0
  return(r)
}
for (index.RDM in c(1:length(RDMs.inference))) {
  if (!is.data.frame(RDMs.inference[[index.RDM]])) {
    for (index.feature in c(1:ncol(emotional.features))) {
      set.seed(6)
      compare.result = boot(RDMs.inference[[index.RDM]]$ind, boot.subject, R=10000, RDM.candidate=emotional.features[,index.feature])
      RSA$tau[index.RDM,index.feature] = compare.result$t0
      RSA$p[index.RDM,index.feature] = sum(compare.result$t<=0)/10000
      print(paste("index.RDM =",index.RDM,"index.feature =",index.feature))
    }
  }else{
    for (index.feature in c(1:ncol(emotional.features))) {
      set.seed(6)
      RDM.matrix = matrix(0,nrow = 27,ncol = 27)
      RDM.matrix[lower.tri(RDM.matrix)] = RDMs.inference[[index.RDM]][,1]
      RDM.inference = RDM.matrix+t(RDM.matrix)
      RDM.matrix = matrix(0,nrow = 27,ncol = 27)
      RDM.matrix[lower.tri(RDM.matrix)] = emotional.features[,index.feature]
      RDM.candidate = RDM.matrix+t(RDM.matrix)
      compare.result = boot(RDM.inference, boot.emotion, R=10000, RDM.candidate=RDM.candidate)
      RSA$tau[index.RDM,index.feature] = compare.result$t0
      RSA$p[index.RDM,index.feature] = sum(compare.result$t<=0)/10000
      print(paste("index.RDM =",index.RDM,"index.feature =",index.feature))
    }
  }
}
# p-value correction by FDR (false discovery rate)
RSA[["q"]] = RSA$p * length(RSA$p) / rank(RSA$p)
# save RSA results
rownames(RSA$tau)=RSA$RDM.inferenced
rownames(RSA$p)=RSA$RDM.inferenced
rownames(RSA$q)=RSA$RDM.inferenced
colnames(RSA$tau)=colnames(emotional.features)
colnames(RSA$p)=colnames(emotional.features)
colnames(RSA$q)=colnames(emotional.features)
export(t(RSA$tau),"Result/RSA_whole_model/RSA_with_all_emotional_features_tau.csv",header = T)
export(t(RSA$p),"Result/RSA_whole_model/RSA_with_all_emotional_features_p.csv",header = T)
export(t(RSA$q),"Result/RSA_whole_model/RSA_with_all_emotional_features_q.csv",header = T)
rm(boot.emotion,boot.subject,index.RDM,index.feature,RDM.matrix,RDM.inference,RDM.candidate,RDMs.inference,compare.result)
# visualize RSA results
plot.data = expand.grid(RDM=RSA$RDM.inferenced, feature=colnames(emotional.features))
plot.data[,"tau"] = as.vector(RSA$tau)
plot.data[plot.data$tau<0,"tau"] = 0
plot.data[as.vector(RSA$q)>=0.05,"signif"] = ""
plot.data[as.vector(RSA$q)<0.05,"signif"] = "*"
plot.data[as.vector(RSA$q)<0.01,"signif"] = "**"
plot.data[as.vector(RSA$q)<0.001,"signif"] = "***"
p.RSA = ggplot(plot.data, aes(x=feature,y=RDM,fill=tau))+
  geom_tile()+
  geom_text(aes(label=signif), colour="black", stat = "identity",size = 2,vjust=0.65)+
  scale_fill_viridis_c(option = "rocket",begin=1-max(plot.data$tau),end=1,direction = -1,name="Kendall's tau",breaks=c(0,max(plot.data$tau)),labels=c("0",round(max(plot.data$tau),2)))+
  scale_x_discrete(name="",guide=guide_axis(angle = 90),position = "top")+
  scale_y_discrete(name="",limits=rev)+
  theme(panel.background=element_blank(), panel.grid = element_blank(), axis.ticks.length=unit(0,units = "mm"))
ggsave("Result/RSA_whole_model/RSA_with_all_emotional_features.pdf",plot = p.RSA, width = 270, height = 135, units = "mm")
rm(plot.data,p.RSA)

## RSA (representational similarity analysis) with emotional PCs as candidates ----
# inferential statistic
RDMs.inference = list(subjective.comparison,
                      # corpus.annotation,
                      word.embedding,
                      prompt.embedding,
                      neurons.activation)
RSA = list(RDM.inferenced=c(colnames(subjective.comparison$avg),
                            # colnames(corpus.annotation),
                            colnames(word.embedding$avg),
                            colnames(prompt.embedding$avg),
                            colnames(neurons.activation$avg)),
           tau=matrix(nrow = length(RDMs.inference),ncol = ncol(PCs.emotion)),
           se=matrix(nrow = length(RDMs.inference),ncol = ncol(PCs.emotion)),
           p=matrix(nrow = length(RDMs.inference),ncol = ncol(PCs.emotion)))
boot.emotion <- function(RDM.inference, index.e, RDM.candidate) {
  RDM.inference <- RDM.inference[index.e,index.e]
  RDM.inference = RDM.inference[lower.tri(RDM.inference)]
  RDM.candidate <- RDM.candidate[index.e,index.e]
  RDM.candidate = RDM.candidate[lower.tri(RDM.candidate)]
  r <- cor(RDM.inference,RDM.candidate,method = "kendall")
  return(r)
}
boot.subject <- function(RDMs.inference, index.s, RDM.candidate) {
  new.RDM <- colMeans(RDMs.inference[index.s,])
  RDM.matrix = matrix(0,nrow = 27,ncol = 27)
  RDM.matrix[lower.tri(RDM.matrix)] = new.RDM
  RDM.inference = RDM.matrix+t(RDM.matrix)
  RDM.matrix = matrix(0,nrow = 27,ncol = 27)
  RDM.matrix[lower.tri(RDM.matrix)] = RDM.candidate
  RDM.candidate = RDM.matrix+t(RDM.matrix)
  r <- boot(RDM.inference, boot.emotion, R=1, RDM.candidate=RDM.candidate)
  r <- r$t0
  return(r)
}
for (index.RDM in c(1:length(RDMs.inference))) {
  if (!is.data.frame(RDMs.inference[[index.RDM]])) {
    for (index.feature in c(1:ncol(PCs.emotion))) {
      set.seed(6)
      compare.result = boot(RDMs.inference[[index.RDM]]$ind, boot.subject, R=10000, RDM.candidate=PCs.emotion[,index.feature])
      RSA$tau[index.RDM,index.feature] = compare.result$t0
      RSA$se[index.RDM,index.feature] = sd(compare.result$t)
      RSA$p[index.RDM,index.feature] = sum(compare.result$t<=0)/10000
      print(paste("index.RDM =",index.RDM,"index.feature =",index.feature))
    }
  }else{
    for (index.feature in c(1:ncol(PCs.emotion))) {
      set.seed(6)
      RDM.matrix = matrix(0,nrow = 27,ncol = 27)
      RDM.matrix[lower.tri(RDM.matrix)] = RDMs.inference[[index.RDM]][,1]
      RDM.inference = RDM.matrix+t(RDM.matrix)
      RDM.matrix = matrix(0,nrow = 27,ncol = 27)
      RDM.matrix[lower.tri(RDM.matrix)] = PCs.emotion[,index.feature]
      RDM.candidate = RDM.matrix+t(RDM.matrix)
      compare.result = boot(RDM.inference, boot.emotion, R=10000, RDM.candidate=RDM.candidate)
      RSA$tau[index.RDM,index.feature] = compare.result$t0
      RSA$se[index.RDM,index.feature] = sd(compare.result$t)
      RSA$p[index.RDM,index.feature] = sum(compare.result$t<=0)/10000
      print(paste("index.RDM =",index.RDM,"index.feature =",index.feature))
    }
  }
}
# p-value correction by FDR (false discovery rate)
RSA[["q"]] = RSA$p * length(RSA$p) / rank(RSA$p)
# save RSA results
rownames(RSA$tau)=RSA$RDM.inferenced
rownames(RSA$se)=RSA$RDM.inferenced
rownames(RSA$p)=RSA$RDM.inferenced
rownames(RSA$q)=RSA$RDM.inferenced
colnames(RSA$tau)=colnames(PCs.emotion)
colnames(RSA$se)=colnames(PCs.emotion)
colnames(RSA$p)=colnames(PCs.emotion)
colnames(RSA$q)=colnames(PCs.emotion)
export(t(RSA$tau),"Result/RSA_whole_model/RSA_with_emotional_PCs_tau.csv",header = T)
export(t(RSA$se),"Result/RSA_whole_model/RSA_with_emotional_PCs_se.csv",header = T)
export(t(RSA$p),"Result/RSA_whole_model/RSA_with_emotional_PCs_p.csv",header = T)
export(t(RSA$q),"Result/RSA_whole_model/RSA_with_emotional_PCs_q.csv",header = T)
rm(boot.emotion,boot.subject,index.RDM,index.feature,RDM.matrix,RDM.inference,RDM.candidate,RDMs.inference,compare.result)
# visualize RSA results
plot.data = expand.grid(RDM=RSA$RDM.inferenced, feature=colnames(PCs.emotion))
plot.data[,"tau"] = as.vector(RSA$tau)
plot.data[plot.data$tau<0,"tau"] = 0
plot.data[as.vector(RSA$q)>=0.05,"signif"] = ""
plot.data[as.vector(RSA$q)<0.05,"signif"] = "*"
plot.data[as.vector(RSA$q)<0.01,"signif"] = "**"
plot.data[as.vector(RSA$q)<0.001,"signif"] = "***"
p.RSA = ggplot(plot.data, aes(x=feature,y=RDM,fill=tau))+
  geom_tile()+
  geom_text(aes(label=signif), colour="black", stat = "identity",size = 2,vjust=0.65)+
  scale_fill_viridis_c(option = "rocket",begin=1-max(plot.data$tau),end=1,direction = -1,name="Kendall's tau",breaks=c(0,max(plot.data$tau)),labels=c("0",round(max(plot.data$tau),2)))+
  scale_x_discrete(name="",guide=guide_axis(angle = 90),position = "top")+
  scale_y_discrete(name="",limits=rev)+
  theme(panel.background=element_blank(), panel.grid = element_blank(), axis.ticks.length=unit(0,units = "mm"))
ggsave("Result/RSA_whole_model/RSA_with_emotional_PCs.pdf",plot = p.RSA, width = 270, height = 135, units = "mm")
rm(plot.data,p.RSA)
# visualize RSA results
library(ggsci)
plot.data = expand.grid(RDM=RSA$RDM.inferenced, feature=colnames(PCs.emotion))
plot.data$RDM = factor(plot.data$RDM, levels = c("Neurons Activation","Word Embedding","Subjective Comparison"))
plot.data[,"tau"] = as.vector(RSA$tau)
plot.data[,"se"] = as.vector(RSA$se)
plot.data[as.vector(RSA$q)>=0.05,"signif"] = ""
plot.data[as.vector(RSA$q)<0.05,"signif"] = "*"
plot.data[as.vector(RSA$q)<0.01,"signif"] = "**"
plot.data[as.vector(RSA$q)<0.001,"signif"] = "***"
# plot.data$signif = factor(plot.data$signif,levels = c("","*"))
# plot.data$feature = factor(plot.data$feature,
                           # levels=colnames(PCs.emotion)[order(plot.data$tau[plot.data$RDM=="Subjective Comparison"])])
p.RSA = ggplot(plot.data, aes(x=feature,y=tau,group=RDM,color=RDM,fill=RDM))+
  geom_hline(yintercept = 0, linetype="dashed")+
  geom_line(color="black",alpha=1,position = position_dodge(width = 0.4))+
  geom_errorbar(aes(ymin=tau-se/2, ymax=tau+se/2, width = 0.8),position = position_dodge(width = 0.4))+
  geom_point(size=2.5,shape=21,colour="black",position = position_dodge(width = 0.4))+
  geom_text(aes(y=tau+0.03+se/2, label=signif), colour="black", size = 4,position = position_dodge(width = 0.3))+
  scale_color_npg(name="inferenced RDM")+
  scale_fill_npg(name="inferenced RDM")+
  scale_x_discrete(name="",guide=guide_axis(angle = 90))+
  scale_y_continuous(name="Kendall's tau",n.breaks = 6,minor_breaks=NULL)+
  theme_bw()+
  theme(legend.position = "none")+
  facet_grid(RDM~.,scales="free_y")
p.RSA
ggsave("Result/RSA_whole_model/RSA_with_emotional_PCs_errorbar.pdf",plot = p.RSA, width = 200, height = 190, units = "mm")
rm(plot.data)



## RSA (representational similarity analysis) with emotion space as candidates ----
# inferential statistic
RDMs.inference = list(subjective.comparison,
                      # corpus.annotation,
                      word.embedding,
                      # prompt.embedding,
                      neurons.activation)
RSA = list(RDM.inferenced=c(colnames(subjective.comparison$avg),
                            # colnames(corpus.annotation),
                            colnames(word.embedding$avg),
                            # colnames(prompt.embedding$avg),
                            colnames(neurons.activation$avg)),
           tau=matrix(nrow = length(RDMs.inference),ncol = ncol(emotion.space)),
           se=matrix(nrow = length(RDMs.inference),ncol = ncol(emotion.space)),
           p=matrix(nrow = length(RDMs.inference),ncol = ncol(emotion.space)))
boot.emotion <- function(RDM.inference, index.e, RDM.candidate) {
  RDM.inference <- RDM.inference[index.e,index.e]
  RDM.inference = RDM.inference[lower.tri(RDM.inference)]
  RDM.candidate <- RDM.candidate[index.e,index.e]
  RDM.candidate = RDM.candidate[lower.tri(RDM.candidate)]
  r <- cor(RDM.inference,RDM.candidate,method = "kendall")
  return(r)
}
boot.subject <- function(RDMs.inference, index.s, RDM.candidate) {
  new.RDM <- colMeans(RDMs.inference[index.s,])
  RDM.matrix = matrix(0,nrow = 27,ncol = 27)
  RDM.matrix[lower.tri(RDM.matrix)] = new.RDM
  RDM.inference = RDM.matrix+t(RDM.matrix)
  RDM.matrix = matrix(0,nrow = 27,ncol = 27)
  RDM.matrix[lower.tri(RDM.matrix)] = RDM.candidate
  RDM.candidate = RDM.matrix+t(RDM.matrix)
  r <- boot(RDM.inference, boot.emotion, R=1, RDM.candidate=RDM.candidate)
  r <- r$t0
  return(r)
}
for (index.RDM in c(1:length(RDMs.inference))) {
  if (!is.data.frame(RDMs.inference[[index.RDM]])) {
    for (index.feature in c(1:ncol(emotion.space))) {
      set.seed(6)
      compare.result = boot(RDMs.inference[[index.RDM]]$ind, boot.subject, R=100, RDM.candidate=emotion.space[,index.feature])
      RSA$tau[index.RDM,index.feature] = compare.result$t0
      RSA$se[index.RDM,index.feature] = sd(compare.result$t)
      RSA$p[index.RDM,index.feature] = sum(compare.result$t<=0)/100
      print(paste("index.RDM =",index.RDM,"index.feature =",index.feature))
    }
  }else{
    for (index.feature in c(1:ncol(emotion.space))) {
      set.seed(6)
      RDM.matrix = matrix(0,nrow = 27,ncol = 27)
      RDM.matrix[lower.tri(RDM.matrix)] = RDMs.inference[[index.RDM]][,1]
      RDM.inference = RDM.matrix+t(RDM.matrix)
      RDM.matrix = matrix(0,nrow = 27,ncol = 27)
      RDM.matrix[lower.tri(RDM.matrix)] = emotion.space[,index.feature]
      RDM.candidate = RDM.matrix+t(RDM.matrix)
      compare.result = boot(RDM.inference, boot.emotion, R=100, RDM.candidate=RDM.candidate)
      RSA$tau[index.RDM,index.feature] = compare.result$t0
      RSA$se[index.RDM,index.feature] = sd(compare.result$t)
      RSA$p[index.RDM,index.feature] = sum(compare.result$t<=0)/100
      print(paste("index.RDM =",index.RDM,"index.feature =",index.feature))
    }
  }
}
# p-value correction by FDR (false discovery rate)
RSA[["q"]] = RSA$p * length(RSA$p) / rank(RSA$p)
# save RSA results
rownames(RSA$tau)=RSA$RDM.inferenced
rownames(RSA$se)=RSA$RDM.inferenced
rownames(RSA$p)=RSA$RDM.inferenced
rownames(RSA$q)=RSA$RDM.inferenced
colnames(RSA$tau)=colnames(emotion.space)
colnames(RSA$se)=colnames(emotion.space)
colnames(RSA$p)=colnames(emotion.space)
colnames(RSA$q)=colnames(emotion.space)
export(t(RSA$tau),"Result/RSA_whole_model/RSA_with_emotion_space_tau.csv",header = T)
export(t(RSA$se),"Result/RSA_whole_model/RSA_with_emotion_space_se.csv",header = T)
export(t(RSA$p),"Result/RSA_whole_model/RSA_with_emotion_space_p.csv",header = T)
export(t(RSA$q),"Result/RSA_whole_model/RSA_with_emotion_space_q.csv",header = T)
rm(boot.emotion,boot.subject,index.RDM,index.feature,RDM.matrix,RDM.inference,RDM.candidate,RDMs.inference,compare.result)
# visualize RSA results
plot.data = expand.grid(RDM=RSA$RDM.inferenced, feature=colnames(emotion.space))
plot.data[,"tau"] = as.vector(RSA$tau)
plot.data[plot.data$tau<0,"tau"] = 0
plot.data[as.vector(RSA$q)>=0.05,"signif"] = ""
plot.data[as.vector(RSA$q)<0.05,"signif"] = "*"
plot.data[as.vector(RSA$q)<0.01,"signif"] = "**"
plot.data[as.vector(RSA$q)<0.001,"signif"] = "***"
p.RSA = ggplot(plot.data, aes(x=feature,y=RDM,fill=tau))+
  geom_tile()+
  geom_text(aes(label=signif), colour="black", stat = "identity",size = 2,vjust=0.65)+
  scale_fill_viridis_c(option = "rocket",begin=1-max(plot.data$tau),end=1,direction = -1,name="Kendall's tau",breaks=c(0,max(plot.data$tau)),labels=c("0",round(max(plot.data$tau),2)))+
  scale_x_discrete(name="",guide=guide_axis(angle = 90),position = "top")+
  scale_y_discrete(name="",limits=rev)+
  theme(panel.background=element_blank(), panel.grid = element_blank(), axis.ticks.length=unit(0,units = "mm"))
ggsave("Result/RSA_whole_model/RSA_with_emotion_space.pdf",plot = p.RSA, width = 270, height = 135, units = "mm")
rm(plot.data,p.RSA)
# visualize RSA results
library(ggsci)
plot.data = expand.grid(RDM=RSA$RDM.inferenced, feature=colnames(emotion.space))
plot.data$RDM = factor(plot.data$RDM, levels = c("Neurons Activation","Word Embedding","Subjective Comparison"))
plot.data[,"tau"] = as.vector(RSA$tau)
plot.data[,"se"] = as.vector(RSA$se)
plot.data[as.vector(RSA$q)>=0.05,"signif"] = ""
plot.data[as.vector(RSA$q)<0.05,"signif"] = "*"
plot.data[as.vector(RSA$q)<0.01,"signif"] = "**"
plot.data[as.vector(RSA$q)<0.001,"signif"] = "***"
# plot.data$signif = factor(plot.data$signif,levels = c("","*"))
# plot.data$feature = factor(plot.data$feature,
# levels=colnames(emotion.space)[order(plot.data$tau[plot.data$RDM=="Subjective Comparison"])])
p.RSA = ggplot(plot.data, aes(x=RDM,y=tau,group=feature,color=feature,fill=feature))+
  geom_col(position = position_dodge(width = 1))+
  geom_errorbar(aes(ymin=tau-se/2, ymax=tau+se/2, width = 0.4),position = position_dodge(width = 1))+
  geom_text(aes(y=tau+0.03+se/2, label=signif), colour="black", size = 4, position = position_dodge(width = 1))+
  scale_color_npg(name="emotion space")+
  scale_fill_npg(name="emotion space")+
  scale_x_discrete(name="",guide=guide_axis(angle = 90))+
  scale_y_continuous(name="Kendall's tau")+
  theme_classic()
p.RSA
ggsave("Result/RSA_whole_model/RSA_with_emotion_space_errorbar.pdf",plot = p.RSA, width = 200, height = 190, units = "mm")
rm(plot.data)



## RSA between RDMs of representational emotion space ----
# directly compare different emotion space
boot.emotion <- function(RDM.inference, index.e, RDM.candidate) {
  RDM.inference <- RDM.inference[index.e,index.e]
  RDM.inference = RDM.inference[lower.tri(RDM.inference)]
  RDM.candidate <- RDM.candidate[index.e,index.e]
  RDM.candidate = RDM.candidate[lower.tri(RDM.candidate)]
  r <- cor(RDM.inference,RDM.candidate,method = "pearson")
  return(r)
}
boot.subject <- function(RDMs.inference, index.s, RDM.candidate) {
  new.RDM <- colMeans(RDMs.inference[index.s,])
  RDM.matrix = matrix(0,nrow = 27,ncol = 27)
  RDM.matrix[lower.tri(RDM.matrix)] = new.RDM
  RDM.inference = RDM.matrix+t(RDM.matrix)
  RDM.matrix = matrix(0,nrow = 27,ncol = 27)
  RDM.matrix[lower.tri(RDM.matrix)] = RDM.candidate
  RDM.candidate = RDM.matrix+t(RDM.matrix)
  r <- boot(RDM.inference, boot.emotion, R=1, RDM.candidate=RDM.candidate)
  r <- r$t0
  return(r)
}
RSA.total = data.table(compare=factor(c("Neurons Activation","Prompt Embedding","Word Embedding"),
                                      levels = c("Neurons Activation","Prompt Embedding","Word Embedding")),
                       r=c(0,0,0), se=c(0,0,0), p=c(0,0,0))
set.seed(6)
compare.result = boot(subjective.comparison$ind, boot.subject, R=10000, RDM.candidate=neurons.activation$avg$`Neurons Activation`)
RSA.total$r[1] = compare.result$t0
RSA.total$se[1] = sd(compare.result$t)
RSA.total$p[1] = sum(compare.result$t<=0)/10000
set.seed(6)
compare.result = boot(subjective.comparison$ind, boot.subject, R=10000, RDM.candidate=prompt.embedding$avg$`Prompt Embedding`)
RSA.total$r[2] = compare.result$t0
RSA.total$se[2] = sd(compare.result$t)
RSA.total$p[2] = sum(compare.result$t<=0)/10000
set.seed(6)
compare.result = boot(subjective.comparison$ind, boot.subject, R=10000, RDM.candidate=word.embedding$avg$`Word Embedding`)
RSA.total$r[3] = compare.result$t0
RSA.total$se[3] = sd(compare.result$t)
RSA.total$p[3] = sum(compare.result$t<=0)/10000
p = ggplot(RSA.total, aes(compare, r, fill = compare))+
  geom_col()+
  geom_errorbar(aes(ymin=r-se/2,ymax=r+se/2), width = 0.8)+
  scale_fill_npg(guide='none')+
  scale_x_discrete(name="",guide=guide_axis(angle = 90))+
  scale_y_continuous(name = "Kendall's tau",n.breaks = 7,expand = expansion(mult = c(0, .1)))+
  theme_classic()
ggsave("Result/RSA_whole_model/RSA_between_Human_Behavior_and_PLM.pdf",plot = p, width = 100, height = 190, units = "mm")



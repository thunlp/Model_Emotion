library(bruceR)
library(boot)
set.wd()
set.wd("../")

# load data
emotions = unlist(import("Data/preprocessed/reordered_emotions.csv",header = F))
emotional.features = import("Data/preprocessed/emotional_features/dist_emotional_features.csv")
PCs.emotion = import("Data/preprocessed/emotional_features/dist_emotional_PCs.csv")
emotion.space = import("Data/preprocessed/emotional_features/dist_emotion_space.csv")



# RSA (to emotional features) by layer ---------
# 
RSA.layer = list(tau=matrix(nrow = ncol(emotional.features), ncol = 12, dimnames = list(colnames(emotional.features),c(1:12))),
                 p=matrix(nrow = ncol(emotional.features), ncol = 12, dimnames = list(colnames(emotional.features),c(1:12))))
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
for (index.layer in c(1:12)) {
  RDM.layer = import(paste0("Data/preprocessed/PLM/prompt_tuning_RoBERTa/neurons_activation/by_layer/dist_of_layer_",index.layer,".csv"),header = T)
  for (index.feature in c(1:ncol(emotional.features))) {
    set.seed(6)
    compare.result = boot(RDM.layer, boot.subject, R=1000, RDM.candidate=emotional.features[,index.feature])
    RSA.layer$tau[index.feature,index.layer] = compare.result$t0
    RSA.layer$p[index.feature,index.layer] = sum(compare.result$t<=0)/1000
    print(paste("index.layer =",index.layer,"index.feature =",index.feature))
    rm(compare.result)
  }
}
# p-value correction by FDR (false discovery rate)
RSA.layer[["q"]] = RSA.layer$p * nrow(RSA.layer$p) * ncol(RSA.layer$p) / rank(RSA.layer$p)
# save RSA.layer results
rownames(RSA.layer$q)=rownames(RSA.layer$p)
colnames(RSA.layer$q)=colnames(RSA.layer$p)
export(t(RSA.layer$tau),"Result/RSA_by_layer/RSA_with_all_emotional_features_on_each_layer_tau.csv")
export(t(RSA.layer$p),"Result/RSA_by_layer/RSA_with_all_emotional_features_on_each_layer_p.csv")
# RSA.layer=list(tau=t(as.matrix(import("Result/RSA_by_layer/RSA_on_each_layer_tau.csv"))),
#          p=t(as.matrix(import("Result/RSA_by_layer/RSA_on_each_layer_p.csv"))))
rm(boot.emotion,boot.subject,index.layer,index.feature,RDM.layer)
# visualize characterized neurons
plot.data = expand.grid(feature=colnames(emotional.features),layer=paste("layer",c(1:12)))
plot.data[,"tau"] = as.vector(RSA.layer$tau)
plot.data[plot.data$tau<0,"tau"] = 0
plot.data[as.vector(RSA.layer$q)>=0.05,"signif"] = ""
plot.data[as.vector(RSA.layer$q)<0.05,"signif"] = "*"
plot.data[as.vector(RSA.layer$q)<0.01,"signif"] = "**"
plot.data[as.vector(RSA.layer$q)<0.001,"signif"] = "***"
p = ggplot(plot.data, aes(x=feature,y=layer,fill=tau))+
  geom_tile()+
  geom_text(aes(label=signif), colour="black", stat = "identity",size = 2,vjust=0.65)+
  scale_fill_viridis_c(option = "rocket",begin=1-max(plot.data$tau),end=1,direction = -1,name="Kendall's tau",breaks=c(0,max(plot.data$tau)),labels=c("0",round(max(plot.data$tau),2)))+
  scale_x_discrete(name="",guide=guide_axis(angle = 90),position = "top")+
  scale_y_discrete(name="",limits=rev)+
  theme(panel.background=element_blank(), panel.grid = element_blank(), axis.ticks.length=unit(0,units = "mm"))
ggsave("Result/RSA_by_layer/tau_of_RSA_with_all_emotional_features_on_each_layers.pdf",plot = p, width = 270, height = 135, units = "mm")
rm(plot.data,p)



# RSA (to emotional PCs) by layer ---------
# 
RSA.layer = list(tau=matrix(nrow = ncol(PCs.emotion), ncol = 12, dimnames = list(colnames(PCs.emotion),c(1:12))),
                 p=matrix(nrow = ncol(PCs.emotion), ncol = 12, dimnames = list(colnames(PCs.emotion),c(1:12))))
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
for (index.layer in c(1:12)) {
  RDM.layer = import(paste0("Data/preprocessed/PLM/prompt_tuning_RoBERTa/neurons_activation/by_layer/dist_of_layer_",index.layer,".csv"),header = T)
  for (index.feature in c(1:ncol(PCs.emotion))) {
    set.seed(6)
    compare.result = boot(RDM.layer, boot.subject, R=1000, RDM.candidate=PCs.emotion[,index.feature])
    RSA.layer$tau[index.feature,index.layer] = compare.result$t0
    RSA.layer$p[index.feature,index.layer] = sum(compare.result$t<=0)/1000
    print(paste("index.layer =",index.layer,"index.feature =",index.feature))
    rm(compare.result)
  }
}
# p-value correction by FDR (false discovery rate)
RSA.layer[["q"]] = RSA.layer$p * nrow(RSA.layer$p) * ncol(RSA.layer$p) / rank(RSA.layer$p)
# save RSA.layer results
rownames(RSA.layer$q)=rownames(RSA.layer$p)
colnames(RSA.layer$q)=colnames(RSA.layer$p)
export(t(RSA.layer$tau),"Result/RSA_by_layer/RSA_with_emotional_PCs_on_each_layer_tau.csv")
export(t(RSA.layer$p),"Result/RSA_by_layer/RSA_with_emotional_PCs_on_each_layer_p.csv")
# RSA.layer=list(tau=t(as.matrix(import("Result/RSA_by_layer/RSA_on_each_layer_tau.csv"))),
#          p=t(as.matrix(import("Result/RSA_by_layer/RSA_on_each_layer_p.csv"))))
rm(boot.emotion,boot.subject,index.layer,index.feature,RDM.layer)
# visualize characterized neurons
plot.data = expand.grid(feature=colnames(PCs.emotion),layer=paste("layer",c(1:12)))
plot.data[,"tau"] = as.vector(RSA.layer$tau)
plot.data[plot.data$tau<0,"tau"] = 0
plot.data[as.vector(RSA.layer$q)>=0.05,"signif"] = ""
plot.data[as.vector(RSA.layer$q)<0.05,"signif"] = "*"
plot.data[as.vector(RSA.layer$q)<0.01,"signif"] = "**"
plot.data[as.vector(RSA.layer$q)<0.001,"signif"] = "***"
p = ggplot(plot.data, aes(x=feature,y=layer,fill=tau))+
  geom_tile()+
  geom_text(aes(label=signif), colour="black", stat = "identity",size = 2,vjust=0.65)+
  scale_fill_viridis_c(option = "rocket",begin=1-max(plot.data$tau),end=1,direction = -1,name="Kendall's tau",breaks=c(0,max(plot.data$tau)),labels=c("0",round(max(plot.data$tau),2)))+
  scale_x_discrete(name="",guide=guide_axis(angle = 90),position = "top")+
  scale_y_discrete(name="",limits=rev)+
  theme(panel.background=element_blank(), panel.grid = element_blank(), axis.ticks.length=unit(0,units = "mm"))
ggsave("Result/RSA_by_layer/tau_of_RSA_with_emotional_PCs_on_each_layers.pdf",plot = p, width = 270, height = 135, units = "mm")
rm(plot.data,p)




# RSA (to emotion space) by layer ----
RSA.layer = list(tau=matrix(nrow = ncol(emotion.space), ncol = 12, dimnames = list(colnames(emotion.space),c(1:12))),
                 p=matrix(nrow = ncol(emotion.space), ncol = 12, dimnames = list(colnames(emotion.space),c(1:12))))
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
for (index.layer in c(1:12)) {
  RDM.layer = import(paste0("Data/preprocessed/PLM/prompt_tuning_RoBERTa/neurons_activation/by_layer/dist_of_layer_",index.layer,".csv"),header = T)
  for (index.feature in c(1:ncol(emotion.space))) {
    set.seed(6)
    compare.result = boot(RDM.layer, boot.subject, R=100, RDM.candidate=emotion.space[,index.feature])
    RSA.layer$tau[index.feature,index.layer] = compare.result$t0
    RSA.layer$p[index.feature,index.layer] = sum(compare.result$t<=0)/100
    print(paste("index.layer =",index.layer,"index.feature =",index.feature))
    rm(compare.result)
  }
}
# p-value correction by FDR (false discovery rate)
RSA.layer[["q"]] = RSA.layer$p * nrow(RSA.layer$p) * ncol(RSA.layer$p) / rank(RSA.layer$p)
# save RSA.layer results
rownames(RSA.layer$q)=rownames(RSA.layer$p)
colnames(RSA.layer$q)=colnames(RSA.layer$p)
export(t(RSA.layer$tau),"Result/RSA_by_layer/RSA_with_emotion_space_on_each_layer_tau.csv")
export(t(RSA.layer$p),"Result/RSA_by_layer/RSA_with_emotion_space_on_each_layer_p.csv")
# RSA.layer=list(tau=t(as.matrix(import("Result/RSA_by_layer/RSA_on_each_layer_tau.csv"))),
#          p=t(as.matrix(import("Result/RSA_by_layer/RSA_on_each_layer_p.csv"))))
rm(boot.emotion,boot.subject,index.layer,index.feature,RDM.layer)
# visualize characterized neurons
plot.data = expand.grid(feature=colnames(emotion.space),layer=paste("layer",c(1:12)))
plot.data[,"tau"] = as.vector(RSA.layer$tau)
plot.data[plot.data$tau<0,"tau"] = 0
plot.data[as.vector(RSA.layer$q)>=0.05,"signif"] = ""
plot.data[as.vector(RSA.layer$q)<0.05,"signif"] = "*"
plot.data[as.vector(RSA.layer$q)<0.01,"signif"] = "**"
plot.data[as.vector(RSA.layer$q)<0.001,"signif"] = "***"
p = ggplot(plot.data, aes(x=feature,y=layer,fill=tau))+
  geom_tile()+
  geom_text(aes(label=signif), colour="black", stat = "identity",size = 2,vjust=0.65)+
  scale_fill_viridis_c(option = "rocket",begin=1-max(plot.data$tau),end=1,direction = -1,name="Kendall's tau",breaks=c(0,max(plot.data$tau)),labels=c("0",round(max(plot.data$tau),2)))+
  scale_x_discrete(name="",guide=guide_axis(angle = 90),position = "top")+
  scale_y_discrete(name="",limits=rev)+
  theme(panel.background=element_blank(), panel.grid = element_blank(), axis.ticks.length=unit(0,units = "mm"))
ggsave("Result/RSA_by_layer/tau_of_RSA_with_emotion_space_on_each_layers.pdf",plot = p, width = 270, height = 135, units = "mm")
rm(plot.data,p)

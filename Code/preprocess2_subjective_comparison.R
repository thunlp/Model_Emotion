library(bruceR)
library(psych)
set.wd()
set.wd("../")

## load data
subjective.comparison = import("Data/raw_data/subjective_comparison/data.xlsx")
index.dist = unlist(import("Data/preprocessed/reorder_distance.csv",header = F))

## individual dist
score.ind = t(subjective.comparison[,5:ncol(subjective.comparison)])
index = matrix(NA,nrow = 28,ncol = 28)
index[lower.tri(index)] = c(1:378)
index = index[1:27,1:27]
index = unique(matrix(index,ncol = 1))[2:352]
score.ind = score.ind[,index]
for (pair in 1:ncol(score.ind)) {
  if (any(is.na(score.ind[,pair]))){
    score.ind[which(is.na(score.ind[,pair])),pair] = mean(score.ind[,pair], na.rm=T)
  }
}
rm(pair, index)
score.ind = score.ind[,index.dist]
export(score.ind, "Data/preprocessed/subjective_comparison/ind_dist_subjective_comparison.csv")

## averaged dist
score.avg = data.frame(colMeans(score.ind))
colnames(score.avg) = "Subjective Comparison"
export(score.avg, "Data/preprocessed/subjective_comparison/avg_dist_subjective_comparison.csv")

## visually check individual RDM
library(pheatmap)
library(MASS)
index.emotion = unlist(import("Data/preprocessed/reordered_emotions.csv",header = F))
for (index.sub in 1:nrow(score.ind)) {
  temp.RDM = matrix(0,nrow = 27,ncol = 27,dimnames = list(index.emotion,index.emotion))
  temp.RDM[lower.tri(temp.RDM)] = score.ind[index.sub,]
  temp.RDM = temp.RDM + t(temp.RDM)
  pheatmap(temp.RDM,filename=paste0("Data/raw_data/subjective_comparison/sub",index.sub,"_heatmap.png"),
           cluster_rows = F, cluster_cols = F)
  # fit = sammon(temp.RDM, k=2)
  # stress = fit$stress
  # plotdata = data.frame(x=fit$points[,1],y=fit$points[,2],label=rownames(fit$points))
  # l = 1.1*max(max(plotdata$x)-min(plotdata$x),max(plotdata$y)-min(plotdata$y))/2
  # x1 = mean(c(min(plotdata$x),max(plotdata$x)))-l
  # x2 = mean(c(min(plotdata$x),max(plotdata$x)))+l
  # y1 = mean(c(min(plotdata$y),max(plotdata$y)))-l
  # y2 = mean(c(min(plotdata$y),max(plotdata$y)))+l
  # p = ggplot(plotdata,aes(x=x,y=y,label=label),)+
  #   geom_text()+
  #   scale_x_continuous(name = "",breaks = NULL)+
  #   scale_y_continuous(name = "",breaks = NULL)+
  #   coord_equal(xlim = c(x1,x2),ylim = c(y1,y2))+
  #   labs(subtitle = paste0("stress = ",stress))+
  #   theme_bw() + 
  #   theme(panel.background=element_blank(), panel.grid = element_blank())
  # ggsave(paste0("Data/raw_data/subjective_comparison/sub",index.sub,"_MDS.png"),plot = p)
}
rm(index.sub,temp.RDM,fit,stress,plotdata,l,x1,x2,y1,y2,p)

# check the intersubject consistency
corr.ind = cor(t(score.ind),method = "spearman")
pheatmap(corr.ind,filename=paste0("Data/raw_data/subjective_comparison/sub_corr.png"))
iccdata = ICC(t(score.ind), lmer = F)

# visualize the averaged RDM
RDM.matrix = matrix(0, nrow = 27, ncol = 27, dimnames = list(index.emotion,index.emotion))
RDM.matrix[lower.tri(RDM.matrix)] = unlist(score.avg)
RDM.matrix = RDM.matrix + t(RDM.matrix)
fit = sammon(RDM.matrix, k=2)
stress = fit$stress
plotdata = data.frame(x=fit$points[,1],y=fit$points[,2],label=rownames(fit$points))
l = 1.1*max(max(plotdata$x)-min(plotdata$x),max(plotdata$y)-min(plotdata$y))/2
x1 = mean(c(min(plotdata$x),max(plotdata$x)))-l
x2 = mean(c(min(plotdata$x),max(plotdata$x)))+l
y1 = mean(c(min(plotdata$y),max(plotdata$y)))-l
y2 = mean(c(min(plotdata$y),max(plotdata$y)))+l
p = ggplot(plotdata,aes(x=x,y=y,label=label),)+
  geom_text()+
  scale_x_continuous(name = "",breaks = NULL)+
  scale_y_continuous(name = "",breaks = NULL)+
  coord_equal(xlim = c(x1,x2),ylim = c(y1,y2))+
  labs(subtitle = paste0("stress = ",stress))+
  theme_bw() + 
  theme(panel.background=element_blank(), panel.grid = element_blank())
ggsave("Data/raw_data/subjective_comparison/averaged_MDS.png",plot = p)



# pheatmap(RDM.matrix,filename=paste0("Data/raw_data/subjective_comparison/sub",index.sub,"_heatmap.png"),
#          cluster_rows = F, cluster_cols = F)
# plot.data = expand.grid(x=index.emotion,y=index.emotion)
# plot.data[,"dist"] = as.vector(RDM.matrix)
# p.RDM = ggplot(plot.data, aes(x=x,y=y,fill=dist))+
#   geom_tile()+
#   scale_fill_viridis_c(option = "magma",name="Percentile\nof distance",breaks=c(min(plot.data$dist),max(plot.data$dist)),labels=c("Min","Max"))+
#   scale_x_discrete(name="",guide=guide_axis(angle = 90))+
#   scale_y_discrete(name="")+
#   coord_fixed(ratio = 1) + 
#   theme(panel.background=element_blank(), panel.grid = element_blank(), axis.ticks.length=unit(0,units = "mm"))
# ggsave("Result/RSA_whole_model/RDM_of_Subjective_Comparison.pdf",plot = p.RDM)
# 

## Load data
f = "stat/"
load(paste(f,"RSA.RData",sep = ""))
# set color
color.states = c("positive"="#e41a1c",
                 "ambiguous"="#ff7f00",
                 "negative"="#377eb8")
color.model = c("Human Behavior"="#A95454",  # H=360, S=50, L=50
                "AI Behavior"="#5454A9",  #  H=240, S=50, L=50
                #"Emotional Feature"="#54A954"  #  H=120, S=50, L=50
                "Affective Features"="#6BDA1B", # H=95, S=88, L=48
                "Physical Features"="#229628",  #  H=123, S=77, L=36
                "Appraisal Features"="#1F5B31" #  H=138, S=66, L=24
)
library(corrplot)
library(viridis)
library(RColorBrewer)
color.RDM = magma(351)
color.corr = rev(colorRampPalette(brewer.pal(11,"RdBu"))(351))
color.RSA = rocket(351,begin=0.2,end=1,direction = -1)
# set random seed
set.seed(6)



## Step 1
# RDM visualization -- heatmap with histogram equalization
library(pheatmap)
library(EBImage)
a = matrix(0, nrow = 27, ncol = 27)
a[lower.tri(a)] = dist.feature[,"valence"]
a = a + t(a)
hc = hclust(as.dist(a), method = 'ward.D2')
for (i in c(1:length(type.feature))){
  b = equalize(RDM.feature[[i]], range = c(0, max(RDM.feature[[i]])), levels = 351)
  rownames(b) = emotions$class
  colnames(b) = emotions$class
  bk = seq(min(b, na.rm = TRUE), max(b, na.rm = TRUE), length.out = 351)
  if (i<=5) {
    pheatmap(b, breaks = bk, color = color.RDM, legend = F, annotation_legend = F,
             cluster_rows = hc, cluster_cols = hc, treeheight_row = 0, treeheight_col = 0,
             angle_col = "90", border_color = NA, show_rownames = T, show_colnames = T,
             filename = paste(f, "heatmap_feature_", i, ".tiff", sep = ""), width = 4.5,height = 4.5)
  }else{
    pheatmap(b, breaks = bk, color = color.RDM, legend = F, annotation_legend = F,
             cluster_rows = hc, cluster_cols = hc, treeheight_row = 0, treeheight_col = 0,
             angle_col = "90", border_color = NA, show_rownames = F, show_colnames = F,
             filename = paste(f, "heatmap_feature_", i, ".tiff", sep = ""))
  }
}
rm(i, hc, bk, a, b)
# visualization -- t-SNE
library(ggplot2)
library(ggrepel)
library(Rtsne)
Emotion.Feature = read.csv("stat/score_EmotionalFeatures.csv",header = T)
colnames(Emotion.Feature) = unlist(read.csv("stat/features.csv"))[6:length(unlist(read.csv("stat/features.csv")))]
for (n in c(1:5)){
  set.seed(6)
  m = Rtsne(as.dist(RDM.feature[[n]]), perplexity = 8, theta = 0.0)
  m = data.frame(m$Y)
  colnames(m) = c("Dimension1", "Dimension2")
  m[,'label'] = emotions$class[1:27]
  l = 1.1*max(max(m$Dimension1)-min(m$Dimension1),max(m$Dimension2)-min(m$Dimension2))/2
  x1 = mean(c(min(m$Dimension1),max(m$Dimension1)))-l
  x2 = mean(c(min(m$Dimension1),max(m$Dimension1)))+l
  y1 = mean(c(min(m$Dimension2),max(m$Dimension2)))-l
  y2 = mean(c(min(m$Dimension2),max(m$Dimension2)))+l
  p = ggplot(m, aes(Dimension1, Dimension2, label = label)) +
    geom_point(size=3) +
    geom_text_repel(size=6) +
    guides(colour = "none") +
    coord_fixed(ratio = 1, expand = FALSE) + 
    scale_x_continuous(limits = c(x1,x2),breaks = NULL,name = "") + 
    scale_y_continuous(limits = c(y1,y2),breaks = NULL,name = "") +
    # scale_colour_brewer(type = "div",palette = "RdYlBu") +
    theme_bw()+
    theme(panel.background = element_rect(fill = 'white', colour = 'white'))
  ggsave(paste("tSNE_feature_", type.feature[n], ".tiff", sep = ""),plot = p, path = f, height = 7, width = 7)
}
rm(n,m,l,x1,x2,y1,y2,p)



## Step 2
# Comparing RDMs [heatmap]
corr.feature = cor(dist.feature, method = "spearman")
b = equalize(corr.feature, range = c(-1, 1), levels = ncol(corr.feature)^2)
colnames(b)=colnames(corr.feature)
rownames(b)=rownames(corr.feature)
tiff(file = paste(f,"corr_feature.tiff",sep = ""), height = 8, width = 12, units = "in", res = 300)
corrplot(corr.feature, method = "color", tl.col = "black", tl.srt = 90, cl.pos = "n",
         col = color.corr, order = 'original')
dev.off()
rm(b)
# t-SNE
set.seed(6)
m = Rtsne(as.dist(1-corr.feature), perplexity = 12, theta = 0.0)
m = data.frame(m$Y)
colnames(m) = c("Dimension1", "Dimension2")
m[, 'label'] = type.feature
m[, 'source'] = type.model
l = 1.1 * max(max(m$Dimension1) - min(m$Dimension1),
              max(m$Dimension2) - min(m$Dimension2)) / 2
x1 = mean(c(min(m$Dimension1), max(m$Dimension1))) - l
x2 = mean(c(min(m$Dimension1), max(m$Dimension1))) + l
y1 = mean(c(min(m$Dimension2), max(m$Dimension2))) - l
y2 = mean(c(min(m$Dimension2), max(m$Dimension2))) + l
p = ggplot(m, aes(Dimension1, Dimension2, colour = source, label = label)) +
  geom_point(size=4) +
  geom_text_repel() +
  coord_fixed(ratio = 1, expand = FALSE) + 
  scale_x_continuous(limits = c(x1, x2), breaks = NULL, name = NULL) + 
  scale_y_continuous(limits = c(y1, y2), breaks = NULL, name = NULL) +
  scale_colour_manual(values = color.model)+
  guides(colour = "none") +
  theme_bw()+
  theme(panel.background = element_rect(fill = 'white', colour = 'white'))
ggsave("tSNE_feature.tiff",plot = p, path = f, height = 7, width = 7)
rm(m, p, l, x1, x2, y1, y2)



## Step 3
# visualization -- bar plot
for (m in c(1:5)) {
  plotdata = RSA[[m]]
  plotdata$r = 1-plotdata$r
  x.positive = 1 + round(max(plotdata$r), digits = 1)/0.1
  plotcolor = color.model
  p = ggplot(plotdata, aes(y=reorder(e,-r),x=r,fill=c))+
    geom_vline(xintercept = 1, color="gray")+
    geom_bar(stat="identity")+
    geom_errorbar(aes(xmin=r-se/2, xmax=r+se/2, width = 0.3))+
    geom_text(mapping = aes(x=x.positive/400,label=q), colour="white", stat = "identity", hjust=0, vjust=0.8)+
    scale_x_continuous(expand = c(0,0), n.breaks = x.positive+1, minor_breaks = NULL)+
    scale_fill_manual(values=plotcolor)+
    theme_classic() + theme(legend.position = "none")
  if (m==1) {
    p = p+labs(x = "RDM deviation (1 - spearman r)\nbootstrap single RDM & stimuli 1000 times, FDR = 0.05",y=NULL ,fill = NULL)
    # geom_vline(xintercept = 1-noise.ceiling[1], color="black",linetype = "dashed")+
    # geom_vline(xintercept = 1-noise.ceiling[2], color="black",linetype = "dashed")+
    # geom_text(aes(x=1-noise.ceiling[1]+0.12,y=e[which.min(r)],label="noise ceiling"), stat = "identity")
  } else {
    p = p+labs(x = "RDM deviation (1 - spearman r)\nbootstrap stimuli 1000 times, FDR = 0.05",y=NULL ,fill = NULL)
  }
  ggsave(paste(f,"RSA_feature_",type.feature[m],".tiff",sep = ""), plot = p, width = 6, height = 8)
}
rm(m,p,plotdata,x.positive,plotcolor)
# visualization -- heatmap
RSA.combine = matrix(NA,nrow = length(type.feature)-5,ncol = 5,dimnames = list(type.feature[6:length(type.feature)],type.feature[1:5]))
q.mat = NULL
for (m in 1:5) {
  q = (RSA[[m]]$p * nrow(RSA[[m]]) / rank(RSA[[m]]$p))
  if (m==1|m==2) {
    a = RSA[[m]][4:nrow(RSA[[m]]),]
    q.mat = cbind(q.mat,q[4:nrow(RSA[[m]])])
  }else{
    a = RSA[[m]][3:nrow(RSA[[m]]),]
    q.mat = cbind(q.mat,q[3:nrow(RSA[[m]])])
  }
  RSA.combine[,m] = a$r
}
colnames(q.mat) = colnames(RSA.combine)
rownames(q.mat) = rownames(RSA.combine)
RSA.combine[RSA.combine<0]=0
tiff(file = paste(f,"RSA_combine.tiff",sep = ""), height = 10, width = 30, units = "in", res = 300)
corrplot(t(RSA.combine), method = "color", is.corr = F,
         tl.col = "black", tl.srt = 90, tl.cex = 3,
         col = color.RSA, col.lim = c(0,0.8),
         cl.align.text = "l", cl.length = 2, cl.cex = 3,
         p.mat = t(q.mat), sig.level = c(0.001,0.01,0.05), insig = "label_sig",
         order = 'original', na.label = "square", na.label.col = "black")
dev.off()
rm(a,m,q)
# visualization -- t-SNE with significant feature
library(gstat)
Emotion.Feature = read.csv("stat/score_EmotionalFeatures.csv",header = T)
colnames(Emotion.Feature) = unlist(read.csv("stat/features.csv"))[6:length(unlist(read.csv("stat/features.csv")))]
for (n in c(1:5)){
  kk = order(q.mat[,n],decreasing = F)
  kn = sum(q.mat[,n]<0.05)
  kk = c(kk[1],kk[ceiling(kn/2)],kk[kn])
  for (k in kk) {
    set.seed(6)
    m = Rtsne(as.dist(RDM.feature[[n]]), perplexity = 8, theta = 0.0)
    m = data.frame(m$Y)
    colnames(m) = c("Dimension1", "Dimension2")
    m[,'label'] = emotions$class[1:27]
    m[,'score'] = Emotion.Feature[,k]
    l = 1.1*max(max(m$Dimension1)-min(m$Dimension1),max(m$Dimension2)-min(m$Dimension2))/2
    x1 = mean(c(min(m$Dimension1),max(m$Dimension1)))-l
    x2 = mean(c(min(m$Dimension1),max(m$Dimension1)))+l
    y1 = mean(c(min(m$Dimension2),max(m$Dimension2)))-l
    y2 = mean(c(min(m$Dimension2),max(m$Dimension2)))+l
    b = expand.grid(x=seq(from=x1,to=x2,length.out=100),
                    y=seq(from=y1,to=y2,length.out=100))
    sp::coordinates(b) = ~x+y
    sp::gridded(b) = TRUE
    b = idw(formula = score ~ 1,
            locations = ~Dimension1+Dimension2,
            idp = 4,
            data = m, newdata = b)
    b = as.data.frame(b)[,1:3]
    colnames(b) = c("x","y","score")
    p = ggplot() +
      geom_contour_filled(data=b, aes(x=x,y=y,z=score)) +
      geom_point(data=m, aes(x=Dimension1, y=Dimension2), size = 5, alpha = 0.8, colour = "white")+
      coord_fixed(ratio = 1, expand = FALSE) + 
      scale_x_continuous(limits = c(x1,x2),breaks = NULL,name = "") + 
      scale_y_continuous(limits = c(y1,y2),breaks = NULL,name = "") +
      scale_fill_viridis_d(direction = 1)+
      guides(fill = "none")+
      theme_bw()+
      theme(panel.background = element_rect(fill = 'white', colour = 'white'))
    ggsave(paste("Color_", type.feature[n], "_", type.feature[k+5], ".tiff", sep = ""),plot = p, path = f, height = 7, width = 7)
  }
}
rm(m, p, l, n, x1, x2, y1, y2, k, kn, kk, b)


# for (n in c(1:5)){
#   kk = order(RSA.combine[,n],decreasing = T)
#   kn = sum(q.mat[,n]<0.05)
#   kk = c(kk[1],kk[ceiling(kn/2)],kk[kn])
#   for (k in kk) {
#     set.seed(6)
#     m = Rtsne(as.dist(RDM.feature[[n]]), perplexity = 8, theta = 0.0)
#     m = data.frame(m$Y)
#     colnames(m) = c("Dimension1", "Dimension2")
#     m[,'label'] = emotions$class[1:27]
#     m[,'score'] = Emotion.Feature[,k]
#     l = 1.1*max(max(m$Dimension1)-min(m$Dimension1),max(m$Dimension2)-min(m$Dimension2))/2
#     x1 = mean(c(min(m$Dimension1),max(m$Dimension1)))-l
#     x2 = mean(c(min(m$Dimension1),max(m$Dimension1)))+l
#     y1 = mean(c(min(m$Dimension2),max(m$Dimension2)))-l
#     y2 = mean(c(min(m$Dimension2),max(m$Dimension2)))+l
#     p = ggplot(m, aes(Dimension1, Dimension2, fill = score, size = score)) +
#       geom_point(alpha = 0.8, shape = 21, colour = "black") +
#       guides(fill = "none", size = "none") +
#       coord_fixed(ratio = 1, expand = FALSE) + 
#       scale_size(range = c(5,35)) +
#       scale_x_continuous(limits = c(x1,x2),breaks = NULL,name = "") + 
#       scale_y_continuous(limits = c(y1,y2),breaks = NULL,name = "") +
#       scale_fill_viridis(option = "D",direction = 1) +
#       theme_bw()+
#       theme(panel.background = element_rect(fill = 'white', colour = 'white'))
#     ggsave(paste("Color_", type.feature[n], "_", type.feature[k+5], ".tiff", sep = ""),plot = p, path = f, height = 7, width = 7)
#   }
# }

# visualization
# library(VennDiagram)
# signif.feature = list("subjective comparison"=NULL,"corpus annotation"=NULL,"word embedding"=NULL,"PLM task prompt"=NULL,"PLM activated neurons"=NULL)
# for (m in 1:5) {
#   signif.feature[[m]] = RSA[[m]]$e[RSA[[m]]$q=="*"]
#   if (m<3) {
#     signif.feature[[m]] = signif.feature[[m]][4:length(signif.feature[[m]])]
#   }else{
#     signif.feature[[m]] = signif.feature[[m]][3:length(signif.feature[[m]])]
#   }
# }
# venn.diagram(signif.feature,filename = "venn_features.tiff",disable.logging = T)

# ## Extract subset of positive and negative emotions
# #
# indexE <- function(ord){
#   combn27 = combn(1:27,2)
#   index1 = logical(length = length(combn27[1,]))
#   index2 = logical(length = length(combn27[2,]))
#   for (i in 1:length(ord)){
#     index1 = index1|(combn27[1,]==ord[i])
#     index2 = index2|(combn27[2,]==ord[i])
#   }
#   index = index1&index2
#   return(index)
# }
# dist.positive = dist.feature[indexE(c(1,2,5,6,9,14,16,18,19,21,22,24)),]
# dist.negative = dist.feature[indexE(c(3,4,10,11,12,13,15,17,20,25,26)),]
# write.csv(dist.positive, file = paste(f,"AllFeatures/dist_positive.csv",sep = ""), row.names = FALSE)
# write.csv(dist.negative, file = paste(f,"AllFeatures/dist_negative.csv",sep = ""), row.names = FALSE)

library(Rtsne)
response = import("Result/search_light_analysis/linear_regression_to_activation_after_nonnegative/all_beta.csv",as="data.table")
response[["label"]] = factor(rep(paste("label",1:12),each=3072),levels = paste("label",1:12))
response = response[!is.na(response$arousal),]
idx.affective=duplicated(response[,1:2])
idx.physical=duplicated(response[,3:8])
idx.appraisal=duplicated(response[,9:14])

tsne.affective=Rtsne(scale(response[!idx.affective,1:2]))
tsne.affective=data.frame(tsne.affective$Y,response$label[!idx.affective])
colnames(tsne.affective)[3]="label"
p.affective=ggplot(tsne.affective, aes(X1,X2,group=label,color=label,fill=label))+
  geom_point()+
  scale_fill_discrete()+
  facet_wrap(~label)
p.affective

tsne.physical=Rtsne(scale(response[!idx.physical,3:8]))
tsne.physical=data.frame(tsne.physical$Y,response$label[!idx.physical])
colnames(tsne.physical)[3]="label"
p.physical=ggplot(tsne.physical, aes(X1,X2,group=label,color=label,fill=label))+
  geom_point()+
  scale_fill_discrete()+
  facet_wrap(~label)
p.physical

tsne.appraisal=Rtsne(scale(response[!idx.appraisal,9:14]))
tsne.appraisal=data.frame(tsne.appraisal$Y,response$label[!idx.appraisal])
colnames(tsne.appraisal)[3]="label"
p.appraisal=ggplot(tsne.appraisal, aes(X1,X2,group=label,color=label,fill=label))+
  geom_point()+
  scale_fill_discrete()+
  facet_wrap(~label)
p.appraisal

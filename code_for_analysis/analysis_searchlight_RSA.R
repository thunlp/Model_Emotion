library(bruceR)
library(boot)
library(ggridges)
library(pheatmap)
library(viridis)
library(VennDiagram)
library(RColorBrewer)
set.wd()
set.wd("../")

# load data ----
dist.emotional.PCs = import("Data/preprocessed/emotional_features/dist_emotional_PCs.csv")
dist.emotion.space = import("Data/preprocessed/emotional_features/dist_emotion_space.csv")[,1:3]
dist.neuron.activation = import("Data/preprocessed/PLM/prompt_tuning_RoBERTa/neurons_activation/by_neuron/avg_dist_of_activation.csv",header = T,as = "data.table")
rank.feature = order(-(import("Result/RSA_whole_model/RSA_with_emotional_PCs_tau.csv")$`Subjective Comparison`))

# compute kendall's tau ----
RSA.by.emotional.PCs = list(tau=as.data.table(matrix(nrow = 36864, ncol = 14)),
                            p.value=data.table(matrix(nrow = 36864, ncol = 14)))
colnames(RSA.by.emotional.PCs$tau) = colnames(dist.emotional.PCs)
colnames(RSA.by.emotional.PCs$p.value) = colnames(dist.emotional.PCs)
for (index.feature in c(1:ncol(dist.emotional.PCs))) {
  a = sapply(dist.neuron.activation,
             function(x,y) {
               a = cor.test(x,y,method="kendall",alternative="greater")
               return(c(a$estimate,a$p.value))},
             y=dist.emotional.PCs[,index.feature])
  RSA.by.emotional.PCs$tau[[index.feature]] = a[1,]
  RSA.by.emotional.PCs$p.value[[index.feature]] = a[2,]
  print(index.feature)
}
RSA.by.emotion.space = list(tau=data.table(matrix(nrow = 36864, ncol = 3)),
                            p.value=data.table(matrix(nrow = 36864, ncol = 3)))
colnames(RSA.by.emotion.space$tau) = colnames(dist.emotion.space)
colnames(RSA.by.emotion.space$p.value) = colnames(dist.emotion.space)
for (index.feature in c(1:ncol(dist.emotion.space))) {
  a = sapply(dist.neuron.activation,
             function(x,y) {
               a = cor.test(x,y,method="kendall",alternative="greater")
               return(c(a$estimate,a$p.value))},
             y=dist.emotion.space[,index.feature])
  RSA.by.emotion.space$tau[[index.feature]] = a[1,]
  RSA.by.emotion.space$p.value[[index.feature]] = a[2,]
  print(index.feature)
}

# correct statistic value (by FDR) ----
RSA.by.emotional.PCs[["corrected.p.value"]] = RSA.by.emotional.PCs$p.value * sum(!is.na(RSA.by.emotional.PCs$p.value[,1])) * ncol(RSA.by.emotional.PCs$p.value) / rank(RSA.by.emotional.PCs$p.value)
RSA.by.emotional.PCs[["significance"]] = RSA.by.emotional.PCs$corrected.p.value<0.01
RSA.by.emotional.PCs[["signif.tau"]] = RSA.by.emotional.PCs$tau
RSA.by.emotional.PCs$signif.tau[!RSA.by.emotional.PCs$significance] = NA
RSA.by.emotion.space[["corrected.p.value"]] = RSA.by.emotion.space$p.value * sum(!is.na(RSA.by.emotion.space$p.value[,1])) * ncol(RSA.by.emotion.space$p.value) / rank(RSA.by.emotion.space$p.value)
RSA.by.emotion.space[["significance"]] = RSA.by.emotion.space$corrected.p.value<0.01
RSA.by.emotion.space[["signif.tau"]] = RSA.by.emotion.space$tau
RSA.by.emotion.space$signif.tau[!RSA.by.emotion.space$significance] = NA

# rank neuron by statistic value ----
lapply(-RSA.by.emotional.PCs$tau, rank, na.last = "keep") %>%
  as.data.frame -> rank.by.emotional.PCs
lapply(-RSA.by.emotion.space$tau, rank, na.last = "keep") %>%
  as.data.frame -> rank.by.emotion.space
colnames(rank.by.emotional.PCs) = colnames(RSA.by.emotional.PCs$tau)
colnames(rank.by.emotion.space) = colnames(RSA.by.emotion.space$tau)

# save results ----
export(RSA.by.emotional.PCs$tau,"Result/search_light_analysis/by_emotional_PCs/searchlight_RSA_to_avg_activation/tau.csv")
export(RSA.by.emotional.PCs$p.value,"Result/search_light_analysis/by_emotional_PCs/searchlight_RSA_to_avg_activation/p_value.csv")
export(rank.by.emotional.PCs,"Result/search_light_analysis/by_emotional_PCs/searchlight_RSA_to_avg_activation/neurons_rank.csv")
export(RSA.by.emotion.space$tau,"Result/search_light_analysis/by_emotion_space/searchlight_RSA_to_avg_activation/tau.csv")
export(RSA.by.emotion.space$p.value,"Result/search_light_analysis/by_emotion_space/searchlight_RSA_to_avg_activation/p_value.csv")
export(rank.by.emotion.space,"Result/search_light_analysis/by_emotion_space/searchlight_RSA_to_avg_activation/neurons_rank.csv")



# visualize neurons' distribution ----
# for 14 property
plotdata = data.table(layer=rep(factor(rep(paste0("layer",1:12),
                                           each=3072),
                                       levels = c(paste0("layer",1:12))),
                                times=14),
                      attribute=rep(factor(colnames(RSA.by.emotional.PCs$tau),
                                       levels = colnames(rank.by.emotional.PCs)[rank.feature]),
                                each=36864),
                      rank=unlist(rank.by.emotional.PCs))
plotdata = plotdata[rank<4000,] %>%
  group_by(layer,attribute) %>%
  summarise_at("rank",
               list(count=~length(.))) %>%
  as.data.table()
p.distribution = ggplot(plotdata, aes(y=attribute,x=count,fill=count))+
  geom_col(width = 1,position = position_stack(reverse = TRUE))+
  scale_y_discrete(limits=rev,name="Attributes",expand = c(0,0))+
  scale_x_continuous(expand = c(0,0), n.breaks = 3, guide = guide_axis(angle=90))+
  scale_fill_viridis_c(name="Count",option = "B")+
  facet_grid(~layer)+
  theme_bw()
p.distribution
ggsave("Result/search_light_analysis/by_emotional_PCs/searchlight_RSA_to_avg_activation/distribution.pdf",
       plot = p.distribution, width = 240, height = 135, units = "mm")
# for 3 space
plotdata = data.table(layer=rep(factor(rep(paste0("layer",1:12),
                                           each=3072),
                                       levels = c(paste0("layer",1:12))),
                                times=3),
                      space=rep(factor(colnames(RSA.by.emotion.space$tau),
                                       levels = colnames(RSA.by.emotion.space$tau)),
                                each=36864),
                      rank=unlist(rank.by.emotion.space))
plotdata = plotdata[rank<4000,] %>%
  group_by(layer,space) %>%
  summarise_at("rank",
               list(count=~length(.))) %>%
  as.data.table()
p.distribution = ggplot(plotdata, aes(y=space, x=count,fill=count))+
  geom_col(width = 1,position = position_stack(reverse = TRUE))+
  scale_y_discrete(limits=rev,name="",expand = c(0,0))+
  scale_x_continuous(expand = c(0,0), n.breaks = 3, guide = guide_axis(angle=90))+
  scale_fill_viridis_c(name="Count",option = "B")+
  facet_grid(~layer)+
  theme_bw()
p.distribution
ggsave("Result/search_light_analysis/by_emotion_space/searchlight_RSA_to_avg_activation/distribution.pdf",
       plot = p.distribution, width = 240, height = 135, units = "mm")



# visualize neurons' overlapping ----
# signif tau by emotional PCs 
plotdata = RSA.by.emotional.PCs$signif.tau
plotdata = plotdata[!is.na(rowMeans(plotdata,na.rm = T)),]
plotdata = setcolorder(plotdata,rank.feature)
plotdata = t(setorder(plotdata))
p = pheatmap(plotdata, border_color = NA,
             color = inferno(n=ncol(plotdata)),
             cluster_rows = F, cluster_cols = F,
             show_colnames = F,
             width = 8, height = 4.5,
             filename = "Result/search_light_analysis/by_emotional_PCs/searchlight_RSA_to_avg_activation/signif_tau.pdf")
# signif tau by emotion space 
plotdata = RSA.by.emotion.space$signif.tau
plotdata = plotdata[!is.na(rowMeans(plotdata,na.rm = T)),]
plotdata = t(setorder(plotdata))
p = pheatmap(plotdata, border_color = NA,
             color = inferno(n=ncol(plotdata)),
             cluster_rows = F, cluster_cols = F,
             show_colnames = F,
             width = 6, height = 4.5,
             filename = "Result/search_light_analysis/by_emotion_space/searchlight_RSA_to_avg_activation/signif_tau.pdf")
rm(p,plotdata)
# neuron rank by emotional PCs 
index.rank = 4000
plotdata = rank.by.emotional.PCs
plotdata = plotdata[(rowSums(plotdata<=index.rank)!=0),]
plotdata = setcolorder(plotdata,rank.feature)
plotdata[plotdata>index.rank] = NA
plotdata = t(setorder(plotdata))
p = pheatmap(plotdata, border_color = NA,
             color = rev(inferno(n=ncol(plotdata))),
             cluster_rows = F, cluster_cols = F,
             show_colnames = F,
             width = 8, height = 4.5,
             filename = paste0("Result/search_light_analysis/by_emotional_PCs/searchlight_RSA_to_avg_activation/neuron_rank_top",index.rank,".pdf"))
# neuron rank by emotion space 
index.rank = 4000
plotdata = rank.by.emotion.space
plotdata = plotdata[(rowSums(plotdata<=index.rank)!=0),]
plotdata[plotdata>index.rank] = NA
plotdata = t(setorder(plotdata))
p = pheatmap(plotdata, border_color = NA,
             color = rev(inferno(n=ncol(plotdata))),
             cluster_rows = F, cluster_cols = F,
             show_colnames = F,
             width = 6, height = 4.5,
             filename = paste0("Result/search_light_analysis/by_emotion_space/searchlight_RSA_to_avg_activation/neuron_rank_top",index.rank,".pdf"))
rm(p,plotdata)
# overlap of three spaces 
index.rank = 4000
plotdata = rank.by.emotion.space
plotdata = plotdata[(rowSums(plotdata<=index.rank)!=0),]
plotdata[plotdata>index.rank] = NA
myCol <- brewer.pal(3, "Pastel2")
venn.diagram(
  x = list(which(!is.na(plotdata[,1])),
           which(!is.na(plotdata[,2])),
           which(!is.na(plotdata[,3]))),
  category.names = c("Affective\nSpace" , "Basic Emotions\nSpace" , "Appraisal\nSpace"),
  filename = 'Result/search_light_analysis/by_emotion_space/searchlight_RSA_to_avg_activation/overlap_of_signif_neurons.png',
  output=F,
  
  # Output features
  imagetype="png" ,
  height = 480 , 
  width = 480 , 
  resolution = 300,
  compression = "lzw",
  
  # Circles
  lwd = 2,
  lty = 'blank',
  fill = myCol,
  euler = T,
  
  # Numbers
  cex = 0.4,
  fontface = "bold",
  fontfamily = "sans",
  
  # Set names
  cat.cex = 0.4,
  cat.fontface = "bold",
  cat.default.pos = "outer",
  cat.pos = c(-27, 27, 135),
  cat.dist = c(0.055, 0.055, 0.085),
  cat.fontfamily = "sans",
  rotation = 1
)








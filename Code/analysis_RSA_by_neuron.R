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
model = "RoBERTa"
dist.emotional.PCs = import("Data/preprocessed/emotional_features/dist_emotional_PCs.csv")
dist.neuron.activation = import(paste0("Data/preprocessed/PLM/prompt_tuning_",model,"/neurons_activation/by_neuron/avg_dist_of_activation.csv"),header = T,as = "data.table")

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

# rank neuron by statistic value ----
lapply(-RSA.by.emotional.PCs$tau, rank, na.last = "keep") %>%
  as.data.frame -> rank.by.emotional.PCs
colnames(rank.by.emotional.PCs) = colnames(RSA.by.emotional.PCs$tau)

# save/load results ----
export(RSA.by.emotional.PCs$tau,paste0("Result/search_light_analysis/",model,"_tau.csv"))
export(RSA.by.emotional.PCs$p.value,paste0("Result/search_light_analysis/",model,"_p_value.csv"))
export(rank.by.emotional.PCs,paste0("Result/search_light_analysis/",model,"_neurons_rank.csv"))
# RSA.by.emotional.PCs = list(tau=as.data.table(matrix(nrow = 36864, ncol = 14)),
#                             p.value=data.table(matrix(nrow = 36864, ncol = 14)))
# colnames(RSA.by.emotional.PCs$tau) = colnames(dist.emotional.PCs)
# colnames(RSA.by.emotional.PCs$p.value) = colnames(dist.emotional.PCs)
# RSA.by.emotional.PCs$tau = import(paste0("Result/search_light_analysis/",model,"_tau.csv"))
# RSA.by.emotional.PCs$p.value = import(paste0("Result/search_light_analysis/",model,"_p_value.csv"))
# rank.by.emotional.PCs = import(paste0("Result/search_light_analysis/",model,"_neurons_rank.csv"))
expand.grid(neuron=rownames(rank.by.emotional.PCs),attribute=colnames(rank.by.emotional.PCs)) %>%
  as.data.table() %>%
  .[,layer:=rep(factor(rep(paste0("layer",1:12),
                           each=3072),
                       levels = c(paste0("layer",1:12))),
                times=14)] %>%
  .[,tau:=unlist(RSA.by.emotional.PCs$tau)] %>%
  .[,p.value:=unlist(RSA.by.emotional.PCs$p.value)] %>%
  .[,rank:=unlist(rank.by.emotional.PCs)] -> RSA.all
  


# correct statistic value (by FDR) ----
RSA.all[,corrected.p.value:=p.value*.N/rank(p.value)]
RSA.all[,significance:=(corrected.p.value<.01)]
RSA.all[,.(N.signif=sum(significance),Selective=(sum(significance)==1)),by=.(neuron)] %>%
  merge(RSA.all,.) -> RSA.all



# visualize the RDM of a neuron (for display) ----
index.neuron = 15512
RDM.matrix = matrix(0, nrow = 27, ncol = 27, dimnames = list(index.emotions,index.emotions))
RDM.matrix[lower.tri(RDM.matrix)] = unlist(dist.neuron.activation[,15512])
RDM.matrix = RDM.matrix + t(RDM.matrix)
RDM.matrix = EBImage::equalize(RDM.matrix, range = c(0, max(RDM.matrix)), levels = 351)
plot.data = expand.grid(x=index.emotions,y=index.emotions)
plot.data[,"dist"] = as.vector(RDM.matrix)
p.RDM = ggplot(plot.data, aes(x=x,y=y,fill=dist,color=dist))+
  geom_tile()+
  scale_fill_viridis_c(option = "viridis",name="Percentile\nof Distance",breaks=c(min(plot.data$dist),max(plot.data$dist)),labels=c("min","max"))+
  scale_color_viridis_c(option = "viridis",name="Percentile\nof Distance",breaks=c(min(plot.data$dist),max(plot.data$dist)),labels=c("min","max"))+
  scale_x_discrete(name=NULL,breaks=NULL)+
  scale_y_discrete(name=NULL,breaks=NULL)+
  coord_fixed(ratio = 1) + 
  theme(panel.background=element_blank(),
        panel.grid = element_blank(),
        plot.background = element_blank(),
        plot.margin = margin(0,0,0,0, "cm"),
        axis.ticks.length=unit(0,units = "mm"),
        axis.title = element_blank(),
        axis.text = element_blank(),
        legend.position = "none")
ggsave(paste0("Result/search_light_analysis/",model,"_neuron_",index.neuron[1],".pdf"),plot=p.RDM, width=120, height=120, units="mm")




# visualize significant neurons' distribution ----
# select significant neurons
copy(RSA.all) %>%
  .[significance==F,`:=`(tau=NA,rank=NA)] %>%
  .[,neuron:=as.numeric(neuron)%%3072] %>%
  .[neuron==0,neuron:=3072] -> plotdata
# tau value
p.tau = ggplot(plotdata, aes(x=neuron,y=attribute,fill=tau,color=tau))+
  geom_tile(linewidth=0)+
  scale_fill_viridis_c(name="Kendall's tau",option="magma",direction = 1)+
  scale_color_viridis_c(name="Kendall's tau",option="magma",direction = 1)+
  scale_x_discrete(name="Artificial Neurons",position = "bottom",breaks=NULL)+
  scale_y_discrete(name="Conceptual Attributes",limits=rev,position = "left")+
  facet_wrap(vars(layer),ncol=1)+
  theme(panel.background=element_blank(),
        panel.grid = element_blank(),
        axis.ticks.length=unit(0,units = "mm"),
        strip.background = element_blank(),
        strip.text.y = element_blank(),
        title = element_text(family = "Helvetica",size=6),
        text = element_text(family = "Helvetica",size=6))
ggsave(paste0("Result/search_light_analysis/",model,"_signif_tau.pdf"),
       plot = p.tau, width = 225, height = 275, units = "mm")
# count by layer
p.count = ggplot(RSA.all[significance==T,], aes(y=layer,group=attribute))+
  geom_bar()+
  scale_y_discrete(limits=rev,name="Position in LLM",expand = c(0,0))+
  scale_x_continuous(name = "Count",expand = c(0,0), n.breaks = 3, guide = guide_axis(angle=90))+
  scale_fill_viridis_c(name="Count",option = "magma")+
  facet_wrap(vars(attribute),scales = "free")+
  theme_bw()+
  theme(title = element_text(family="Helvetica",size = 6),
        text = element_text(family = "Helvetica",size=6))
ggsave(paste0("Result/search_light_analysis/",model,"_signif_count.pdf"),
       plot = p.count, width = 180, height = 220, units = "mm")



# visualize distribution of top N neurons ----
# select top N neurons
for (index.rank in c(1500,2000,2500,3000,4000,5000,6000)) {
  neuron.set = RSA.all[rank<=index.rank,unique(neuron)]
  RSA.all[rank<=index.rank] %>%
    setorder(.,attribute,rank) %>%
    .[,unique(neuron)] -> neuron.order
  RSA.all[neuron%in%neuron.set] %>%
    .[,neuron:=factor(neuron,levels = neuron.order)] %>%
    .[rank>index.rank,`:=`(tau=NA,rank=NA)] -> plotdata
  # rank (sorted to show overlapping)
  p.rank = ggplot(plotdata, aes(x=neuron,y=attribute,fill=rank,color=rank))+
    geom_tile(linewidth=0)+
    scale_fill_viridis_c(name="Rank\nby tau",option="magma",direction = -1,
                         breaks=c(1,index.rank),guide=guide_colorbar(reverse = TRUE))+
    scale_color_viridis_c(name="Rank\nby tau",option="magma",direction = -1,
                          breaks=c(1,index.rank),guide=guide_colorbar(reverse = TRUE))+
    scale_x_discrete(name="Attribute-Specific Neurons (Sorted by Rank)",position = "top",breaks=NULL)+
    scale_y_discrete(name="Conceptual Attributes",limits=rev,position = "right")+
    theme(panel.background=element_blank(),
          panel.grid = element_blank(),
          # legend.key.size = unit(2,units = "mm"),
          axis.ticks.length = unit(0,units = "mm"),
          strip.background = element_blank(),
          strip.text.y = element_blank(),
          title = element_text(family = "Helvetica",size=6),
          text = element_text(family = "Helvetica",size=6))
  ggsave(paste0("Result/search_light_analysis/",model,"_top",index.rank,"_rank.pdf"),
         # plot = p.rank, width = 128, height = 46, units = "mm")
         plot = p.rank, width = 180, height = 30, units = "mm")
}



# # count by layer
# p.count = ggplot(RSA.all[rank<=N,], aes(y=attribute,group=layer))+
#   geom_bar()+
#   scale_y_discrete(limits=rev,name="Conceptual Attributes",expand = c(0,0))+
#   scale_x_continuous(name = "Count",expand = c(0,0), n.breaks = 3, guide = guide_axis(angle=90))+
#   scale_fill_viridis_c(name="Count",option = "magma")+
#   facet_grid(~layer,scales = "free")+
#   theme_bw()+
#   theme(title = element_text(family="Helvetica",size = 6),
#         text = element_text(family = "Helvetica",size=6))
# p.count
# ggsave(paste0("Result/search_light_analysis/",model,"_top",index.rank,"_count.pdf"),
#        plot = p.count, width = 256, height = 46, units = "mm")



# # for top N neurons
# N=4000
# p.distribution = ggplot(RSA.all[rank<=N,], aes(y=layer,
#                                                        group=attribute))+
#   geom_bar()+
#   scale_y_discrete(limits=rev,name="Attributes",expand = c(0,0))+
#   scale_x_continuous(name = "Count",expand = c(0,0), n.breaks = 3, guide = guide_axis(angle=90))+
#   scale_fill_viridis_c(name="Count",option = "magma")+
#   facet_grid(~attribute,scales = "free")+
#   theme_bw()+
#   theme(title = element_text(family="Helvetica",size = 6),
#         text = element_text(family = "Helvetica",size=6))
# p.distribution
# ggsave(paste0("Result/search_light_analysis/",model,"_distribution_top",N,".pdf"),
#        plot = p.distribution, width = 256, height =46, units = "mm")



# # for significant neurons
# neuron.set = RSA.all[significance==T,unique(neuron)]
# RSA.all[significance==T] %>%
#   setorder(.,N.signif,attribute,-tau) %>%
#   .[,unique(neuron)] -> neuron.order
# RSA.all[neuron%in%neuron.set] %>%
#   .[,neuron:=factor(neuron,levels = neuron.order)] %>%
#   .[significance==F,`:=`(tau=NA,rank=NA)] -> plotdata
# p.tau = ggplot(plotdata, aes(x=neuron,y=attribute,fill=tau,color=tau))+
#   geom_tile(linewidth=0)+
#   scale_fill_viridis_c(name="Kendall's tau",option="magma",direction = 1)+
#   scale_color_viridis_c(name="Kendall's tau",option="magma",direction = 1)+
#   scale_x_discrete(name="Attribute-Specific Neurons",position = "top",breaks=NULL)+
#   scale_y_discrete(name="Conceptual Attributes",limits=rev,position = "right")+
#   theme(panel.background=element_blank(),
#         panel.grid = element_blank(),
#         axis.ticks.length=unit(0,units = "mm"),
#         strip.background = element_blank(),
#         strip.text.y = element_blank(),
#         title = element_text(family = "Helvetica",size=6),
#         text = element_text(family = "Helvetica",size=6))
# ggsave(paste0("Result/search_light_analysis/",model,"_tau_signif.pdf"), plot = p.tau, width = 256, height = 46, units = "mm")
# p.rank = ggplot(plotdata, aes(x=neuron,y=attribute,fill=rank,color=rank))+
#   geom_tile(linewidth=0)+
#   scale_fill_viridis_c(name="Rank\nby tau",option="magma",direction = -1,guide=guide_colorbar(reverse = TRUE))+
#   scale_color_viridis_c(name="Rank\nby tau",option="magma",direction = -1,guide=guide_colorbar(reverse = TRUE))+
#   scale_x_discrete(name="Attribute-Specific Neurons",position = "top",breaks=NULL)+
#   scale_y_discrete(name="Conceptual Attributes",limits=rev,position = "right")+
#   theme(panel.background=element_blank(),
#         panel.grid = element_blank(),
#         axis.ticks.length=unit(0,units = "mm"),
#         strip.background = element_blank(),
#         strip.text.y = element_blank(),
#         title = element_text(family = "Helvetica",size=6),
#         text = element_text(family = "Helvetica",size=6))
# ggsave(paste0("Result/search_light_analysis/",model,"_rank_signif.pdf"), plot = p.rank, width = 256, height = 46, units = "mm")



# p.tau = ggplot(plotdata, aes(x=neuron,y=attribute,fill=tau,color=tau))+
#   geom_tile(linewidth=0)+
#   scale_fill_viridis_c(name="Kendall's tau",option="magma",direction = 1)+
#   scale_color_viridis_c(name="Kendall's tau",option="magma",direction = 1)+
#   scale_x_discrete(name="Attribute-Specific Neurons",position = "top",breaks=NULL)+
#   scale_y_discrete(name="Conceptual Attributes",limits=rev,position = "right")+
#   theme(panel.background=element_blank(),
#         panel.grid = element_blank(),
#         axis.ticks.length=unit(0,units = "mm"),
#         strip.background = element_blank(),
#         strip.text.y = element_blank(),
#         title = element_text(family = "Helvetica",size=6),
#         text = element_text(family = "Helvetica",size=6))
# ggsave(paste0("Result/search_light_analysis/",model,"_tau_top",index.rank,".pdf"), plot = p.tau, width = 128, height = 46, units = "mm")

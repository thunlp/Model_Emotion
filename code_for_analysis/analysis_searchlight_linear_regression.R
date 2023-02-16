library(bruceR)
library(boot)
set.wd()
set.wd("../")

# load data ----
import("Data/preprocessed/emotional_features/score_emotional_PCs.csv",header = T,as="data.table") %>%
  scale() %>%
  as.data.table() -> emotion.score
feature.names = colnames(emotion.score)
colnames(emotion.score) = sub("-",".",colnames(emotion.score))
import("Data/preprocessed/PLM/prompt_tuning_RoBERTa/neurons_activation/by_neuron/avg_param_of_activation.csv",header = T,as="data.table") %>%
  scale() %>%
  as.data.table() -> neuron.activation

# fit linear regression model with affective features ----
fit.affective = function(x){
  if (any(is.na(x))){
    return(list(matrix(NA,nrow = 2, ncol = 3,
                       dimnames = list(c("arousal","valence"),
                                       c("Estimate","t value","Pr(>|t|)"))),
                numeric(adj.r.squared = NA, F.value = NA, p.value = NA, p.value = NA)))
  }else{
    data=data.table(x,emotion.score)
    colnames(data)[1] = "y"
    a = lm(y ~ arousal + valence, data=data)
    b = summary(a)
    return(list(b$coefficients[2:3,c(1,3,4)],
                c(adj.r.squared = b$adj.r.squared,
                  F.value = b$fstatistic[1],
                  p.value = pf(b$fstatistic[1],b$fstatistic[2],b$fstatistic[3],lower.tail=FALSE))))
  }
}
neuron.activation %>%
  lapply(fit.affective) -> fitted.affective
# extract BETA and p-value of each feature as rank index
fitted.affective %>%
  lapply(function(x){as.data.table(t(x[[1]][,1]))}) %>%
  rbindlist() -> fitted.affective.beta
fitted.affective %>%
  lapply(function(x){as.data.table(t(x[[1]][,2]))}) %>%
  rbindlist() -> fitted.affective.beta.t
fitted.affective %>%
  lapply(function(x){as.data.table(t(x[[1]][,3]))}) %>%
  rbindlist() -> fitted.affective.beta.p # DF=24
# extract R2 and F-value of feature combination as rank index
fitted.affective %>%
  lapply(function(x){as.data.table(x[[2]][1])}) %>%
  rbindlist() -> fitted.affective.R2
fitted.affective %>%
  lapply(function(x){as.data.table(x[[2]][2])}) %>%
  rbindlist() -> fitted.affective.R2.F
fitted.affective %>%
  lapply(function(x){as.data.table(x[[2]][3])}) %>%
  rbindlist() -> fitted.affective.R2.p # DF-numerator = 2, DF-denominator = 24
rm(fitted.affective)

# fit linear regression model with basic features ----
fit.basic = function(x){
  if (any(is.na(x))){
    return(list(matrix(NA,nrow = 6, ncol = 3,
                       dimnames = list(c("happy","anger","sad","fear","surprise","disgust"),
                                       c("Estimate","t value","Pr(>|t|)"))),
                numeric(adj.r.squared = NA, F.value = NA, p.value = NA, p.value = NA)))
  }else{
    data=data.table(x,emotion.score)
    colnames(data)[1] = "y"
    a = lm(y ~ happy + anger + sad + fear + surprise + disgust, data=data)
    b = summary(a)
    return(list(b$coefficients[2:7,c(1,3,4)],
                c(adj.r.squared = b$adj.r.squared,
                  F.value = b$fstatistic[1],
                  p.value = pf(b$fstatistic[1],b$fstatistic[2],b$fstatistic[3],lower.tail=FALSE))))
  }
}
neuron.activation %>%
  lapply(fit.basic) -> fitted.basic
# extract BETA and p-value of each feature as rank index
fitted.basic %>%
  lapply(function(x){as.data.table(t(x[[1]][,1]))}) %>%
  rbindlist() -> fitted.basic.beta
fitted.basic %>%
  lapply(function(x){as.data.table(t(x[[1]][,2]))}) %>%
  rbindlist() -> fitted.basic.beta.t
fitted.basic %>%
  lapply(function(x){as.data.table(t(x[[1]][,3]))}) %>%
  rbindlist() -> fitted.basic.beta.p # DF=20
# extract R2 and F-value of feature combination as rank index
fitted.basic %>%
  lapply(function(x){as.data.table(x[[2]][1])}) %>%
  rbindlist() -> fitted.basic.R2
fitted.basic %>%
  lapply(function(x){as.data.table(x[[2]][2])}) %>%
  rbindlist() -> fitted.basic.R2.F
fitted.basic %>%
  lapply(function(x){as.data.table(x[[2]][3])}) %>%
  rbindlist() -> fitted.basic.R2.p # DF-numerator = 6, DF-denominator = 20
rm(fitted.basic)

# fit linear regression model with appraisal features ----
fit.appraisal = function(x){
  if (any(is.na(x))){
    return(list(matrix(NA,nrow = 6, ncol = 3,
                       dimnames = list(c("control","fairness","self.related","other.related","expectedness","non.novelty"),
                                       c("Estimate","t value","Pr(>|t|)"))),
                numeric(adj.r.squared = NA, F.value = NA, p.value = NA, p.value = NA)))
  }else{
    data=data.table(x,emotion.score)
    colnames(data)[1] = "y"
    a = lm(y ~ control + fairness + self.related + other.related + expectedness + non.novelty, data=data)
    b = summary(a)
    return(list(b$coefficients[2:7,c(1,3,4)],
                c(adj.r.squared = b$adj.r.squared,
                  F.value = b$fstatistic[1],
                  p.value = pf(b$fstatistic[1],b$fstatistic[2],b$fstatistic[3],lower.tail=FALSE))))
  }
}
neuron.activation %>%
  lapply(fit.appraisal) -> fitted.appraisal
# extract BETA and p-value of each feature as rank index
fitted.appraisal %>%
  lapply(function(x){as.data.table(t(x[[1]][,1]))}) %>%
  rbindlist() -> fitted.appraisal.beta
fitted.appraisal %>%
  lapply(function(x){as.data.table(t(x[[1]][,2]))}) %>%
  rbindlist() -> fitted.appraisal.beta.t
fitted.appraisal %>%
  lapply(function(x){as.data.table(t(x[[1]][,3]))}) %>%
  rbindlist() -> fitted.appraisal.beta.p # DF=20
# extract R2 and F-value of feature combination as rank index
fitted.appraisal %>%
  lapply(function(x){as.data.table(x[[2]][1])}) %>%
  rbindlist() -> fitted.appraisal.R2
fitted.appraisal %>%
  lapply(function(x){as.data.table(x[[2]][2])}) %>%
  rbindlist() -> fitted.appraisal.R2.F
fitted.appraisal %>%
  lapply(function(x){as.data.table(x[[2]][3])}) %>%
  rbindlist() -> fitted.appraisal.R2.p # DF-numerator = 6, DF-denominator = 20
rm(fitted.appraisal)

# merge fitting results ----
fitted.beta = cbind(fitted.affective.beta,fitted.basic.beta,fitted.appraisal.beta)
fitted.beta.t = cbind(fitted.affective.beta.t,fitted.basic.beta.t,fitted.appraisal.beta.t)
fitted.beta.p = cbind(fitted.affective.beta.p,fitted.basic.beta.p,fitted.appraisal.beta.p)
fitted.R2 = cbind(fitted.affective.R2,fitted.basic.R2,fitted.appraisal.R2)
fitted.R2.F = cbind(fitted.affective.R2.F,fitted.basic.R2.F,fitted.appraisal.R2.F)
fitted.R2.p = cbind(fitted.affective.R2.p,fitted.basic.R2.p,fitted.appraisal.R2.p)
rm(fitted.affective.beta,fitted.basic.beta,fitted.appraisal.beta,
   fitted.affective.beta.t,fitted.basic.beta.t,fitted.appraisal.beta.t,
   fitted.affective.beta.p,fitted.basic.beta.p,fitted.appraisal.beta.p,
   fitted.affective.R2,fitted.basic.R2,fitted.appraisal.R2,
   fitted.affective.R2.F,fitted.basic.R2.F,fitted.appraisal.R2.F,
   fitted.affective.R2.p,fitted.basic.R2.p,fitted.appraisal.R2.p)
colnames(fitted.beta) = feature.names
colnames(fitted.beta.t) = feature.names
colnames(fitted.beta.p) = feature.names
colnames(fitted.R2) = c("Affective Space","Basic Emotions Space","Appraisal Space")
colnames(fitted.R2.F) = c("Affective Space","Basic Emotions Space","Appraisal Space")
colnames(fitted.R2.p) = c("Affective Space","Basic Emotions Space","Appraisal Space")
fitted.by.emotional.PCs = list(beta = fitted.beta,
                               t.value = fitted.beta.t,
                               p.value = fitted.beta.p)
fitted.by.emotion.space = list(R2 = fitted.R2,
                               F.value = fitted.R2.F,
                               p.value = fitted.R2.p)
rm(fitted.beta,fitted.beta.t,fitted.beta.p,
   fitted.R2,fitted.R2.F,fitted.R2.p)

# correct statistic value (by FDR) ----
fitted.by.emotional.PCs[["corrected.p.value"]] = fitted.by.emotional.PCs$p.value * sum(!is.na(fitted.by.emotional.PCs$p.value[,1])) * ncol(fitted.by.emotional.PCs$p.value) / rank(fitted.by.emotional.PCs$p.value)
fitted.by.emotional.PCs[["significance"]] = fitted.by.emotional.PCs$corrected.p.value<0.05
fitted.by.emotional.PCs[["signif.beta"]] = fitted.by.emotional.PCs$beta
fitted.by.emotional.PCs$signif.beta[!fitted.by.emotional.PCs$significance] = NA
fitted.by.emotional.PCs[["signif.t.value"]] = fitted.by.emotional.PCs$t.value
fitted.by.emotional.PCs$signif.t.value[!fitted.by.emotional.PCs$significance] = NA
fitted.by.emotion.space[["corrected.p.value"]] = fitted.by.emotion.space$p.value * sum(!is.na(fitted.by.emotion.space$p.value[,1])) * ncol(fitted.by.emotion.space$p.value) / rank(fitted.by.emotion.space$p.value)
fitted.by.emotion.space[["significance"]] = fitted.by.emotion.space$corrected.p.value<0.05
fitted.by.emotion.space[["signif.R2"]] = fitted.by.emotion.space$R2
fitted.by.emotion.space$signif.R2[!fitted.by.emotion.space$significance] = NA
fitted.by.emotion.space[["signif.F.value"]] = fitted.by.emotion.space$F.value
fitted.by.emotion.space$signif.F.value[!fitted.by.emotion.space$significance] = NA

# rank neuron by statistic value ----
lapply(-abs(fitted.by.emotional.PCs$beta), rank, na.last = "keep") %>%
  as.data.frame -> rank.by.emotional.PCs
lapply(-fitted.by.emotion.space$R2, rank, na.last = "keep") %>%
  as.data.frame -> rank.by.emotion.space
colnames(rank.by.emotional.PCs) = feature.names
colnames(rank.by.emotion.space) = c("Affective Space","Basic Emotions Space","Appraisal Space")

# save results----
export(fitted.by.emotional.PCs$beta,"Result/search_light_analysis/by_emotional_PCs/linear_regression_to_avg_activation/beta.csv")
export(fitted.by.emotional.PCs$t.value,"Result/search_light_analysis/by_emotional_PCs/linear_regression_to_avg_activation/t_value.csv")
export(fitted.by.emotional.PCs$p.value,"Result/search_light_analysis/by_emotional_PCs/linear_regression_to_avg_activation/p_value.csv")
export(rank.by.emotional.PCs,"Result/search_light_analysis/by_emotional_PCs/linear_regression_to_avg_activation/neurons_rank.csv")
export(fitted.by.emotion.space$R2,"Result/search_light_analysis/by_emotion_space/linear_regression_to_avg_activation/R2.csv")
export(fitted.by.emotion.space$F.value,"Result/search_light_analysis/by_emotion_space/linear_regression_to_avg_activation/F_value.csv")
export(fitted.by.emotion.space$p.value,"Result/search_light_analysis/by_emotion_space/linear_regression_to_avg_activation/p_value.csv")
export(rank.by.emotion.space,"Result/search_light_analysis/by_emotion_space/linear_regression_to_avg_activation/neurons_rank.csv")

# visualize neuron selectivity ----
library(pheatmap)
library(viridis)
# signif beta
plotdata = fitted.by.emotional.PCs$signif.beta
plotdata = plotdata[!is.na(rowMeans(plotdata,na.rm = T)),]
plotdata = t(setorder(plotdata))
p = pheatmap(plotdata, border_color = NA,
             cluster_rows = F, cluster_cols = F,
             show_colnames = F,
             filename = "Result/search_light_analysis/by_emotional_PCs/linear_regression_to_avg_activation/signif_beta.pdf")
# signif t value
plotdata = fitted.by.emotional.PCs$signif.t.value
plotdata = plotdata[!is.na(rowMeans(plotdata,na.rm = T)),]
plotdata = t(setorder(plotdata))
p = pheatmap(plotdata, border_color = NA,
             cluster_rows = F, cluster_cols = F,
             show_colnames = F,
             filename = "Result/search_light_analysis/by_emotional_PCs/linear_regression_to_avg_activation/signif_t_value.pdf")
# signif R2
plotdata = fitted.by.emotion.space$signif.R2
plotdata = plotdata[!is.na(rowMeans(plotdata,na.rm = T)),]
plotdata = t(setorder(plotdata))
p = pheatmap(plotdata, border_color = NA,
             color = inferno(n=ncol(plotdata)),
             cluster_rows = F, cluster_cols = F,
             show_colnames = F,
             filename = "Result/search_light_analysis/by_emotion_space/linear_regression_to_avg_activation/signif_R2.pdf")
# signif F value
plotdata = fitted.by.emotion.space$signif.F.value
plotdata = plotdata[!is.na(rowMeans(plotdata,na.rm = T)),]
plotdata = t(setorder(plotdata))
p = pheatmap(plotdata, border_color = NA,
             color = inferno(n=ncol(plotdata)),
             cluster_rows = F, cluster_cols = F,
             show_colnames = F,
             filename = "Result/search_light_analysis/by_emotion_space/linear_regression_to_avg_activation/signif_F_value.pdf")
rm(p,plotdata)
# overlap of three spaces
library(VennDiagram)
library(RColorBrewer)
myCol <- brewer.pal(3, "Pastel2")
venn.diagram(
  x = list(rownames(fitted.by.emotion.space$R2)[fitted.by.emotion.space$significance[,1]],
           rownames(fitted.by.emotion.space$R2)[fitted.by.emotion.space$significance[,2]],
           rownames(fitted.by.emotion.space$R2)[fitted.by.emotion.space$significance[,3]]),
  category.names = c("Affective\nSpace" , "Basic Emotions\nSpace" , "Appraisal\nSpace"),
  filename = 'Result/search_light_analysis/by_emotion_space/linear_regression_to_avg_activation/overlap_of_signif_neurons.png',
  output=F,
  
  # Output features
  imagetype="png" ,
  height = 480 , 
  width = 480 , 
  resolution = 300,
  compression = "lzw",
  
  # Circles
  lwd = 3,
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



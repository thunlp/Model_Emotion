library(bruceR)
set.wd()
set.wd("../")

## set raw data folder
model = "RoBERTa"
folder = paste0("Data/raw_data/PLM/prompt_tuning_",model,"/neurons_activation_before_relu/")

##------activation------
## load data and save individual parameters and distance
emotions = import("Data/preprocessed/reordered_emotions.csv", header = FALSE)[1:27,1]
seed = c(1,3,5,7,9,11,13,15,17,19,42,100)
merged.parameters = array(dim=c(27,36864,12),dimnames = list(emotions,c(1:36864),seed))
dist.parameter = array(dim=c(351,36864,12),dimnames = list(c(1:351),c(1:36864),seed))
n = 0
for (s in seed) {
  n = n+1
  for (e in 1:27) {
    merged.parameters[e,,n] = as.numeric(data.table::fread(file=paste(folder,emotions[e],"-",s,".csv",sep = ""),header = F, skip = 1))
    print(paste("n =",n,", e =",e))
  }
  for (p in 1:36864) {
    d = as.matrix(dist(merged.parameters[,p,n],method = "euclidean"))
    dist.parameter[,p,n] = d[lower.tri(d)]
  }
  export(dist.parameter[,,n], paste0("Data/preprocessed/PLM/prompt_tuning_",model,"/neurons_activation/by_neuron/dist_of_activation_ind",n,".csv"))
  export(merged.parameters[,,n], paste0("Data/preprocessed/PLM/prompt_tuning_",model,"/neurons_activation/by_neuron/param_of_activation_ind",n,".csv"))
}
rm(n, s, e, RDM)
## save averaged parameters of each neuron
avg.merged.parameters = rowMeans(merged.parameters, dims = 2)
export(avg.merged.parameters, paste0("Data/preprocessed/PLM/prompt_tuning_",model,"/neurons_activation/by_neuron/avg_param_of_activation.csv"))
## save averaged distance of each neuron
avg.dist.parameter = rowMeans(dist.parameter, dims = 2)
export(avg.dist.parameter, paste0("Data/preprocessed/PLM/prompt_tuning_",model,"/neurons_activation/by_neuron/avg_dist_of_activation.csv"))


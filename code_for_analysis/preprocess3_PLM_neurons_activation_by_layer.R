library(bruceR)
set.wd()
set.wd("../")

## set raw data folder
folder = "Data/raw_data/PLM/prompt_tuning_RoBERTa/neurons_activation/"

## load data and save individual parameters
emotions = import("Data/preprocessed/reordered_emotions.csv", header = FALSE)[1:27,1]
seed = c(1,3,5,7,9,11,13,15,17,19,42,100)
merged.parameters = array(dim=c(27,36864,12),dimnames = list(emotions,c(1:36864),seed))
neurons.activation.layer = array(dim=c(12,351,12),dimnames = list(seed,c(1:351),c(1:12)))
n = 0
for (s in seed) {
  n = n+1
  for (e in 1:27) {
    merged.parameters[e,,n] = as.numeric(data.table::fread(file=paste(folder,emotions[e],"-",s,".csv",sep = ""),header = F, skip = 1))
    print(paste("n =",n,", e =",e))
  }
  for (l in c(1:12)) {
    RDM = 1-lsa::cosine(t(merged.parameters[,c((3072*(l-1)+1):(3072*l)),n]))
    neurons.activation.layer[n,,l] = RDM[lower.tri(RDM)]
  }
}
rm(n, s, e, l, RDM)

## save distance of each layers
for (l in c(1:12)) {
  export(neurons.activation.layer[,,l], paste0("Data/preprocessed/PLM/prompt_tuning_RoBERTa/neurons_activation/by_layer/dist_of_layer_",l,".csv"))
}


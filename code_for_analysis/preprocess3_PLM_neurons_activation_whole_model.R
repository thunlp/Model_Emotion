library(bruceR)
set.wd()
set.wd("../")

## set raw data folder
folder = "Data/raw_data/PLM/prompt_tuning_RoBERTa/neurons_activation_before_rule/"

## load data and save individual parameters
emotions = import("Data/preprocessed/reordered_emotions.csv", header = FALSE)[1:27,1]
seed = c(1,3,5,7,9,11,13,15,17,19,42,100)
merged.parameters = array(dim=c(27,36864,12),dimnames = list(emotions,c(1:36864),seed))
neurons.activation.whole = as.data.frame(matrix(nrow = 12,ncol = 351,dimnames = list(seed,c(1:351))))
n = 0
for (s in seed) {
  n = n+1
  for (e in 1:27) {
    merged.parameters[e,,n] = as.numeric(data.table::fread(file=paste(folder,emotions[e],"-",s,".csv",sep = ""),header = F, skip = 1))
    print(paste("n =",n,", e =",e))
  }
  RDM = 1-lsa::cosine(t(merged.parameters[,,n]))
  neurons.activation.whole[n,] = RDM[lower.tri(RDM)]
}
rm(n, s, e, RDM)

## save distance of whole model
export(neurons.activation.whole, "Data/preprocessed/PLM/prompt_tuning_RoBERTa/neurons_activation/whole_model/ind_dist_neurons_activation.csv")
neurons.activation.whole = data.frame(colMeans(neurons.activation.whole))
colnames(neurons.activation.whole) = "Neurons Activation"
export(neurons.activation.whole, "Data/preprocessed/PLM/prompt_tuning_RoBERTa/neurons_activation/whole_model/avg_dist_neurons_activation.csv")

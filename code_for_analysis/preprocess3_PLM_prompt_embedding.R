library(bruceR)
set.wd()
set.wd("../")

## set raw data folder
folder = "Data/raw_data/PLM/prompt_tuning_RoBERTa/prompt_embedding/"

## load data and save individual parameters
emotions = import("Data/preprocessed/reordered_emotions.csv", header = FALSE)[1:27,1]
seed = c(1,3,5,7,9,11,13,15,17,19,42,100)
merged.parameters = array(dim=c(27,76800,12),dimnames = list(emotions,c(1:76800),seed))
prompt.embedding = as.data.frame(matrix(nrow = 12,ncol = 351,dimnames = list(seed,c(1:351))))
n = 0
for (s in seed) {
  n = n+1
  for (e in 1:27) {
    merged.parameters[e,,n] = as.numeric(data.table::fread(file=paste(folder,emotions[e],"-",s,".csv",sep = ""),header = F, skip = 1))
    print(paste("n =",n,", e =",e))
  }
  # export(merged.parameters[,,n], paste("Data/preprocessed/PLM/prompt_tuning_RoBERTa/ind",s,"_param_prompt_embedding.csv",sep = ""))
  RDM = 1-lsa::cosine(t(merged.parameters[,,n]))
  prompt.embedding[n,] = RDM[lower.tri(RDM)]
}
rm(n, s, e, RDM)

# ## save averaged parameters
# merged.parameters = rowMeans(merged.parameters, dims = 2)
# export(merged.parameters, "Data/preprocessed/PLM/prompt_tuning_RoBERTa/avg_param_prompt_embedding.csv")

# save distance
export(prompt.embedding, "Data/preprocessed/PLM/prompt_tuning_RoBERTa/prompt_embedding/ind_dist_prompt_embedding.csv")
prompt.embedding = data.frame(colMeans(prompt.embedding))
colnames(prompt.embedding) = "Prompt Embedding"
export(prompt.embedding, "Data/preprocessed/PLM/prompt_tuning_RoBERTa/prompt_embedding/avg_dist_prompt_embedding.csv")




library(bruceR)
set.wd()
set.wd("../")

## set raw data folder
folder = "Data/raw_data/PLM/word_embedding_GloVe/reddit768d/"

## load data
emotions = import("Data/preprocessed/reordered_emotions.csv", header = FALSE)[1:27,1]
seed = c(1:12)
embedding = as.data.frame(matrix(nrow = 12,ncol = 351,dimnames = list(seed,c(1:351))))
n = 0
for (s in seed) {
  n = n+1
  param = as.matrix(import(paste(folder,"class28_glove_reddit768d_",s,".csv",sep = ""))[,1:27])
  RDM = 1-lsa::cosine(param)
  embedding[n,] = RDM[lower.tri(RDM)]
}
rm(n, s, param, RDM)

## save distance
export(embedding, "Data/preprocessed/PLM/word_embedding_GloVe/ind_dist_reddit768d.csv")
embedding = data.frame(colMeans(embedding))
colnames(embedding) = "Word Embedding"
export(embedding, "Data/preprocessed/PLM/word_embedding_GloVe/avg_dist_reddit768d.csv")



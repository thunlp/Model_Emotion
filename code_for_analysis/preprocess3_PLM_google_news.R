library(bruceR)
set.wd()
set.wd("../")

## load data
embedding = import("Data/raw_data/PLM/word_embedding_GloVe/google_news.csv")[,1:27]

## dist
index.dist = unlist(import("Data/preprocessed/reorder_distance.csv",header = F))
embedding = 1-lsa::cosine(as.matrix(embedding))
embedding = data.frame(embedding[lower.tri(embedding)])
embedding = data.frame(embedding[index.dist,1])
colnames(embedding) = "Word Embedding (google news)"
export(embedding, "Data/preprocessed/PLM/word_embedding_GloVe/dist_google_news.csv")

library(bruceR)
set.wd()
set.wd("../")

## load data
GoEmotions.annotation = import("Data/raw_data/control/GoEmotions_annotation.csv")
rownames(GoEmotions.annotation) = GoEmotions.annotation[,1]
GoEmotions.annotation = GoEmotions.annotation[1:27,2:28]

## dist
index.dist = unlist(import("Data/preprocessed/reorder_distance.csv",header = F))
GoEmotions.annotation = data.frame(1-as.matrix(GoEmotions.annotation[lower.tri(GoEmotions.annotation)]))
GoEmotions.annotation = data.frame(GoEmotions.annotation[index.dist,1])
colnames(GoEmotions.annotation) = "Corpus Annotation"
export(GoEmotions.annotation, "Data/preprocessed/control/dist_GoEmotions_annotation.csv")

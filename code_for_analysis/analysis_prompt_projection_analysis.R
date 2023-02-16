library(bruceR)
set.wd()
set.wd("../")

## load data --------
# prompt vector
folder = "Data/raw_data/PLM/prompt_tuning_RoBERTa/prompt_embedding/"
emotions = import("Data/preprocessed/reordered_emotions.csv", header = FALSE)[1:27, 1]
seed = c(1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 42, 100)
prompt.vector = data.table(emotion = factor(rep(emotions, each = 12), levels = emotions),
                           random.seed = factor(rep(seed, 27)))
prompt.vector = cbind(prompt.vector,
                      setNames(data.table(matrix(
                        0, nrow = 12 * 27, ncol = 76800
                      )),
                      paste0("Dim", c(1:76800))))
for (i in 1:nrow(prompt.vector)) {
  prompt.vector[i, paste0("Dim", c(1:76800))] = data.table::fread(
    file = paste0(
      folder,
      prompt.vector$emotion[i],
      "-",
      prompt.vector$random.seed[i],
      ".csv"
    ),
    header = F,
    skip = 1
  )
}
rm(folder, i)
# candidate properties
emotion.property = cbind(
  data.table(emotion = factor(emotions, levels = emotions)),
  import(
    "Data/preprocessed/emotional_features/score_emotional_PCs.csv",
    as = "data.table"
  )
)
emotion.property = emotion.property[rep(1:27, each = 12), ]

## reduce the dimensions of prompt space
prompt.PCs = bruceR::PCA(
  prompt.vector,
  var = "Dim",
  items = c(1:76800),
  rotation = "none",
  #nfactors = "parallel"
)

## canon corr
cca = cancor(emotion.property[,c(2:15)],prompt.vector[,c(3:76802)])



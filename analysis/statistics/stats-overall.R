library("stats")
library("tidyverse")
library("PMCMRplus")
library("rcompanion")
library("gdata")

df <- matrix(c(
  5,6,1,3,4,0,2,7,
  3,7,4,0,2,6,1,5,
  5,6,2,3,4,1,0,7,
  4,6,1,2,5,3,0,7,
  4,5,0,3,1,2,6,7,
  4,6,5,3,1,2,0,7,
  6,7,2,0,3,4,1,5
),
nrow=7,
byrow=TRUE,
dimnames = list(
  c("BlobDS1","BlobDS2","DataClassDS1","DataClassDS2","LongMethodDS1","LongMethodDS2","FeatureEnvyDS1"),
  c("CTree","FDA","KNN","kSVM","libSVM","MDA","NaiveBayes","RandomForest")
)
)

print(friedmanTest(df))
print(frdAllPairsNemenyiTest(df))

vdadf <- as.data.frame(unmatrix(x=df,byrow=TRUE))
vdadf$seps <- rownames(vdadf)
colnames(vdadf) <- c("value", "seps")
vdadf <- separate(data=vdadf, col="seps", into=c("DS", "algorithm"))

print(multiVDA(value ~ algorithm, data=vdadf))

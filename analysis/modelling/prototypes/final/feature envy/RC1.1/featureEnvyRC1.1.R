#
# codebeat model builder scripts - a set of scripts to build models used for code smells detection in codebeat
# Copyright (C) 2018-2022 code quest sp. z o.o.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#

#-- (Command line) Attributes -------------------------------------------------
argv <- commandArgs()
argv$date = Sys.Date() # "2019-06-06"
argv$dataSource = "/Users/lma/Documents/lm16/CQ/gitALL/ncbr-model-builder/src/R/prototypes/final/2019-09-13-all-with-metrics.csv" # this is the final data set (2019-09-13)!
argv$utilsSource = "/Users/lma/Documents/lm16/CQ/gitALL/ncbr-model-builder/src/R/prototypes/final/utilsLM2.R"  
argv$ncbrModels = "./tmp/ncbr-models/models/" 
argv$targetPath = ""
argv$targetModel = "target.model"
argv$targetData = "target.data"
argv$targetPerf = "target.perf"
argv$evaluate = FALSE
argv$build = TRUE
argv$beta = 0.8
argv$outputFolder = "/Users/lma/Documents/lm16/CQ/gitALL/ncbr-model-builder/src/R/prototypes/"
#argv$learner = "classif.xgboost" #"classif.kknn"
argv$version = "RC1.1"
argv$investigatedSmell = "feature envy" #"blob", "data class", "long method", "feature envy"


investigatedSmell <- argv$investigatedSmell

outputFolder = stringr::str_c(argv$outputFolder, "final", "/", investigatedSmell, "/", argv$version, "/")

if (!file.exists(outputFolder))
  dir.create(outputFolder, recursive = TRUE)


#-- Utils: 
source(argv$utilsSource)


#-- Set seed ------------------------------------------------------------------
#set.seed(1234)
set.seed(123, "L'Ecuyer")


#-- Detect the number of CPUs -------------------------------------------------
cpus = parallel::detectCores()
cpusToUse = cpus - 1
# Define number of CPU cores to use when training models
#parallelStartSocket(cpusToUse)
parallelMap::parallelStartMulticore(cpus = cpusToUse)



#--  LOAD DATA ----------------------------------------------------------------
importedData <- readr::read_csv(file = argv$dataSource) # return a tibble

myMeasures = list(
  multiclass.AvFbeta
)


#-- DATA PREPROCESSING PER SMELL

dataPerSmell <-
  importedData %>%
  cleanDataByConvertingToCorrectTypes() %>%
  cleanDataByTakingCareOfMultipleReviewsOfTheSameSample() %>%
  cleanDataByRemovingIrrelevantData(investigatedSmell) #%>%
  #NA_preproc() 
  # contrasts can be applied only to factors with 2 or more levels
  # "The problem isn't with the prediction, it's with the dummy features wrapper, which assumes that factors have at least 2 levels (otherwise a dummy encoding doesn't make sense)."
  # https://github.com/mlr-org/mlr/issues/2422

  #mlr::removeConstantFeatures(perc = 0.02,
  #                            dont.rm = "severity") %>%
  #mlr::normalizeFeatures()


taskPerSmell <- makeClassifTask(id = investigatedSmell,
                                data = dataPerSmell,
                                target = "severity")



sinkOn <- function() {
  sink(stringr::str_c(outputFolder, "_", investigatedSmell, "_", argv$version, "_results.txt"), split = TRUE, append = TRUE) #send output to both: file AND console 
}

#--MODELS

#-- classif.naiveBayes


classif.sda <- makeUniversalLearner("classif.sda")
classif.sda.resample.rep10cv10stratify = tryLoadResample(
  outputFolder = outputFolder, 
  rdsFileName = stringr::str_c(classif.sda$id, ".resample.rep10cv10stratify.RDS"), 
  save = TRUE, 
  learner = classif.sda,
  task = taskPerSmell, 
  resampling = rep10cv10stratify,
  measures = myMeasures
)
sinkOn()
classif.sda.resample.rep10cv10stratify
sink()

# > classif.sda.resample.rep10cv10stratify
# Resample Result
# Task: feature envy
# Learner: classif.sda.preproc.imputed
# Aggr perf: multiclass.AvFbeta.test.mean=0.2003800 (0.2078935)
# Runtime: 20.9076

classif.lda <- makeUniversalLearner("classif.lda")
classif.lda.resample.rep10cv10stratify = tryLoadResample(
  outputFolder = outputFolder, 
  rdsFileName = stringr::str_c(classif.lda$id, ".resample.rep10cv10stratify.RDS"), 
  save = TRUE, 
  learner = classif.lda,
  task = taskPerSmell, 
  resampling = rep10cv10stratify,
  measures = myMeasures
)
sinkOn()
classif.lda.resample.rep10cv10stratify
sink()
# > classif.lda.resample.rep10cv10stratify
# Resample Result
# Task: feature envy
# Learner: classif.lda.preproc.imputed
# Aggr perf: multiclass.AvFbeta.test.mean=0.2250432
# Runtime: 19.8164


classif.sparseLDA <- makeUniversalLearner("classif.sparseLDA")
classif.sparseLDA.resample.rep10cv10stratify = tryLoadResample(
  outputFolder = outputFolder, 
  rdsFileName = stringr::str_c(classif.sparseLDA$id, ".resample.rep10cv10stratify.RDS"), 
  save = TRUE, 
  learner = classif.sparseLDA,
  task = taskPerSmell, 
  resampling = rep10cv10stratify,
  measures = myMeasures
)
sinkOn()
classif.sparseLDA.resample.rep10cv10stratify
sink()
# > classif.sparseLDA.resample.rep10cv10stratify
# Resample Result
# Task: feature envy
# Learner: classif.sparseLDA.preproc.imputed
# Aggr perf: multiclass.AvFbeta.test.mean=0.2098941
# Runtime: 27.4384


classif.ctree <- makeUniversalLearner("classif.ctree")
classif.ctree.resample.rep10cv10stratify = tryLoadResample(
  outputFolder = outputFolder, 
  rdsFileName = stringr::str_c(classif.ctree$id, ".resample.rep10cv10stratify.RDS"), 
  save = TRUE, 
  learner = classif.ctree,
  task = taskPerSmell, 
  resampling = rep10cv10stratify,
  measures = myMeasures
)
sinkOn()
classif.ctree.resample.rep10cv10stratify
sink()

# Train top models #
myFinalTask <- taskPerSmell


# Train model: lda
myFinalLearner <- classif.lda
myFinalModelTrained <- tryLoad(folder = outputFolder, 
                               filename = str_c("FinalTrainedLearner.", myFinalLearner$id, ".RDS"), 
                               save = TRUE,
                               expr = {
                                 mlr::train(myFinalLearner, myFinalTask)
                               })


myFinalLearner <- classif.sparseLDA
myFinalModelTrained <- tryLoad(folder = outputFolder, 
                               filename = str_c("FinalTrainedLearner.", myFinalLearner$id, ".RDS"), 
                               save = TRUE,
                               expr = {
                                 mlr::train(myFinalLearner, myFinalTask)
                               })


myFinalLearner <- classif.sda
myFinalModelTrained <- tryLoad(folder = outputFolder, 
                               filename = str_c("FinalTrainedLearner.", myFinalLearner$id, ".RDS"), 
                               save = TRUE,
                               expr = {
                                 mlr::train(myFinalLearner, myFinalTask)
                               })



# Other solutions, incl. stacked ensemble models, do not offer visibly better performance.




parallelMap::parallelStop()
#sink()

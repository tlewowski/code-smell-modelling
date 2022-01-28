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
argv$utilsSource = "/Users/lma/Documents/lm16/CQ/gitALL/ncbr-model-builder/src/R/prototypes/final/utilsLM.R"  
argv$ncbrModels = "./tmp/ncbr-models/models/" #"./ncbr-models/models/" #as should be
argv$targetPath = ""
argv$targetModel = "target.model"
argv$targetData = "target.data"
argv$targetPerf = "target.perf"
argv$evaluate = FALSE
argv$build = TRUE
argv$beta = 0.8
argv$outputFolder = "/Users/lma/Documents/lm16/CQ/gitALL/ncbr-model-builder/src/R/prototypes/"
argv$learner = "classif.xgboost" #"classif.kknn"
argv$version = "RC1.0"
argv$investigatedSmell = "blob" #"blob", "data class", "long method", "feature envy"


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
  cleanDataByRemovingIrrelevantData(investigatedSmell) %>%
  mlr::removeConstantFeatures(perc = 0.02,
                              dont.rm = "severity") %>%
  mlr::normalizeFeatures()


taskPerSmell <- makeClassifTask(id = investigatedSmell,
                                data = dataPerSmell,
                                target = "severity")

# compatible

sinkOn <- function() {
  sink(stringr::str_c(outputFolder, "_", investigatedSmell, "_", argv$version, "_results.txt"), split = TRUE, append = TRUE) #send output to both: file AND console 
}

#--MODELS

#-- classif.naiveBayes
tic("classif.naiveBayes")
classif.naiveBayes = myMakeLearner("classif.naiveBayes") 

classif.naiveBayes.resample.rep10cv10stratify = tryLoadResample(outputFolder, 
                                                                "classif.naiveBayes.resample.rep10cv10stratify.RDS", 
                                                                save = TRUE, 
                                                                learner = classif.naiveBayes, 
                                                                task = taskPerSmell, 
                                                                resampling = rep10cv10stratify,
                                                                measures = myMeasures)

sinkOn()
classif.naiveBayes.resample.rep10cv10stratify
sink()
# > classif.naiveBayes.resample.rep10cv10stratify
# Resample Result
# Task: blob
# Learner: classif.naiveBayes
# Aggr perf: multiclass.AvFbeta.test.mean=0.1828386
# Runtime: 4.83364
# -----------------------------------------------------------
# SIMPLE MODEL: classif.naiveBayes
# RESULT: multiclass.AvFbeta.test.mean=0.1828386
# -----------------------------------------------------------

classif.sda = makeLearner("classif.sda") 

classif.sda.NAI = mlr::makeImputeWrapper(
  learner = classif.sda,
  classes = list(
    integer = imputeLearner(makeLearner("regr.rpart")),
    numeric = imputeLearner(makeLearner("regr.rpart")),
    #takes care of double values!
    factor = imputeLearner(makeLearner("classif.rpart"))
  )
)

classif.sda.NAI.resample.rep10cv10stratify = tryLoadResample(outputFolder, 
                                                             "classif.sda.NAI.resample.rep10cv10stratify.RDS", 
                                                             save = TRUE, 
                                                             learner = classif.sda.NAI, 
                                                             task = taskPerSmell, 
                                                             resampling = rep10cv10stratify,
                                                             measures = myMeasures)

sinkOn()
classif.sda.NAI.resample.rep10cv10stratify
sink()
# > classif.sda.NAI.resample.rep10cv10stratify
# Resample Result
# Task: blob
# Learner: classif.sda.imputed
# Aggr perf: multiclass.AvFbeta.test.mean=0.1858955
# Runtime: 64.1486
# -----------------------------------------------------------
# SIMPLE MODEL classif.sda.NAI
# RESULT: multiclass.AvFbeta.test.mean=0.1858955 ... 0.2463149
# -----------------------------------------------------------


# Tuning of classif.sda.impute is computationally intesive and inefficitent - omitted


classif.lda = makeLearner("classif.lda") 

classif.lda.NAI = mlr::makeImputeWrapper(
  learner = classif.lda,
  classes = list(
    integer = imputeLearner(makeLearner("regr.rpart")),
    numeric = imputeLearner(makeLearner("regr.rpart")),
    #takes care of double values!
    factor = imputeLearner(makeLearner("classif.rpart"))
  )
)

classif.lda.NAI.resample.rep10cv10stratify = tryLoadResample(outputFolder, 
                                                             "classif.lda.NAI.resample.rep10cv10stratify.RDS", 
                                                             save = TRUE, 
                                                             learner = classif.lda.NAI, 
                                                             task = taskPerSmell, 
                                                             resampling = rep10cv10stratify,
                                                             measures = myMeasures)

sinkOn()
classif.lda.NAI.resample.rep10cv10stratify
sink()
# > classif.lda.NAI.resample.rep10cv10stratify
# Resample Result
# Task: blob
# Learner: classif.lda.imputed
# Aggr perf: multiclass.AvFbeta.test.mean=0.1411048
# Runtime: 32.0842



# Feature selection on classif.naiveBayes does not lead to serious improvement - omitted

######################### classif.naiveBayes.featureSelection

# classif.naiveBayes.selectFeatures.res <- tryLoad(outputFolder, "classif.naiveBayes.selectFeatures.res.RDS", save = TRUE, {
#   selectFeatures(learner = classif.naiveBayes,
#                  task = taskPerSmell,
#                  resampling = cv10stratify, #myCV, #cv10stratify,
#                  measures = multiclass.AvFbeta,
#                  control = makeFeatSelControlGA(maxit = 200), 
#                  show.info = FALSE
#   )
# }) #slow - takes 2580sec (maxit = 200, rep10cv10stratify)
# classif.naiveBayes.selectFeatures.res
# classif.naiveBayes.selectFeatures.res$x
# classif.naiveBayes.fs.dataPerSmell <- dataPerSmell[, c("severity", classif.naiveBayes.selectFeatures.res$x)]
# classif.naiveBayes.fs.taskPerSmell <- makeClassifTask(data = classif.naiveBayes.fs.dataPerSmell, target = "severity")
# #classif.naiveBayes.featureSelection.learner <- classif.naiveBayes.selectFeatures.res$learner #Learner that was optimized 
# classif.naiveBayes.selectFeatures.res$learner #Learner that was optimized 
# classif.naiveBayes.fs.learner <- setLearnerId(classif.naiveBayes.selectFeatures.res$learner, "classif.naiveBayes.fs.learner")
# classif.naiveBayes.fs.learner
# #wrapperModel <- train(selectedFeatures$learner, taskPerSmell.fs)
# 
# # > classif.naiveBayes.selectFeatures.res
# # FeatSel result:
# #   Features (21): abc_size, attributes, average_not_accessor_or_m...
# # multiclass.AvFbeta.test.mean=0.4183246
# # > classif.naiveBayes.selectFeatures.res$x
# # [1] "abc_size"                                          "attributes"                                       
# # [3] "average_not_accessor_or_mutator_method_complexity" "depth"                                            
# # [5] "instability"                                       "instance_variables"                               
# # [7] "lines_of_code"                                     "not_accessors_or_mutators_weighted"               
# # [9] "pmd_fanout"                                        "pmd_ncss"                                         
# # [11] "pmd_noa"                                           "pmd_noc"                                          
# # [13] "pmd_nocm"                                          "pmd_nopa"                                         
# # [15] "pmd_nopm"                                          "pmd_nopra"                                        
# # [17] "pmd_wmc"                                           "pmd_wmcnamm"                                      
# # [19] "pmd_woc"                                           "used_by"                                          
# # [21] "uses"    
# 
# 
# classif.naiveBayes.fs.learner.resample.rep10cv10stratify = tryLoadResample(
#   outputFolder,
#   "classif.naiveBayes.fs.learner.resample.rep10cv10stratify.RDS",
#   save = TRUE,
#   learner = classif.naiveBayes.fs.learner, #classif.naiveBayes.fs.learner,
#   task = taskPerSmell, #classif.naiveBayes.featureSelection.taskPerSmell,
#   resampling = rep10cv10stratify,
#   measures = myMeasures
# )
# classif.naiveBayes.fs.learner.resample.rep10cv10stratify
# # > classif.naiveBayes.fs.learner.resample.rep10cv10stratify
# # Resample Result
# # Task: blob
# # Learner: classif.naiveBayes.fs.learner
# # Aggr perf: multiclass.AvFbeta.test.mean=0.1828386
# # Runtime: 59.4121
# # 
# # classif.naiveBayes.fs.benchmark.rep10cv10stratify = tryLoadBenchmark(
# #   learner = classif.naiveBayes.fs.learner, #classif.naiveBayes.featureSelection,
# #   outputFolder = outputFolder,
# #   rdsFileName = "classif.naiveBayes.fs.benchmark.rep10cv10stratify.RDS",
# #   save = TRUE,
# #   task = taskPerSmell,
# #   resampling = rep10cv10stratify, #rep2cv10stratify, #rep10cv10stratify,
# #   measures = myMeasures
# # )
# # classif.naiveBayes.fs.benchmark.rep10cv10stratify

# # Train model: 
# # Best model: classif.naiveBayes.fs.learner 
# #classif.naiveBayes.featureSelection <- train(classif.naiveBayes.imputed.tuned.selectFeatures.res$learner, taskPerSmell)
# myFinalLearner <- classif.naiveBayes.fs.learner
# #myFinalLearner <- classif.naiveBayes.featureSelectionWrapper
# myFinalTask <- classif.naiveBayes.fs.taskPerSmell
# 
# #-- Training final model with tuned params
# myFinalModelTrained <- mlr::train(myFinalLearner, myFinalTask)


# Train model: nb
myFinalLearner <- classif.naiveBayes
#myFinalLearner <- classif.naiveBayes.featureSelectionWrapper
myFinalTask <- taskPerSmell
myFinalModelTrained <- tryLoad(folder = outputFolder, 
                               filename = str_c("FinalTrainedLearner.", myFinalLearner$id, ".RDS"), 
                               save = TRUE,
                               expr = {
                                 mlr::train(myFinalLearner, myFinalTask)
                               })


# Train model: sda
myFinalLearner <- classif.sda.NAI
#myFinalLearner <- classif.naiveBayes.featureSelectionWrapper
myFinalTask <- taskPerSmell

myFinalModelTrained <- tryLoad(folder = outputFolder, 
                               filename = str_c("FinalTrainedLearner.", myFinalLearner$id, ".RDS"), 
                               save = TRUE,
                               expr = {
                                 mlr::train(myFinalLearner, myFinalTask)
                               })




toc() #naiveBayes



#--ENSEMBLE MODEL
listOfBaseLearners <- list(
  "classif.boosting",          
  "classif.C50",              
  "classif.gbm",               
  "classif.naiveBayes",
  "classif.xgboost"
)

listOfNAILearners <- list(
  "classif.earth", #0.0953726 / 4s  rep10: 0.1409303 / 5s
  "classif.evtree", #0.0621404 /51s  rep10: 0.1039271 / 39s
  "classif.glmnet", # 0.0469049 / 7s  rep10: 0.0468901 / 7s
  "classif.ksvm", # 0.0129965 /4s  rep10:  0.0152405 /4s
  "classif.lda", #0.1494676 / 4s rep10: 0.1423402 / 4s
  "classif.LiblineaRL1LogReg", #0.1996670 / 4s  rep10: 0.1282306 / 4s
  "classif.LiblineaRL2LogReg", #0.2502140 / 4s  rep10: 0.1378776 / 4s
  "classif.multinom", # 0.2042833 / 3s   rep10: 0.1599420 / 5s
  "classif.randomForest", # 0.0636942 / 5s   rep10: 0.0596785 / 7s
  "classif.ranger", # 0.0650204 / 4s   rep10: 0.0640248 / 5s
  "classif.rpart", # 0.0538932 / 3s    rep10: 0.0577461 / 3s
  "classif.sda", # 0.1742918 / 3s    rep10: 0.2463149 / 3s
  "classif.sparseLDA", # 0.1369098 / 11s   rep10: 0.1323447 / 8s
  "classif.svm"#, # 0.0330674 / 4s    rep10: 0.0386330 / 4s
)


#numberOfLearners <- length(listOfBaseLearners)

#listOfBaseLearners

learners <- lapply(listOfBaseLearners, myMakeLearner)
learners <- lapply(learners, setPredictType, "prob")


learnersNAI <- lapply(listOfNAILearners, myMakeLearnerNAI)
learnersNAI <- lapply(learnersNAI, setPredictType, "prob")


learnersCombined <- append(learners, learnersNAI)


sinkOn()
#learners
#learnersNAI
learnersCombined
sink()

numberOfLearners <- length(learners)



ensemble1.learners <-
  mlr::makeStackedLearner(
    base.learners = learners,
    super.learner = classif.naiveBayes, 
    predict.type = "prob",
    method = "stack.cv",
    use.feat = TRUE
  ) %>%
  setLearnerId("ensemble1.learners") 

ensemble1.learners.resample.cv10stratify = tryLoadResample(
  outputFolder = outputFolder, 
  rdsFileName = stringr::str_c(ensemble1.learners$id, ".resample.cv10stratify.RDS"), 
  save = TRUE, 
  learner = ensemble1.learners,
  task = taskPerSmell, 
  resampling = cv10stratify,
  measures = myMeasures
)

sinkOn()
ensemble1.learners.resample.cv10stratify
sink()
# > ensemble1.learners.resample.cv10stratify
# Resample Result
# Task: blob
# Learner: ensemble1.learners
# Aggr perf: multiclass.AvFbeta.test.mean=0.1788586
# Runtime: 306.311
# -----------------------------------------------------------
# ENSEMBLE MODEL: eensemble1.learners
# RESULT: multiclass.AvFbeta.test.mean=0.1788586
# -----------------------------------------------------------


ensemble1.learnersNAI <-
  mlr::makeStackedLearner(
    base.learners = learnersNAI,
    super.learner = classif.naiveBayes, 
    predict.type = "prob",
    method = "stack.cv",
    use.feat = TRUE
  ) %>%
  setLearnerId("ensemble1.learnersNAI") 

ensemble1.learnersNAI.resample.cv10stratify = tryLoadResample(
  outputFolder = outputFolder, 
  rdsFileName = stringr::str_c(ensemble1.learnersNAI$id, ".resample.cv10stratify.RDS"), 
  save = TRUE, 
  learner = ensemble1.learnersNAI,
  task = taskPerSmell, 
  resampling = cv10stratify,
  measures = myMeasures
)

sinkOn()
ensemble1.learnersNAI.resample.cv10stratify
sink()
# > ensemble1.learnersNAI.resample.cv10stratify
# Resample Result
# Task: blob
# Learner: ensemble1.learnersNAI
# Aggr perf: multiclass.AvFbeta.test.mean=0.2213512
# Runtime: 842.646
# -----------------------------------------------------------
# ENSEMBLE MODEL: ensemble1.learnersNAI
# RESULT: multiclass.AvFbeta.test.mean=0.2213512
# -----------------------------------------------------------


ensembleSDA.learnersNAI <-
  mlr::makeStackedLearner(
    base.learners = learnersNAI,
    super.learner = classif.sda.NAI, 
    predict.type = "prob",
    method = "stack.cv",
    use.feat = TRUE
  ) %>%
  setLearnerId("ensembleSDA.learnersNAI") 

ensembleSDA.learnersNAI.resample.cv10stratify = tryLoadResample(
  outputFolder = outputFolder, 
  rdsFileName = stringr::str_c(ensembleSDA.learnersNAI$id, ".resample.cv10stratify.RDS"), 
  save = TRUE, 
  learner = ensembleSDA.learnersNAI,
  task = taskPerSmell, 
  resampling = cv10stratify,
  measures = myMeasures
)

sinkOn()
ensembleSDA.learnersNAI.resample.cv10stratify
sink()
# > ensembleSDA.learnersNAI.resample.cv10stratify
# Resample Result
# Task: blob
# Learner: ensembleSDA.learnersNAI
# Aggr perf: multiclass.AvFbeta.test.mean=0.1951513
# Runtime: 498.096

ensembleLDA.learnersNAI <-
  mlr::makeStackedLearner(
    base.learners = learnersNAI,
    super.learner = classif.lda.NAI, 
    predict.type = "prob",
    method = "stack.cv",
    use.feat = TRUE
  ) %>%
  setLearnerId("ensembleLDA.learnersNAI") 

ensembleLDA.learnersNAI.resample.cv10stratify = tryLoadResample(
  outputFolder = outputFolder, 
  rdsFileName = stringr::str_c(ensembleLDA.learnersNAI$id, ".resample.cv10stratify.RDS"), 
  save = TRUE, 
  learner = ensembleLDA.learnersNAI,
  task = taskPerSmell, 
  resampling = cv10stratify,
  measures = myMeasures
)

sinkOn()
ensembleLDA.learnersNAI.resample.cv10stratify
sink()
# > ensembleLDA.learnersNAI.resample.cv10stratify
# Resample Result
# Task: blob
# Learner: ensembleLDA.learnersNAI
# Aggr perf: multiclass.AvFbeta.test.mean=0.1217460
# Runtime: 450.835


ensemble1.learnersCombined <-
  mlr::makeStackedLearner(
    base.learners = learnersCombined,
    super.learner = classif.naiveBayes, 
    predict.type = "prob",
    method = "stack.cv",
    use.feat = TRUE
  ) %>%
  setLearnerId("ensemble1.learnersCombined") 

ensemble1.learnersCombined.resample.cv10stratify = tryLoadResample(
  outputFolder = outputFolder, 
  rdsFileName = stringr::str_c(ensemble1.learnersCombined$id, ".resample.cv10stratify.RDS"), 
  save = TRUE, 
  learner = ensemble1.learnersCombined,
  task = taskPerSmell, 
  resampling = cv10stratify,
  measures = myMeasures
)

sinkOn()
ensemble1.learnersCombined.resample.cv10stratify
sink()
# > ensemble1.learnersCombined.resample.cv10stratify
# Resample Result
# Task: blob
# Learner: ensemble1.learnersCombined
# Aggr perf: multiclass.AvFbeta.test.mean=0.2205904
# Runtime: 1377.22
# -----------------------------------------------------------
# ENSEMBLE MODEL: ensemble1.learnersCombined
# RESULT: multiclass.AvFbeta.test.mean=0.2205904
# -----------------------------------------------------------

sinkOn()
ensemble1.learners
ensemble1.learnersNAI
ensemble1.learnersCombined
sink()

# Train model: 
myFinalLearner <- ensemble1.learners
#myFinalLearner <- classif.naiveBayes.featureSelectionWrapper
myFinalTask <- taskPerSmell
#-- Training final model
myFinalModelTrained <- tryLoad(folder = outputFolder, 
                               filename = str_c("FinalTrainedLearner.", myFinalLearner$id, ".RDS"), 
                               save = TRUE,
                               expr = {
                                 mlr::train(myFinalLearner, myFinalTask)
                               })


#-- Training final model 
myFinalLearner <- ensemble1.learnersNAI
myFinalModelTrained <- tryLoad(folder = outputFolder, 
                               filename = str_c("FinalTrainedLearner.", myFinalLearner$id, ".RDS"), 
                               save = TRUE,
                               expr = {
                                 mlr::train(myFinalLearner, myFinalTask)
                               })


#-- Training final model 
myFinalLearner <- ensemble1.learnersCombined
myFinalModelTrained <- tryLoad(folder = outputFolder, 
                               filename = str_c("FinalTrainedLearner.", myFinalLearner$id, ".RDS"), 
                               save = TRUE,
                               expr = {
                                 mlr::train(myFinalLearner, myFinalTask)
                               })

myFinalTask <- taskPerSmell

myFinalLearner <- classif.naiveBayes
myFinalModelTrained <- tryLoad(folder = outputFolder,
                               filename = str_c("FinalTrainedLearner.", myFinalLearner$id, ".RDS"),
                               save = TRUE,
                               expr = {
                                 mlr::train(myFinalLearner, myFinalTask)
                               })

myFinalLearner <- classif.sda.NAI
myFinalModelTrained <- tryLoad(folder = outputFolder,
                               filename = str_c("FinalTrainedLearner.", myFinalLearner$id, ".RDS"),
                               save = TRUE,
                               expr = {
                                 mlr::train(myFinalLearner, myFinalTask)
                               })

myFinalLearner <- classif.lda.NAI
myFinalModelTrained <- tryLoad(folder = outputFolder,
                               filename = str_c("FinalTrainedLearner.", myFinalLearner$id, ".RDS"),
                               save = TRUE,
                               expr = {
                                 mlr::train(myFinalLearner, myFinalTask)
                               })

parallelMap::parallelStop()
#sink()

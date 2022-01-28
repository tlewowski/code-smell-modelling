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
#argv$utilsSource = "/Users/lma/Documents/lm16/CQ/gitALL/ncbr-model-builder/src/R/prototypes/final/blob/R1/utilsLM.R" #"/Users/lma/Documents/lm16/CQ/gitALL/ncbr-model-builder/src/R/prototypes/final/utilsLM_full.R" 
argv$utilsSource = "/Users/lma/Documents/lm16/CQ/gitALL/ncbr-model-builder/src/R/prototypes/final/utilsLM.R"  
#argv$utilsSource = "/Users/lma/Documents/lm16/CQ/gitALL/ncbr-model-builder/src/R/prototypes/final/utilsLM_full.R" 
#argv$dataSource = "../../final/2019-09-13-all-with-metrics.csv" # this is the final data set (2019-09-13)!
#argv$utilsSource = "../../utilsLM_full.R" 
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
argv$investigatedSmell = "long method" #"blob", "data class", "long method", "feature envy"


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


sinkOn <- function() {
  sink(stringr::str_c(outputFolder, "_", investigatedSmell, "_", argv$version, "_results.txt"), split = TRUE, append = TRUE) #send output to both: file AND console 
}

#--MODELS

classif.cforest = myMakeLearner("classif.cforest") 
classif.cforest.resample.rep10cv10stratify = tryLoadResample(outputFolder, 
                                                                "classif.cforest.resample.rep10cv10stratify.RDS", 
                                                                save = TRUE, 
                                                                learner = classif.cforest, 
                                                                task = taskPerSmell, 
                                                                resampling = rep10cv10stratify,
                                                                measures = myMeasures)
sinkOn()
classif.cforest.resample.rep10cv10stratify
sink()

#############

classif.ctree = myMakeLearner("classif.ctree") 
classif.ctree.resample.rep10cv10stratify = tryLoadResample(outputFolder, 
                                                             "classif.ctree.resample.rep10cv10stratify.RDS", 
                                                             save = TRUE, 
                                                             learner = classif.ctree, 
                                                             task = taskPerSmell, 
                                                             resampling = rep10cv10stratify,
                                                             measures = myMeasures)
sinkOn()
classif.ctree.resample.rep10cv10stratify
sink()

#############

classif.h2o.randomForest = myMakeLearner("classif.h2o.randomForest") 
classif.h2o.randomForest.resample.rep10cv10stratify = tryLoadResample(outputFolder, 
                                                             "classif.h2o.randomForest.resample.rep10cv10stratify.RDS", 
                                                             save = TRUE, 
                                                             learner = classif.h2o.randomForest, 
                                                             task = taskPerSmell, 
                                                             resampling = rep10cv10stratify,
                                                             measures = myMeasures)
sinkOn()
classif.h2o.randomForest.resample.rep10cv10stratify
sink()

#############

classif.randomForestSRC = myMakeLearner("classif.randomForestSRC") 
classif.randomForestSRC.resample.rep10cv10stratify = tryLoadResample(outputFolder, 
                                                             "classif.randomForestSRC.resample.rep10cv10stratify.RDS", 
                                                             save = TRUE, 
                                                             learner = classif.randomForestSRC, 
                                                             task = taskPerSmell, 
                                                             resampling = rep10cv10stratify,
                                                             measures = myMeasures)
sinkOn()
classif.randomForestSRC.resample.rep10cv10stratify
sink()

#############

classif.rpart = myMakeLearner("classif.rpart") 
classif.rpart.resample.rep10cv10stratify = tryLoadResample(outputFolder, 
                                                             "classif.rpart.resample.rep10cv10stratify.RDS", 
                                                             save = TRUE, 
                                                             learner = classif.rpart, 
                                                             task = taskPerSmell, 
                                                             resampling = rep10cv10stratify,
                                                             measures = myMeasures)
sinkOn()
classif.rpart.resample.rep10cv10stratify
sink()

#############

classif.evtree = myMakeLearner("classif.evtree") 
classif.evtree.NAI = mlr::makeImputeWrapper(
  learner = classif.evtree,
  classes = list(
    integer = imputeLearner(makeLearner("regr.rpart")),
    numeric = imputeLearner(makeLearner("regr.rpart")),
    #takes care of double values!
    factor = imputeLearner(makeLearner("classif.rpart"))
  )
)
classif.evtree.NAI.resample.rep10cv10stratify = tryLoadResample(outputFolder, 
                                                          "classif.evtree.NAI.resample.rep10cv10stratify.RDS", 
                                                          save = TRUE, 
                                                          learner = classif.evtree.NAI, 
                                                          task = taskPerSmell, 
                                                          resampling = rep10cv10stratify,
                                                          measures = myMeasures)
sinkOn()
classif.evtree.NAI.resample.rep10cv10stratify
sink()


#############

classif.kknn = myMakeLearner("classif.kknn") 
classif.kknn.NAI = mlr::makeImputeWrapper(
  learner = classif.kknn,
  classes = list(
    integer = imputeLearner(makeLearner("regr.rpart")),
    numeric = imputeLearner(makeLearner("regr.rpart")),
    #takes care of double values!
    factor = imputeLearner(makeLearner("classif.rpart"))
  )
)
classif.kknn.NAI.resample.rep10cv10stratify = tryLoadResample(outputFolder, 
                                                             "classif.kknn.NAI.resample.rep10cv10stratify.RDS", 
                                                             save = TRUE, 
                                                             learner = classif.kknn.NAI, 
                                                             task = taskPerSmell, 
                                                             resampling = rep10cv10stratify,
                                                             measures = myMeasures)
sinkOn()
classif.kknn.NAI.resample.rep10cv10stratify
sink()

#############

classif.randomForest = myMakeLearner("classif.randomForest") 
classif.randomForest.NAI = mlr::makeImputeWrapper(
  learner = classif.randomForest,
  classes = list(
    integer = imputeLearner(makeLearner("regr.rpart")),
    numeric = imputeLearner(makeLearner("regr.rpart")),
    #takes care of double values!
    factor = imputeLearner(makeLearner("classif.rpart"))
  )
)
classif.randomForest.NAI.resample.rep10cv10stratify = tryLoadResample(outputFolder, 
                                                             "classif.randomForest.NAI.resample.rep10cv10stratify.RDS", 
                                                             save = TRUE, 
                                                             learner = classif.randomForest.NAI, 
                                                             task = taskPerSmell, 
                                                             resampling = rep10cv10stratify,
                                                             measures = myMeasures)
sinkOn()
classif.randomForest.NAI.resample.rep10cv10stratify
sink()

#############

classif.ranger = myMakeLearner("classif.ranger") 
classif.ranger.NAI = mlr::makeImputeWrapper(
  learner = classif.ranger,
  classes = list(
    integer = imputeLearner(makeLearner("regr.rpart")),
    numeric = imputeLearner(makeLearner("regr.rpart")),
    #takes care of double values!
    factor = imputeLearner(makeLearner("classif.rpart"))
  )
)
classif.ranger.NAI.resample.rep10cv10stratify = tryLoadResample(outputFolder, 
                                                             "classif.ranger.NAI.resample.rep10cv10stratify.RDS", 
                                                             save = TRUE, 
                                                             learner = classif.ranger.NAI, 
                                                             task = taskPerSmell, 
                                                             resampling = rep10cv10stratify,
                                                             measures = myMeasures)
sinkOn()
classif.ranger.NAI.resample.rep10cv10stratify
sink()

#############
# Defined below / earlier
# classif.sparseLDA = myMakeLearner("classif.sparseLDA") 
# classif.sparseLDA.NAI = mlr::makeImputeWrapper(
#   learner = classif.sparseLDA,
#   classes = list(
#     integer = imputeLearner(makeLearner("regr.rpart")),
#     numeric = imputeLearner(makeLearner("regr.rpart")),
#     #takes care of double values!
#     factor = imputeLearner(makeLearner("classif.rpart"))
#   )
# )
# classif.sparseLDA.NAI.resample.rep10cv10stratify = tryLoadResample(outputFolder, 
#                                                              "classif.sparseLDA.NAI.resample.rep10cv10stratify.RDS", 
#                                                              save = TRUE, 
#                                                              learner = classif.sparseLDA.NAI, 
#                                                              task = taskPerSmell, 
#                                                              resampling = rep10cv10stratify,
#                                                              measures = myMeasures)
# sinkOn()
# classif.sparseLDA.NAI.resample.rep10cv10stratify
# sink()

#############




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

# > classif.naiveBayes.resample.rep10cv10stratify
# Resample Result
# Task: feature envy
# Learner: classif.naiveBayes
# Aggr perf: multiclass.AvFbeta.test.mean=0.1137879
# Runtime: 1.73435
# -----------------------------------------------------------
# Task: feature envy
# SIMPLE MODEL: classif.naiveBayes
# RESULT: multiclass.AvFbeta.test.mean=0.1137879
# -----------------------------------------------------------

# > classif.naiveBayes.resample.rep10cv10stratify
# Resample Result
# Task: data class
# Learner: classif.naiveBayes
# Aggr perf: multiclass.AvFbeta.test.mean=0.1726649
# Runtime: 4.33405

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
# SMELL: Blob
# SIMPLE MODEL classif.sda.NAI
# RESULT: multiclass.AvFbeta.test.mean=0.1858955 ... 0.2463149
# -----------------------------------------------------------
# > classif.sda.NAI.resample.rep10cv10stratify
# Resample Result
# Task: feature envy
# Learner: classif.sda.imputed
# Aggr perf: multiclass.AvFbeta.test.mean=0.2051642
# Runtime: 3.31096
# -----------------------------------------------------------
# SMELL: Feature Envy
# SIMPLE MODEL classif.sda.NAI
# RESULT: multiclass.AvFbeta.test.mean=0.2051642
# -----------------------------------------------------------

# > classif.sda.NAI.resample.rep10cv10stratify
# Resample Result
# Task: data class
# Learner: classif.sda.imputed
# Aggr perf: multiclass.AvFbeta.test.mean=0.3177816
# Runtime: 22.7981
# -----------------------------------------------------------
# SMELL: data class
# SIMPLE MODEL classif.sda.NAI
# RESULT: multiclass.AvFbeta.test.mean=0.3177816
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
# Task: feature envy
# Learner: classif.lda.imputed
# Aggr perf: multiclass.AvFbeta.test.mean=0.1956251
# Runtime: 3.25722
# -----------------------------------------------------------
# SMELL: Feature Envy
# SIMPLE MODEL classif.lda.NAI
# RESULT: multiclass.AvFbeta.test.mean=0.1956251
# -----------------------------------------------------------

# > classif.lda.NAI.resample.rep10cv10stratify
# Resample Result
# Task: data class
# Learner: classif.lda.imputed
# Aggr perf: multiclass.AvFbeta.test.mean=0.3272614
# Runtime: 21.8205
# -----------------------------------------------------------
# SMELL: data class
# SIMPLE MODEL classif.lda.NAI
# RESULT: multiclass.AvFbeta.test.mean=0.3272614
# -----------------------------------------------------------



classif.lda = makeLearner("classif.lda") 

classif.sparseLDA.NAI = mlr::makeImputeWrapper(
  learner = classif.lda,
  classes = list(
    integer = imputeLearner(makeLearner("regr.rpart")),
    numeric = imputeLearner(makeLearner("regr.rpart")),
    #takes care of double values!
    factor = imputeLearner(makeLearner("classif.rpart"))
  )
)

classif.sparseLDA.NAI.resample.rep10cv10stratify = tryLoadResample(outputFolder, 
                                                                   "classif.sparseLDA.NAI.resample.rep10cv10stratify.RDS", 
                                                                   save = TRUE, 
                                                                   learner = classif.sparseLDA.NAI, 
                                                                   task = taskPerSmell, 
                                                                   resampling = rep10cv10stratify,
                                                                   measures = myMeasures)

sinkOn()
classif.sparseLDA.NAI.resample.rep10cv10stratify
sink()


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


# Train model: lda
myFinalLearner <- classif.lda.NAI
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
  #"classif.boosting",          #35sec/cv10 slow-344/rep10cv10 Blob/..,: 0.0969944/0.0251969/0.2764924!
  "classif.C50",               #0.26 - 2                                0.1231158/0.0103388/0.2605380!
  "classif.cforest",
  "classif.ctree",
  "classif.gbm",               #0.3                                     0.1256403/0.0175553/0.2210092!
  "classif.h2o.gbm",           #7   ~slow - 29 (sometimes stops)        0.0873189/0.0242063/0.2741805!
  "classif.h2o.randomForest",  #7   ~slow                               0.0684085/0.0234511/0.2434570
  #"classif.J48",               # TOO SLOW  
  #"classif.JRip",              # TOO SLOW 
  "classif.naiveBayes",                     #                           0.1762174!!/0.1131065!!/0.1787817
  #"classif.OneR",              # TOO SLOW  
  #"classif.PART",              # TOO SLOW 
  "classif.randomForestSRC",    #10  ~slow                              0.0636282/0.0220929/0.2579931
  "classif.rpart",              #0.1                                    0.0711352/0.0247685/0.2298515
  "classif.xgboost",            #0.3                                    0.1084216/0.0663084/0.2613941
  "classif.xgboost.custom"#,     #0.152                                 0.1090508/0.0668097/0.2667305
)

listOfNAILearners <- list(
  "classif.earth", #0.0953726 / 4s  rep10: 0.1409303 / 5s             0.0917362/0.0935077/0.2081120
  "classif.evtree",
  #"classif.glmnet", # 0.0469049 / 7s  rep10: 0.0468901 / 7s           0.0439452/0.0183013/0.1650386
  "classif.kknn", #                                                   ........./........./
  "classif.lda", #0.1494676 / 4s rep10: 0.1423402 / 4s                0.1578773/0.2731291!!!/0.2957844
  #"classif.LiblineaRL1LogReg", #0.1996670 / 4s  rep10: 0.1282306 / 4s 0.1517479/0.0638610/0.2018130
  "classif.multinom", # 0.2042833 / 3s   rep10: 0.1599420 / 5s        0.1271467/0.0781992/0.2149153
  "classif.randomForest", # 0.0636942 / 5s   rep10: 0.0596785 / 7s    0.0697491/0.0146065/0.2796798
  "classif.ranger", # 0.0650204 / 4s   rep10: 0.0640248 / 5s          0.0603054/0.0270588/0.2111702
  #"classif.rda",  # 0.1883533 / 42s    rep10: 0.0579134 / 41s         0.1199243/0.1403843!/0.2723278 error: Column 'classif.rda.imputed.critical' contains NaN values.
  "classif.rpart", # 0.0538932 / 3s    rep10: 0.0577461 / 3s          0.0542638/0.0368093/0.1714130
  #"classif.RRF", #error
  "classif.sda", # 0.1742918 / 3s    rep10: 0.2463149 / 3s            0.1942894!!!/0.2259902!!!/0.3269700!!!
  "classif.sparseLDA" # 0.1369098 / 11s   rep10: 0.1323447 / 8s      0.1134550/0.2014830!!!/0.3194353!!!
  #  "classif.svm"#, # 0.0330674 / 4s    rep10: 0.0386330 / 4s           0.0361680/0.0068499/0.1214286
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




# ensemble1.learners <-
#   mlr::makeStackedLearner(
#     base.learners = learners,
#     super.learner = classif.naiveBayes, 
#     predict.type = "prob",
#     method = "stack.cv",
#     use.feat = TRUE,
#   ) %>%
#   setLearnerId("ensemble1.learners") 
# 
# ensemble1.learners.resample.cv10stratify = tryLoadResample(
#   outputFolder = outputFolder, 
#   rdsFileName = stringr::str_c(ensemble1.learners$id, ".resample.cv10stratify.RDS"), 
#   save = TRUE, 
#   learner = ensemble1.learners,
#   task = taskPerSmell, 
#   resampling = cv10stratify,
#   measures = myMeasures
# )
# 
# sinkOn()
# ensemble1.learners.resample.cv10stratify
# sink()
# > ensembleBlob1.learners.resample.cv10stratify
# Resample Result
# Task: blob
# Learner: ensembleBlob1.learners
# Aggr perf: multiclass.AvFbeta.test.mean=0.1788586
# Runtime: 306.311
# -----------------------------------------------------------
# SMELL: blob
# ENSEMBLE MODEL: eensembleBlob1.learners
# RESULT: multiclass.AvFbeta.test.mean=0.1788586
# -----------------------------------------------------------
# > ensemble1.learners.resample.cv10stratify
# Resample Result
# Task: feature envy
# Learner: ensemble1.learners
# Aggr perf: multiclass.AvFbeta.test.mean=0.1225141
# Runtime: 226.8
# -----------------------------------------------------------
# SMELL: blob
# ENSEMBLE MODEL: eensemble1.learners
# RESULT: multiclass.AvFbeta.test.mean=0.1225141
# -----------------------------------------------------------

# ensemble1NB2.learners <-
#   mlr::makeStackedLearner(
#     base.learners = learners,
#     super.learner = classif.naiveBayes, 
#     predict.type = "prob",
#     method = "stack.cv",
#     use.feat = TRUE,
#     resampling = cv10stratify
#   ) %>%
#   setLearnerId("ensemble1NB2.learners") 
# 
# ensemble1NB2.learners.resample.rep2cv10stratify = tryLoadResample(
#   outputFolder = outputFolder, 
#   rdsFileName = stringr::str_c(ensemble1.learners$id, ".resample.rep2cv10stratify.RDS"), 
#   save = TRUE, 
#   learner = ensemble1.learners,
#   task = taskPerSmell, 
#   resampling = rep2cv10stratify,
#   measures = myMeasures
# )
# 
# sinkOn()
# ensemble1NB2.learners.resample.rep2cv10stratify
# sink()
# > ensemble1NB2.learners.resample.rep2cv10stratify
# Resample Result
# Task: feature envy
# Learner: ensemble1.learners
# Aggr perf: multiclass.AvFbeta.test.mean=0.1085152
# Runtime: 510.143



# ensemble1.learnersNAI <-
#   mlr::makeStackedLearner(
#     base.learners = learnersNAI,
#     super.learner = classif.naiveBayes, 
#     predict.type = "prob",
#     method = "stack.cv",
#     use.feat = TRUE
#   ) %>%
#   setLearnerId("ensemble1.learnersNAI") 
# 
# ensemble1.learnersNAI.resample.cv10stratify = tryLoadResample(
#   outputFolder = outputFolder, 
#   rdsFileName = stringr::str_c(ensemble1.learnersNAI$id, ".resample.cv10stratify.RDS"), 
#   save = TRUE, 
#   learner = ensemble1.learnersNAI,
#   task = taskPerSmell, 
#   resampling = cv10stratify,
#   measures = myMeasures
# )
# 
# sinkOn()
# ensemble1.learnersNAI.resample.cv10stratify
# sink()
# > ensembleBlob1.learnersNAI.resample.cv10stratify
# Resample Result
# Task: blob
# Learner: ensembleBlob1.learnersNAI
# Aggr perf: multiclass.AvFbeta.test.mean=0.2213512
# Runtime: 842.646
# -----------------------------------------------------------
# SMELL: blob
# ENSEMBLE MODEL: ensembleBlob1.learnersNAI
# RESULT: multiclass.AvFbeta.test.mean=0.2213512
# -----------------------------------------------------------

# > ensemble1.learnersNAI.resample.cv10stratify
# Resample Result
# Task: feature envy
# Learner: ensemble1.learnersNAI
# Aggr perf: multiclass.AvFbeta.test.mean=0.1051304
# Runtime: 119.46
# -----------------------------------------------------------
# SMELL: feature envy
# ENSEMBLE MODEL: ensemble1.learnersNAI
# RESULT: multiclass.AvFbeta.test.mean=0.1051304
# -----------------------------------------------------------



# ensembleSDA.learnersNAI <-
#   mlr::makeStackedLearner(
#     base.learners = learnersNAI,
#     super.learner = classif.sda.NAI, 
#     predict.type = "prob",
#     method = "stack.cv",
#     use.feat = TRUE,
#     resampling = cv10stratify
#   ) %>%
#   setLearnerId("ensembleSDA.learnersNAI") 
# 
# ensembleSDA.learnersNAI.resample.cv10stratify = tryLoadResample(
#   outputFolder = outputFolder, 
#   rdsFileName = stringr::str_c(ensembleSDA.learnersNAI$id, ".resample.cv10stratify.RDS"), 
#   save = TRUE, 
#   learner = ensembleSDA.learnersNAI,
#   task = taskPerSmell, 
#   resampling = cv10stratify,
#   measures = myMeasures
# )
# 
# sinkOn()
# ensembleSDA.learnersNAI.resample.cv10stratify
# sink()
# > ensembleSDA.learnersNAI.resample.cv10stratify
# Resample Result
# Task: feature envy
# Learner: ensembleSDA.learnersNAI
# Aggr perf: multiclass.AvFbeta.test.mean=0.2097357
# Runtime: 123.924
# -----------------------------------------------------------
# SMELL: feature envy
# ENSEMBLE MODEL: ensembleSDA.learnersNAI
# RESULT: multiclass.AvFbeta.test.mean=0.2097357
# -----------------------------------------------------------

# > ensembleSDA.learnersNAI.resample.cv10stratify
# Resample Result
# Task: data class
# Learner: ensembleSDA.learnersNAI
# Aggr perf: multiclass.AvFbeta.test.mean=0.3707338
# Runtime: 502.559


ensembleLDA.learnersNAI <-
  mlr::makeStackedLearner(
    base.learners = learnersNAI,
    super.learner = classif.lda.NAI, 
    predict.type = "prob",
    method = "stack.cv",
    use.feat = TRUE,
    resampling = cv10stratify
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
# Task: feature envy
# Learner: ensembleLDA.learnersNAI
# Aggr perf: multiclass.AvFbeta.test.mean=0.1083060
# Runtime: 129.781
# -----------------------------------------------------------
# SMELL: feature envy
# ENSEMBLE MODEL: ensembleLDA.learnersNAI
# RESULT: multiclass.AvFbeta.test.mean=0.1083060
# -----------------------------------------------------------


# > ensembleLDA.learnersNAI.resample.cv10stratify
# Resample Result
# Task: data class
# Learner: ensembleLDA.learnersNAI
# Aggr perf: multiclass.AvFbeta.test.mean=0.4193077
# Runtime: 515.883
# -----------------------------------------------------------
# Task: data class
# ENSEMBLE MODEL: ensembleLDA.learnersNAI
# RESULT: multiclass.AvFbeta.test.mean=0.4193077
# -----------------------------------------------------------

# > ensembleLDA.learnersNAI.resample.cv10stratify
# Resample Result
# Task: long method
# Learner: ensembleLDA.learnersNAI
# Aggr perf: multiclass.AvFbeta.test.mean=0.5499986
# Runtime: 456.778
# -----------------------------------------------------------
# Task: long method
# ENSEMBLE MODEL: ensembleLDA.learnersNAI
# RESULT: multiclass.AvFbeta.test.mean=0.5499986
# -----------------------------------------------------------

ensembleLDA.learners <-
  mlr::makeStackedLearner(
    base.learners = learners,
    super.learner = classif.lda.NAI, 
    predict.type = "prob",
    method = "stack.cv",
    use.feat = TRUE,
    resampling = cv10stratify
  ) %>%
  setLearnerId("ensembleLDA.learners") 
ensembleLDA.learners.resample.cv10stratify = tryLoadResample(
  outputFolder = outputFolder, 
  rdsFileName = stringr::str_c(ensembleLDA.learners$id, ".resample.cv10stratify.RDS"), 
  save = TRUE, 
  learner = ensembleLDA.learners,
  task = taskPerSmell, 
  resampling = cv10stratify,
  measures = myMeasures
)
sinkOn()
ensembleLDA.learners.resample.cv10stratify
sink()
# > ensembleLDA.learners.resample.cv10stratify
# Resample Result
# Task: long method
# Learner: ensembleLDA.learners
# Aggr perf: multiclass.AvFbeta.test.mean=0.4497088
# Runtime: 470.212


ensemble.sparseLDA.learnersNAI <-
  mlr::makeStackedLearner(
    base.learners = learnersNAI,
    super.learner = classif.sparseLDA.NAI, 
    predict.type = "prob",
    method = "stack.cv",
    use.feat = TRUE,
    resampling = cv10stratify
  ) %>%
  setLearnerId("ensemble.sparseLDA.learnersNAI") 
ensemble.sparseLDA.learnersNAI.resample.cv10stratify = tryLoadResample(
  outputFolder = outputFolder, 
  rdsFileName = stringr::str_c(ensemble.sparseLDA.learnersNAI$id, ".resample.cv10stratify.RDS"), 
  save = TRUE, 
  learner = ensemble.sparseLDA.learnersNAI,
  task = taskPerSmell, 
  resampling = cv10stratify,
  measures = myMeasures
)
sinkOn()
ensemble.sparseLDA.learnersNAI.resample.cv10stratify
sink()
# > ensemble.sparseLDA.learnersNAI.resample.cv10stratify
# Resample Result
# Task: data class
# Learner: ensemble.sparseLDA.learnersNAI
# Aggr perf: multiclass.AvFbeta.test.mean=0.3235317
# Runtime: 454.043
# -----------------------------------------------------------
# Task: data class
# ENSEMBLE MODEL: ensemble.sparseLDA.learnersNAI
# RESULT: multiclass.AvFbeta.test.mean=0.3235317
# -----------------------------------------------------------

# > ensemble.sparseLDA.learnersNAI.resample.cv10stratify
# Resample Result
# Task: long method
# Learner: ensemble.sparseLDA.learnersNAI
# Aggr perf: multiclass.AvFbeta.test.mean=0.5147663
# Runtime: 431.002




# ensemble1.learnersCombined <-
#   mlr::makeStackedLearner(
#     base.learners = learnersCombined,
#     super.learner = classif.naiveBayes, 
#     predict.type = "prob",
#     method = "stack.cv",
#     use.feat = TRUE
#   ) %>%
#   setLearnerId("ensemble1.learnersCombined") 
# 
# ensemble1.learnersCombined.resample.cv10stratify = tryLoadResample(
#   outputFolder = outputFolder, 
#   rdsFileName = stringr::str_c(ensemble1.learnersCombined$id, ".resample.cv10stratify.RDS"), 
#   save = TRUE, 
#   learner = ensemble1.learnersCombined,
#   task = taskPerSmell, 
#   resampling = cv10stratify,
#   measures = myMeasures
# )
# 
# sinkOn()
# ensemble1.learnersCombined.resample.cv10stratify
# sink()
# > ensembleBlob1.learnersCombined.resample.cv10stratify
# Resample Result
# Task: blob
# Learner: ensembleBlob1.learnersCombined
# Aggr perf: multiclass.AvFbeta.test.mean=0.2205904
# Runtime: 1377.22
# -----------------------------------------------------------
# ENSEMBLE MODEL: ensembleBlob1.learnersCombined
# RESULT: multiclass.AvFbeta.test.mean=0.2205904
# -----------------------------------------------------------
# > ensemble1.learnersCombined.resample.cv10stratify
# Resample Result
# Task: feature envy
# Learner: ensemble1.learnersCombined
# Aggr perf: multiclass.AvFbeta.test.mean=0.1132465
# Runtime: 403.694



#########
ensemble.cforest.learners <-
  mlr::makeStackedLearner(
    base.learners = learners,
    super.learner = classif.cforest, 
    predict.type = "prob",
    method = "stack.cv",
    use.feat = TRUE,
    resampling = cv10stratify
  ) %>%
  setLearnerId("ensemble.cforest.learners") 
ensemble.cforest.learners.resample.cv10stratify = tryLoadResample(
  outputFolder = outputFolder, 
  rdsFileName = stringr::str_c(ensemble.cforest.learners$id, ".resample.cv10stratify.RDS"), 
  save = TRUE, 
  learner = ensemble.cforest.learners,
  task = taskPerSmell, 
  resampling = cv10stratify,
  measures = myMeasures
)
sinkOn()
ensemble.cforest.learners.resample.cv10stratify
sink()
# > ensemble.cforest.learners.resample.cv10stratify
# Resample Result
# Task: long method
# Learner: ensemble.cforest.learners
# Aggr perf: multiclass.AvFbeta.test.mean=0.6289960
# Runtime: 467.484
# -----------------------------------------------------------
# Task: long method
# ENSEMBLE MODEL: ensemble.cforest.learners
# RESULT: multiclass.AvFbeta.test.mean=0.6289960
# -----------------------------------------------------------


#########
ensemble.cforest.learnersCombined <-
  mlr::makeStackedLearner(
    base.learners = learnersCombined,
    super.learner = classif.cforest, 
    predict.type = "prob",
    method = "stack.cv",
    use.feat = TRUE,
    resampling = cv10stratify
  ) %>%
  setLearnerId("ensemble.cforest.learnersCombined") 
ensemble.cforest.learnersCombined.resample.cv10stratify = tryLoadResample(
  outputFolder = outputFolder, 
  rdsFileName = stringr::str_c(ensemble.cforest.learnersCombined$id, ".resample.cv10stratify.RDS"), 
  save = TRUE, 
  learner = ensemble.cforest.learnersCombined,
  task = taskPerSmell, 
  resampling = cv10stratify,
  measures = myMeasures
)
sinkOn()
ensemble.cforest.learnersCombined.resample.cv10stratify
sink()
# > ensemble.cforest.learnersCombined.resample.cv10stratify
# Resample Result
# Task: long method
# Learner: ensemble.cforest.learnersCombined
# Aggr perf: multiclass.AvFbeta.test.mean=0.5610741
# Runtime: 919.012
# -----------------------------------------------------------
# Task: long method
# ENSEMBLE MODEL: ensemble.cforest.learnersCombined
# RESULT: multiclass.AvFbeta.test.mean=0.5610741
# -----------------------------------------------------------



#########
ensemble.cforest.learnersNAI <-
  mlr::makeStackedLearner(
    base.learners = learnersNAI,
    super.learner = classif.cforest, 
    predict.type = "prob",
    method = "stack.cv",
    use.feat = TRUE,
    resampling = cv10stratify
  ) %>%
  setLearnerId("ensemble.cforest.learnersNAI") 
ensemble.cforest.learnersNAI.resample.cv10stratify = tryLoadResample(
  outputFolder = outputFolder, 
  rdsFileName = stringr::str_c(ensemble.cforest.learnersNAI$id, ".resample.cv10stratify.RDS"), 
  save = TRUE, 
  learner = ensemble.cforest.learnersNAI,
  task = taskPerSmell, 
  resampling = cv10stratify,
  measures = myMeasures
)
sinkOn()
ensemble.cforest.learnersNAI.resample.cv10stratify
sink()
# > ensemble.cforest.learnersNAI.resample.cv10stratify
# Resample Result
# Task: long method
# Learner: ensemble.cforest.learnersNAI
# Aggr perf: multiclass.AvFbeta.test.mean=0.5849071
# Runtime: 530.302
# -----------------------------------------------------------
# Task: long method
# ENSEMBLE MODEL: ensemble.cforest.learnersNAI
# RESULT: multiclass.AvFbeta.test.mean=0.5849071
# -----------------------------------------------------------


#########
ensemble.ctree.learners <-
  mlr::makeStackedLearner(
    base.learners = learners,
    super.learner = classif.ctree, 
    predict.type = "prob",
    method = "stack.cv",
    use.feat = TRUE,
    resampling = cv10stratify
  ) %>%
  setLearnerId("ensemble.ctree.learners") 
ensemble.ctree.learners.resample.cv10stratify = tryLoadResample(
  outputFolder = outputFolder, 
  rdsFileName = stringr::str_c(ensemble.ctree.learners$id, ".resample.cv10stratify.RDS"), 
  save = TRUE, 
  learner = ensemble.ctree.learners,
  task = taskPerSmell, 
  resampling = cv10stratify,
  measures = myMeasures
)
sinkOn()
ensemble.ctree.learners.resample.cv10stratify
sink()
# > ensemble.ctree.learners.resample.cv10stratify
# Resample Result
# Task: long method
# Learner: ensemble.ctree.learners
# Aggr perf: multiclass.AvFbeta.test.mean=0.5506673
# Runtime: 448.455
# -----------------------------------------------------------
# Task: long method
# ENSEMBLE MODEL: ensemble.ctree.learners
# RESULT: multiclass.AvFbeta.test.mean=0.5506673
# -----------------------------------------------------------



#########
ensemble.h2o.randomForest.learners <-
  mlr::makeStackedLearner(
    base.learners = learners,
    super.learner = classif.h2o.randomForest, 
    predict.type = "prob",
    method = "stack.cv",
    use.feat = TRUE,
    resampling = cv10stratify
  ) %>%
  setLearnerId("ensemble.h2o.randomForest.learners") 
ensemble.h2o.randomForest.learners.resample.cv10stratify = tryLoadResample(
  outputFolder = outputFolder, 
  rdsFileName = stringr::str_c(ensemble.h2o.randomForest.learners$id, ".resample.cv10stratify.RDS"), 
  save = TRUE, 
  learner = ensemble.h2o.randomForest.learners,
  task = taskPerSmell, 
  resampling = cv10stratify,
  measures = myMeasures
)
sinkOn()
ensemble.h2o.randomForest.learners.resample.cv10stratify
sink()
# > ensemble.h2o.randomForest.learners.resample.cv10stratify
# Resample Result
# Task: long method
# Learner: ensemble.h2o.randomForest.learners
# Aggr perf: multiclass.AvFbeta.test.mean=0.5377981
# Runtime: 465.325
# -----------------------------------------------------------
# Task: long method
# ENSEMBLE MODEL: ensemble.h2o.randomForest.learners
# RESULT: multiclass.AvFbeta.test.mean=0.5377981
# -----------------------------------------------------------



#########
ensemble.rpart.learners <-
  mlr::makeStackedLearner(
    base.learners = learners,
    super.learner = classif.rpart, 
    predict.type = "prob",
    method = "stack.cv",
    use.feat = TRUE,
    resampling = cv10stratify
  ) %>%
  setLearnerId("ensemble.rpart.learners") 
ensemble.rpart.learners.resample.cv10stratify = tryLoadResample(
  outputFolder = outputFolder, 
  rdsFileName = stringr::str_c(ensemble.rpart.learners$id, ".resample.cv10stratify.RDS"), 
  save = TRUE, 
  learner = ensemble.rpart.learners,
  task = taskPerSmell, 
  resampling = cv10stratify,
  measures = myMeasures
)
sinkOn()
ensemble.rpart.learners.resample.cv10stratify
sink()
# > ensemble.rpart.learners.resample.cv10stratify
# Resample Result
# Task: long method
# Learner: ensemble.rpart.learners
# Aggr perf: multiclass.AvFbeta.test.mean=0.4748965
# Runtime: 456.329
# -----------------------------------------------------------
# Task: long method
# ENSEMBLE MODEL: ensemble.rpart.learners
# RESULT: multiclass.AvFbeta.test.mean=0.4748965
# -----------------------------------------------------------



##########################################################

ensemble.ranger.learnersNAI <-
  mlr::makeStackedLearner(
    base.learners = learnersNAI,
    super.learner = classif.ranger.NAI, 
    predict.type = "prob",
    method = "stack.cv",
    use.feat = TRUE,
    resampling = cv10stratify
  ) %>%
  setLearnerId("ensemble.ranger.learnersNAI") 
ensemble.ranger.learnersNAI.resample.cv10stratify = tryLoadResample(
  outputFolder = outputFolder, 
  rdsFileName = stringr::str_c(ensemble.ranger.learnersNAI$id, ".resample.cv10stratify.RDS"), 
  save = TRUE, 
  learner = ensemble.ranger.learnersNAI,
  task = taskPerSmell, 
  resampling = cv10stratify,
  measures = myMeasures
)
sinkOn()
ensemble.ranger.learnersNAI.resample.cv10stratify
sink()
# > ensemble.ranger.learnersNAI.resample.cv10stratify
# Resample Result
# Task: long method
# Learner: ensemble.ranger.learnersNAI
# Aggr perf: multiclass.AvFbeta.test.mean=0.4925440
# Runtime: 510.002
# -----------------------------------------------------------
# Task: long method
# ENSEMBLE MODEL: ensemble.ranger.learnersNAI
# RESULT: multiclass.AvFbeta.test.mean=0.4925440
# -----------------------------------------------------------



ensemble.randomForest.learnersNAI <-
  mlr::makeStackedLearner(
    base.learners = learnersNAI,
    super.learner = classif.randomForest.NAI, 
    predict.type = "prob",
    method = "stack.cv",
    use.feat = TRUE,
    resampling = cv10stratify
  ) %>%
  setLearnerId("ensemble.randomForest.learnersNAI") 
ensemble.randomForest.learnersNAI.resample.cv10stratify = tryLoadResample(
  outputFolder = outputFolder, 
  rdsFileName = stringr::str_c(ensemble.randomForest.learnersNAI$id, ".resample.cv10stratify.RDS"), 
  save = TRUE, 
  learner = ensemble.randomForest.learnersNAI,
  task = taskPerSmell, 
  resampling = cv10stratify,
  measures = myMeasures
)
sinkOn()
ensemble.randomForest.learnersNAI.resample.cv10stratify
sink()


# defined earlier:
# ensemble.sparseLDA.learnersNAI <-
#   mlr::makeStackedLearner(
#     base.learners = learnersNAI,
#     super.learner = classif.sparseLDA, 
#     predict.type = "prob",
#     method = "stack.cv",
#     use.feat = TRUE,
#     resampling = cv10stratify
#   ) %>%
#   setLearnerId("ensemble.sparseLDA.learnersNAI") 
# ensemble.sparseLDA.learnersNAI.resample.cv10stratify = tryLoadResample(
#   outputFolder = outputFolder, 
#   rdsFileName = stringr::str_c(ensemble.sparseLDA.learnersNAI$id, ".resample.cv10stratify.RDS"), 
#   save = TRUE, 
#   learner = ensemble.sparseLDA.learnersNAI,
#   task = taskPerSmell, 
#   resampling = cv10stratify,
#   measures = myMeasures
# )
# sinkOn()
# ensemble.sparseLDA.learnersNAI.resample.cv10stratify
# sink()




#########
ensemble.evtree.learnersNAI <-
  mlr::makeStackedLearner(
    base.learners = learnersNAI,
    super.learner = classif.evtree.NAI, 
    predict.type = "prob",
    method = "stack.cv",
    use.feat = TRUE,
    resampling = cv10stratify
  ) %>%
  setLearnerId("ensemble.evtree.learnersNAI") 
ensemble.evtree.learnersNAI.resample.cv10stratify = tryLoadResample(
  outputFolder = outputFolder, 
  rdsFileName = stringr::str_c(ensemble.evtree.learnersNAI$id, ".resample.cv10stratify.RDS"), 
  save = TRUE, 
  learner = ensemble.evtree.learnersNAI,
  task = taskPerSmell, 
  resampling = cv10stratify,
  measures = myMeasures
)
sinkOn()
ensemble.evtree.learnersNAI.resample.cv10stratify
sink()
# > ensemble.evtree.learnersNAI.resample.cv10stratify
# Resample Result
# Task: long method
# Learner: ensemble.evtree.learnersNAI
# Aggr perf: multiclass.AvFbeta.test.mean=0.5675672
# Runtime: 599.353



# ensemble1.SDA.learnersCombined <-
#   mlr::makeStackedLearner(
#     base.learners = learnersCombined,
#     super.learner = classif.sda.NAI, 
#     predict.type = "prob",
#     method = "stack.cv",
#     use.feat = TRUE
#   ) %>%
#   setLearnerId("ensemble1.SDA.learnersCombined") 
# 
# ensemble1.SDA.learnersCombined.resample.cv10stratify = tryLoadResample(
#   outputFolder = outputFolder, 
#   rdsFileName = stringr::str_c(ensemble1.SDA.learnersCombined$id, ".resample.cv10stratify.RDS"), 
#   save = TRUE, 
#   learner = ensemble1.SDA.learnersCombined,
#   task = taskPerSmell, 
#   resampling = cv10stratify,
#   measures = myMeasures
# )
# 
# sinkOn()
# ensemble1.SDA.learnersCombined.resample.cv10stratify
# sink()
# > ensemble1.SDA.learnersCombined.resample.cv10stratify
# Resample Result
# Task: data class
# Learner: ensemble1.SDA.learnersCombined
# Aggr perf: multiclass.AvFbeta.test.mean=0.3183277
# Runtime: 768.45
# -----------------------------------------------------------
# Task: data class
# ENSEMBLE MODEL: ensemble1.SDA.learnersCombined
# RESULT: multiclass.AvFbeta.test.mean=0.3183277
# -----------------------------------------------------------


sinkOn()
ensemble1.learners
ensemble1.learnersNAI
ensemble1.learnersCombined
sink()

# # Train model: only models selected to use
# myFinalLearner <- ensemble1.learners
# #myFinalLearner <- classif.naiveBayes.featureSelectionWrapper
# myFinalTask <- taskPerSmell
# #-- Training final model
# myFinalModelTrained <- tryLoad(folder = outputFolder, 
#                                filename = str_c("FinalTrainedLearner.", myFinalLearner$id, ".RDS"), 
#                                save = TRUE,
#                                expr = {
#                                  mlr::train(myFinalLearner, myFinalTask)
#                                })
# 
# 
# #-- Training final model 
# myFinalLearner <- ensemble1.learnersNAI
# myFinalModelTrained <- tryLoad(folder = outputFolder, 
#                                filename = str_c("FinalTrainedLearner.", myFinalLearner$id, ".RDS"), 
#                                save = TRUE,
#                                expr = {
#                                  mlr::train(myFinalLearner, myFinalTask)
#                                })
# 
# 
# #-- Training final model 
# myFinalLearner <- ensemble1.SDA.learnersCombined
# myFinalModelTrained <- tryLoad(folder = outputFolder, 
#                                filename = str_c("FinalTrainedLearner.", myFinalLearner$id, ".RDS"), 
#                                save = TRUE,
#                                expr = {
#                                  mlr::train(myFinalLearner, myFinalTask)
#                                })

#-- Training final model
myFinalTask <- taskPerSmell

myFinalLearner <- classif.cforest
myFinalModelTrained <- tryLoad(folder = outputFolder,
                               filename = str_c("FinalTrainedLearner.", myFinalLearner$id, ".RDS"),
                               save = TRUE,
                               expr = {
                                 mlr::train(myFinalLearner, myFinalTask)
                               })

myFinalLearner <- classif.ranger.NAI
myFinalModelTrained <- tryLoad(folder = outputFolder,
                               filename = str_c("FinalTrainedLearner.", myFinalLearner$id, ".RDS"),
                               save = TRUE,
                               expr = {
                                 mlr::train(myFinalLearner, myFinalTask)
                               })

myFinalLearner <- ensemble.cforest.learners
myFinalModelTrained <- tryLoad(folder = outputFolder,
                               filename = str_c("FinalTrainedLearner.", myFinalLearner$id, ".RDS"),
                               save = TRUE,
                               expr = {
                                 mlr::train(myFinalLearner, myFinalTask)
                               })

parallelMap::parallelStop()
#sink()

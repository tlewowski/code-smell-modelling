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

#-- Lech Madeyski

#-- (Command line) Attributes -------------------------------------------------
argv <- commandArgs()
argv$date = Sys.Date() # "2019-06-06"
argv$dataSource = "2019-09-05-all-with-metrics.csv" #"all-with-metrics.csv"
argv$ncbrModels = "./tmp/ncbr-models/models/" #"./ncbr-models/models/" #as should be
argv$targetPath = ""
argv$targetModel = "target.model"
argv$targetData = "target.data"
argv$targetPerf = "target.perf"
argv$evaluate = FALSE
argv$build = TRUE
argv$beta = 0.8
argv$outputFolderRoot = "outputFolder/"
argv$learner = "classif.kknn"
argv$version = "1"
argv$investigatedSmell = "long method" #"blob", "data class", "long method", "feature envy"

investigatedSmell <- argv$investigatedSmell

outputFolder = stringr::str_c(argv$outputFolderRoot, argv$date, "/", argv$learner, "/", argv$investigatedSmell, "-", argv$version, "/")
if (!file.exists(outputFolder))
  dir.create(outputFolder, recursive = TRUE)


#-- Utils:
source("utilsLM.R")

#-- Set seed ------------------------------------------------------------------
set.seed(1234)


#--  LOAD DATA ----------------------------------------------------------------
importedData <- readr::read_csv(file = argv$dataSource) # return a tibble

myMeasures = list(
  multiclass.AvFbeta # new performance metric defined in the R4PerformanceMetricV2.1.pdf report by Madeyski
)

myCV <- cv10stratify # for more precise results change into repeated CV (e.g., rep20cv10stratify)


#-- PREPROCESS DATA PER SMELL
dataPerSmell <- importedData

dataPerSmell <-
  dataPerSmell %>% cleanDataByConvertingToCorrectTypes()

dataPerSmell <-
  dataPerSmell %>% cleanDataByTakingCareOfMultipleReviewsOfTheSameSample()

dataPerSmell <-
  dataPerSmell %>% cleanDataByRemovingIrrelevantData(investigatedSmell)


#dataPerSmell <- dataPerSmell[,-nearZeroVar(dataPerSmell)]
# remove all features where fraction of values differing
# from mode value is <= 2% #1%
dataPerSmell <- mlr::removeConstantFeatures(dataPerSmell, 
                                            perc = 0.02,
                                            dont.rm = "severity")



myLearner <- myMakeLearner(argv$learner) #("classif.kknn") 

if("missings" %in% getLearnerProperties(myLearner$id)){
  taskPerSmell <- makeClassifTask(id = investigatedSmell,
                                  data = dataPerSmell, 
                                  target = "severity")
} else{
  dataPerSmell <- dataPerSmell %>% cleanDataByMLbasedImputingNA(investigatedSmell, createDummyFeatures = FALSE)
  taskPerSmell <- makeClassifTask(id = stringr::str_c(investigatedSmell, "NAimputedViaML"),
                                  data = dataPerSmell, 
                                  target = "severity")
}
print(getTaskFeatureNames(taskPerSmell))
print(taskPerSmell)
print(myLearner)
print(myCV)
print(myMeasures)

#-- Performing (preliminary) repeated k-fold cross-validation ----
resampleResult <- resample(learner = myLearner, task = taskPerSmell, 
                           resampling = myCV, measures = myMeasures)
resampleResult$aggr
resampleResult$measures.test
print(calculateConfusionMatrix(resampleResult$pred, relative = TRUE))




#-- Hyperparameter tuning of K ----
paramSpace <- makeParamSet(makeDiscreteParam("k", values = 1:50))

searchStrategy <- makeTuneControlGrid()

tunedParams <-  
  tuneParams(myLearner, task = taskPerSmell, 
             resampling = myCV, 
             measures = myMeasures, 
             par.set = paramSpace, 
             control = searchStrategy)



#-- Training final model with tuned params (e.g., K in knn)
myLearnerTuned <- setHyperPars(myLearner, par.vals = tunedParams$x)
myModelTuned <- train(myLearnerTuned, taskPerSmell)

modelRDSFile <- str_c(outputFolder, myLearner$id, ".RDS")
saveRDS(myModelTuned, file = modelRDSFile)



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

library(getopt)
library('R.utils')
library(jsonlite)
library(rjson)
library(caret)
library(ModelMetrics)
library(dplyr)
library(mlr)
mlr::configureMlr()
library(stringr)
# names(argsL) <- argsDF$V1

args <- commandArgs(trailingOnly=TRUE)
if(is.null(args[1])) {
  iterations <- 20
} else {
  iterations <- as.numeric(args[1])
}

if(is.null(args[2])) {
  startIteration <- 0
} else {
  startIteration <- as.numeric(args[2])
}

print(paste("Running reproduction for", as.character(iterations), "iterations starting from", as.character(startIteration)))

argv <- list(beta=0.8)
source("./utils-repro.R")

dataSource <- "../../data/mlcq-with-metrics.csv"
modelsRoot <- "../../models/2022-04-01-repro-prec"
mkdirs(modelsRoot)

initialData <- read.csv(file=dataSource)
smells <- list(
#  list(schema="../../schemas/schema-functions-v2.json", name="long method", id="longmethod"),
#  list(schema="../../schemas/schema-functions-v2.json", name="feature envy", id="featureenvy")#,
  list(schema="../../schemas/schema-classes-v2.json", name="blob", id="blob"),
  list(schema="../../schemas/schema-classes-v2.json", name="data class", id="dataclass")
)

commonPath <- "./algorithms"
models <- list(
    #list(path="analytic/MDA_MLR.R", name="MDA"),
    #list(path="knn/KNN.R", name="KNN"),
    #list(path="meta/AdaBoost_MLR.R", name="adaboost"),
    #list(path="neural/NeuralNet.R", name="neural_net"),
    #list(path="SVM/SVM_kSVM.R", name="kSVM"),
    #list(path="SVM/SVM_libsvm.R", name="libSVM"),
    list(path="trees/RandomForest_MLR-repro.R", name="RandomForest")#,
    #list(path="trees/ctree.R", name="CTree"),
    #list(path="analytic/FDA.R", name="FDA"),
    #list(path="evolutionary/evtree.R", name="evtree"),
    #list(path="statistical/NaiveBayes.R", name="NaiveBayes")
    #list(path="statistical/GaussianProcesses.R", name="gaussian_processes")
)

#thresholds <- list(
#  c("minor", "major", "critical"),
#  c("major", "critical")
#)

thresholds <- c(0.25, 0.5, 0.75, 1.0, 1.25, 1.50, 1.75, 2.0, 2.25, 2.50)

for(threshold in thresholds) {
    threshold_name <- str_replace(toString(threshold), "\\.", "_")
    print(paste("Running reproduction on ", threshold_name))
 
    for(smell in smells) {
        smellLoc <- paste(modelsRoot, smell$id, threshold_name, sep="/")
        mkdirs(smellLoc)
        json_data <- fromJSON(file=smell$schema)
        inputs <- sapply(json_data$inputVariables, function(var) return(list(key=var$key, type=var$variableType)))
        input_names <- sapply(json_data$inputVariables, function(var) return(var$key))
        outputs <- sapply(json_data$outputVariables, function(var) return(list(key=var$key, type=var$variableType)))
        output_names <- sapply(json_data$outputVariables, function(var) return(var$key))

        input_selector <- paste(input_names, collapse=" + ")
        output_selector <- paste(output_names, collapse=" + ")

        column_selector <- as.formula(paste(output_selector," ~ ",input_selector))
        myMeasures = list( ppv )# new performance metric defined in the R4PerformanceMetricV2.1.pdf report by Madeyski
        myCV <- cv5stratify # for more precise results change into repeated CV (e.g., rep20cv10stratify)

	print("Data cleanup - type conversion")
        data <- initialData %>% cleanDataByConvertingToCorrectTypes()

	print("Data cleanup - averaging")
        data <- data %>% cleanDataByTakingCareOfMultipleReviewsOfTheSameSample()

        # Threshold for smelly class
	print("Data cleanup - thresholding")
        data <- data %>% cleanDataByThresholdingSeverity(threshold)

        print("Data cleanup - removal of irrelevant stuff")
        data <- data %>% cleanDataByRemovingIrrelevantData(smell$name)

	# remove all features where fraction of values differing
        # from mode value is <= 2% #1%
	print("Data cleanup - removal of constant features")
        data <- mlr::removeConstantFeatures(data, perc = 0.02, dont.rm = "severity", show.info = FALSE)
        # data <- mlr::normalizeFeatures(data, target = "severity")

        modelRes <- {}
        for(model in models) {
            print(paste("Handling: ", model))
            source(paste(commonPath, model$path, sep="/"))

            res2 <- try(learner <- makeCQLearner())
            if(inherits(res2, "try-error"))
            {
                print(paste("Failed to create", model$name, "skipping results, continuing with the next one"))
                next
            }

            res3 <- try(if("missings" %in% mlr::getLearnerProperties(learner$id)){
                taskPerSmell = mlr::makeClassifTask(id = smell$name,
                data = data,
                target = "severity",
                positive = "TRUE"
                )
            } else{
                data <- data %>% cleanDataByMLbasedImputingNA(smell$name, createDummyFeatures = FALSE)
                taskPerSmell = mlr::makeClassifTask(id = stringr::str_c(smell$name, "NAimputedViaML"),
                data = data,
                target = "severity",
                positive = "TRUE"
                )
            })
            if(inherits(res3, "try-error"))
            {
                print(paste("Failed to prepare", model$name, "skipping results, continuing with the next one"))
                next
            }

            totalPerf <- NULL
            modelLoc <- paste(smellLoc, model$name, sep="/")
	    mkdirs(modelLoc)
            learner <- makeOverBaggingWrapper(learner, obw.rate = 10, obw.iters =10)
            res <- # try(
		       for(i in startIteration:(startIteration+iterations-1)) {
                target = paste(modelLoc, i, "target.model", sep="/")
                print(paste("Saving model to", target))
                #-- Performing (preliminary) repeated k-fold cross-validation ----
                tunedModel <- buildTunedModel(learner, myCV, myMeasures)
                resampleResult <- mlr::resample(learner = tunedModel$learner, task = taskPerSmell,
                resampling = myCV, measures = myMeasures, show.info = FALSE)

                perf <- makeStats(resampleResult$pred)
		print("AFTER STATS")
                if(is.null(totalPerf)) {
                    totalPerf <- perf$metrics
                } else {
                    totalPerf <- totalPerf + perf$metrics
                }

                metadata <- list(
                    timestamp = format(Sys.time()),
                    id = paste(smell$id, model$name, i, sep="/"),
                    libraries = c(), #getPackages(),
                    schema = smell$schema,
                    script = model$path,
                    datasets = dataSource,
                    predictors = input_names,
                    outputs = output_names,
                    performance = perf
                )

                mkdirs(target)
                saveRDS(tunedModel, file=paste(target, "/model.rds", sep=""))
                meta <- toJSON(metadata, indent=2)
                write(meta, file=paste(target, "/metadata.json",sep=""))
            }
	    #)

            if(inherits(res, "try-error"))
            {
                print(paste("Failed to handle", model$name, "skipping results, continuing with the next one"))
	        traceback()
            } else {
                totalPerf <- totalPerf / iterations
                modelRes[[model$name]] <- totalPerf

                restarget <- paste(modelLoc, "perf.json", sep="/")
                print(paste("Saving average results for", model$name ,"to", restarget))
                write(toJSON(totalPerf, indent=2), file=restarget)
            }
    }

        smelltarget <- paste(smellLoc, "totalPerf.json", sep="/")
        print(paste("Saving all results for ", smell$id, "to", smelltarget))
        write(toJSON(modelRes, indent=2), file=smelltarget)
    }
}

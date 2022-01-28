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

wd <- getwd()
setwd('/opt/model-builder/scripts')
packrat::on()
setwd(wd)

library(getopt)
library('R.utils')
library(jsonlite)
library(rjson)
library(caret)
library(hashmap)
library(stringr)
library(ModelMetrics)
library(mlr)
library(dplyr)

makeStats <- function(model) {
    print(performance(model))

    matrix <- calculateConfusionMatrix(model)
    levels <- length(model$result$task.desc$class.levels)
    confMatrix <- matrix$result[1:levels,1:levels]

    if(is.null(confMatrix)) {
        return(list(
            evaluation = "not done"
        ))
    }

    performanceMetrics <- performance(model, measures=list(
        acc,
        bac,
        ber,
        mmce,
        kappa,
        wkappa,
        multiclass.AvFbeta
    ))

#    cm <- lapply(seq_len(ncol(confMatrix)), function(i) confMatrix  [,i])

    obj <- list(
#        confusionMatrix = cm,
        metrics = performanceMetrics,
        evaluation = "10CV"
    )

    return(obj)
}

getPackages <- function() {
    installed <- installed.packages(fields = c("Version"))
    used_packages <- installed[(.packages()), c("Version")]
    return(used_packages)
}

parser = matrix(c("args", 'a', 0, "logical",
"build", 'b', 0, "logical",
"evaluate", 'v', 0, "logical",
"modelId", 'm', 1, "character",
"dataSource", 'd', 1, "character",
"schema", 's', 1, "character",
"scriptsDir", 'c', 1, "character",
"scriptPath", 'r', 1, "character",
"target", 't', 1, "character",
"model", 'e', 1, "character",
"workspace", 'w', 1, "character",
"mountPoint", "p", 1, "character",
"investigatedSmell", "g", 1, "character"
), byrow=TRUE, ncol=4)


argv <- getopt(parser)
# names(argsL) <- argsDF$V1

argv$beta=0.8
source(paste(argv$scriptsDir, "utils.R", sep=''))

if ( is.null(argv$build ) ) { argv$build = FALSE }
if ( is.null(argv$evaluate ) ) { argv$evaluate = FALSE }

json_data <- fromJSON(file=argv$schema)
inputs <- sapply(json_data$inputVariables, function(var) return(list(key=var$key, type=var$variableType)))
input_names <- sapply(json_data$inputVariables, function(var) return(var$key))
outputs <- sapply(json_data$outputVariables, function(var) return(list(key=var$key, type=var$variableType)))
output_names <- sapply(json_data$outputVariables, function(var) return(var$key))
data <- read.csv(file=argv$dataSource)

evdata <- select(data, c(output_names, input_names))

if(argv$evaluate) {
    writeLines(paste("=== Classes: ", outputs[,1]$key))
    model <- readRDS(file=argv$model)
    res <- predict(model, task = mlr::makeClassifTask(id = "smells", data = evdata, target = "severity"))
    for(row in 1:nrow(res$data)) {
        writeLines(paste("=== Classification:", res$data[row, 'id'], ":", res$data[row,'response']))
    }

    quit(save= "no", status = 0)
}


myMeasures = list( multiclass.AvFbeta )# new performance metric defined in the R4PerformanceMetricV2.1.pdf report by Madeyski
myCV <- cv10stratify # for more precise results change into repeated CV (e.g., rep20cv10stratify)

data <- data %>%
    cleanDataByTakingCareOfMultipleReviewsOfTheSameSample() %>%
    cleanDataByRemovingIrrelevantData(argv$investigatedSmell)
data <- select(data, c(output_names, input_names))

source(argv$scriptPath)

learner <- makeCQLearner()

if("missings" %in% getLearnerProperties(learner)){
    learner <- mlr::makeRemoveConstantFeaturesWrapper(learner, perc = 0.02, dont.rm = "severity")
    learner <- mlr::makePreprocWrapperCaret(learner, method = c("center", "scale", "pca", "spartialSign", thresh = 0.9, na.remove = FALSE))

    taskPerSmell <- mlr::makeClassifTask(id = argv$investigatedSmell,
    data = data,
    target = "severity")
} else{
    learner <- mlr::makeRemoveConstantFeaturesWrapper(learner, perc = 0.02, dont.rm = "severity")
    learner <- mlr::makePreprocWrapperCaret(learner, method = c("center", "scale", "pca", "spartialSign", thresh = 0.9, na.remove = FALSE))
    learner <- mlr::makeImputeWrapper(
        learner = learner,
        classes = list(
        integer = imputeConstant(-1),
        numeric = imputeConstant(-1),
        #takes care of double values!
        factor = imputeConstant(-1)
        )
    )
    taskPerSmell <- mlr::makeClassifTask(id = argv$investigatedSmell, data = data, target = "severity")
}

#-- Performing (preliminary) repeated k-fold cross-validation ----
resampleResult <- resample(learner = learner, task = taskPerSmell,
resampling = myCV, measures = myMeasures)

model = buildTunedModel(learner, myCV, myMeasures)

metadata <- list(
    timestamp = format(Sys.time()),
    id = argv$modelId,
    libraries = getPackages(),
    schema = argv$schema,
    script = argv$scriptPath,
    datasets = argv$dataSource,
    predictors = input_names,
    outputs = output_names,
    performance = makeStats(resampleResult$pred)
)

mkdirs(argv$target)
saveRDS(model, file=paste(argv$target, "/model.model", sep=""))
meta <- toJSON(metadata, indent=2)
write(meta, file=paste(argv$target, "/metadata.json",sep=""))

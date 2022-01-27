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
#-- My utils for R

#-- Install and load [if needed] regular R packages from CRAN  ----------------
requiredPackages = c(
  "bartMachine",
  "C50",
  "caret",
  "DescTools",
  "dplyr",
  "devtools",
  "earth",
  "fs",
  "gbm",
  "getopt",
  "git2r",
  "glmnet",
  "httr",
  "jsonlite",
  "kknn",
  "mlr",
  "mlrMBO",
  "parallel",
  "parallelMap",
  "randomForest",
  "readr",
  "rjson",
  "rpart.plot",
  "RPostgres",
  "rstudioapi",
  "stringr",
  "tictoc",
  "tidyverse",
  "xgboost"
)
for (p in requiredPackages) {
  if (!require(p, character.only = TRUE))
    install.packages(p)
  library(p, character.only = TRUE)
}


#-- Function: cleanDataByConvertingToCorrectTypes -----------------------------
cleanDataByConvertingToCorrectTypes <- function(data) {
  result <- data %>%
    mutate_at(
      .vars = c(
        # "reviewer_id",  # factor
        # "ABC_SIZE", # double
        "accessors",
        #integer, incl. NA
        "arity",
        #new - integer, incl. NA
        "attributes",
        #integer, incl. NA
        # "average_method_complexity", # double, incl. NA
        # "average_not_accessor_or_mutator_method_complexity", # double, są NA
        "block_nesting",
        #integer, incl. NA
        "children",
        #integer, incl. NA
        "cyclo",
        #integer, incl. NA
        "depth",
        #integer, incl. NA
        "functions_m",
        #integer, incl. NA
        #"instability", # double, incl. NA
        "instance_variables",
        #integer, incl. NA
        "lines_of_code",
        #integer
        "methods",
        #integer, incl. NA in functions
        "methods_weighted",
        #integer, incl. NA in functions
        "mutators",
        #integer, incl. NA in functions
        "not_accessors_or_mutators",
        #integer, incl. NA in functions
        "not_accessors_or_mutators_weighted",
        #integer, incl. NA in functions
        #"pmd_amw", # double, incl. NA
        "pmd_atfd",
        #integer, incl. NA
        "pmd_cbo",
        #integer, incl. NA 
        "pmd_cyclo",
        #integer, incl. NA
        "pmd_dit",
        #integer, incl. NA 
        "pmd_fanout",
        #integer, incl. NA 
        #"pmd_lcom5", # double, incl. NA
        "pmd_loc",
        #integer, incl. NA
        "pmd_locnamm",
        #integer, incl. NA
        "pmd_ncss",
        #integer, incl. NA
        "pmd_nlv",
        #integer, incl. NA
        "pmd_nmo",
        #integer, incl. NA
        "pmd_noa",
        #integer, incl. NA
        "pmd_noam",
        #integer, incl. NA
        "pmd_noc",
        #integer, incl. NA
        "pmd_nocm",
        #integer, incl. NA
        "pmd_nofa",
        #integer, incl. NA
        "pmd_nom",
        #integer, incl. NA
        "pmd_nomnamm",
        #integer, incl. NA
        "pmd_nonfnsa",
        #integer, incl. NA
        "pmd_nonfnsm",
        #integer, incl. NA
        "pmd_nop",
        #integer, incl. NA
        "pmd_nopa",
        #integer, incl. NA
        "pmd_nopm",
        #integer, incl. NA
        "pmd_nopra",
        #integer, incl. NA
        "pmd_nopva",
        #integer, incl. NA
        "pmd_nos",
        #integer, incl. NA
        "pmd_npath",
        #integer, incl. NA
        #"pmd_tcc", # double, incl. NA
        "pmd_wmc",
        #integer, incl. NA
        "pmd_wmcnamm",
        #integer, incl. NA
        #"pmd_woc", # double, incl. NA
        "return_values",
        # integer, incl. ONLY NA and 0!!!
        #"tree_impurity", # double, incl. NA
        "used_by",
        #integer, incl. NA
        "uses"#, #integer, incl. NA
        #"max" # ? incl. ONLY NA!!!
      ),
      .funs = as.integer
    ) %>%
    mutate_at(.vars = c(#"reviewer_id", # removed - requested by @tomek et al.
      "severity",
      "type",
      "smell"),
      .funs = as.factor) # factor))
  return(result)
}


#-- Function: cleanDataByRemovingIrrelevantData -------------------------------
cleanDataByRemovingIrrelevantData <- function(data, smell) {
  data <- select(
    data,-c(
      X1,
      id,
      sample_id,
      reviewer_id, #might be reconsidered!
      sample_id..14,
      review_timestamp,
      type, #might be reconsidered in case of multioutput predictions
      name,
      analysis_timestamp,
      name..10,
      language,
      repository,
      commit_hash
    )
  )
  switch(
    smell,
    "blob" = {
      data <- data %>%
        filter(smell == "blob")
    },
    "data class" = {
      data <- data %>%
        filter(smell == "data class")
    },
    "long method" = {
      data <- data %>%
        filter(smell == "long method")
    },
    "feature envy" = {
      data <- data %>%
        filter(smell == "feature envy")
    }
  )
}


#-- Function: median_ordered --------------------------------------------------
median_ordered <- function(x)
{
  levs <- levels(x)
  m <- median(as.integer(x))
  if (floor(m) != m)
  {
    warning("Median is between two values; using the first one")
    m <- floor(m)
  }
  ordered(m, labels = levs, levels = seq_along(levs))
}


#-- Function: cleanDataByTakingCareOfMultipleReviewsOfTheSameSample------------
cleanDataByTakingCareOfMultipleReviewsOfTheSameSample <-
  function(data) {
    
    numberOfRows <- nrow(data)
    
    dataGrouped <- data %>%
      group_by(sample_id, smell)
    
    numberOfGroups <- n_groups(dataGrouped)

    
    if (numberOfRows > numberOfGroups) {
      #grouping was necessary
      
      dataCleaned <- dataGrouped %>%
        # remove sode smaples with 2 different seveiry values assigned by reviewers
        filter(n() == 1L |
                 ((n() == 2) &
                    n_distinct(severity) == 1) | ((n() >= 3))) %>%
        # in case of samples with different severity values assign median
        mutate(severity = replace(
          severity,
          n_distinct(severity) != 1,
          median_ordered(severity)
        )) %>%
        # retain only one severity value per code sample 
        filter(n() == 1L |
                 ((n() >= 2) & row_number() == ceiling(n() / 2)))

    }
    else{
      dataCleaned <- dataGrouped
    }
    
    dataCleanedUngrouped <- ungroup(dataCleaned)
    
    return(dataCleanedUngrouped) 
    
  }


cleanDataByImputingNA <- function(data, smell, createDummyFeatures = FALSE) {
  #consider smell-targeted imputing data if needed
  if (smell == "blob" |
      smell == "data class" |
      smell == "long method" | smell == "feature envy") {
    
    if(createDummyFeatures == FALSE) 
      dummyFeatures = character(0L)
    else
      dummyFeatures = c("integer", "numeric", "factor")
    
    imputeResult <- mlr::impute(
      data,
      target = "severity",
      classes = list(
        integer = imputeMedian(),
        numeric = imputeMean(),
        #takes care of double values!
        factor = imputeMode()
      ),
      dummy.classes = dummyFeatures
    )
    data <- imputeResult$data
  }
  return(data)
}


#-- Function: cleanDataByMLbasedImputingNA ------------------------------------
cleanDataByMLbasedImputingNA <- function(data, smell, createDummyFeatures = FALSE) {
  #consider smell-targeted imputing data if needed
  if (smell == "blob" |
      smell == "data class" |
      smell == "long method" | smell == "feature envy") {
    
    if(createDummyFeatures == FALSE) 
      dummyFeatures = character(0L)
    else
      dummyFeatures = c("integer", "numeric", "factor")
    
    imputeResult <- mlr::impute(
      data,
      target = "severity",
      classes = list(
        integer = imputeLearner(makeLearner("regr.rpart")),
        numeric = imputeLearner(makeLearner("regr.rpart")),
        #takes care of double values!
        factor = imputeLearner(makeLearner("classif.rpart"))
      ),
       dummy.classes = dummyFeatures
    )
    
    data <- imputeResult$data
  }
  return(data)
}



myMakeLearner <- function(learnerName){
  learner <- mlr::makeLearner(learnerName)
  if("factors" %in% mlr::getLearnerProperties(learnerName)){
  } else{
    
    learner <- learner %>% mlr::makeDummyFeaturesWrapper()
  }
  return(learner)
}




#-- New multiclass classification performance metric multiclass.AvFbeta 
#-- suited for imbalanced data sets
#-- described in R4PerformanceMetricV2.1.pdf report by Lech Madeyski
measuremulticlass.AvFbeta = function(pred){ 
  levels <- length(pred$task.desc$class.levels)
  cm <- mlr::calculateConfusionMatrix(pred)
  confMatrix <- cm$result[1:levels,1:levels]
  
  beta <- argv$beta #0.8 # beta <- 1
 
  result <- 0
  factor1denominator <- 0
  factor2 <- 0
  tp <- vector(mode="numeric", length=levels)
  t <- vector(mode="numeric", length=levels)
  p <- vector(mode="numeric", length=levels)
  phi <- vector(mode="numeric", length=levels)
  
  
  for(i in 1:levels) {
    t[i] <- 0
    
    # tpi is the number of true positives for class i (i.e., mii)
    tp[i] <- confMatrix[i,i]
    
    # pi is the total number of predicted examples for class i (i.e., pi = m1i + m2i + m3i + m4i);
    p <- apply(confMatrix, 2, sum) # here the value of the second parameter in apply(), 1 or 2, indicates that the sum should be performed for each row or column
    
    # ti is the total number of true examples for class i (i.e., ti = mi1 + mi2 + mi3 + mi4);
    t <- apply(confMatrix, 1, sum) # here the value of the second parameter in apply(), 1 or 2, indicates that the sum should be performed for each row or column
    
    phi[i] <- ( (1/t[i])/sum(1/t[1:levels]) )
    
    #factor1denominator <- factor1denominator + phi[i]    # factor1denominator is always equal 1 so can be removed
    factor2 <- factor2 + ( phi[i] * (1 + (beta^2) ) * confMatrix[i,i]) / ( ( (beta^2) * t[i]) + p[i] )
  }
  #result <- (1/factor1denominator) * factor2   # factor1denominator is always equal 1 so can be removed (kept to reflect equation in R4PerformanceMetricV2.1.pdf report)
  result <- factor2
  return(result)
}

multiclass.AvFbeta = makeMeasure(
  id = "multiclass.AvFbeta", 
  minimize = FALSE,
  best = 1, 
  worst = 0,
  properties = c("classif", "classif.multi", "req.pred", "req.truth"),
  name = "multiclass.AvFbeta measure",
  note = "Defined as: see R4PerformanceMetricV2.1.pdf report by Lech Madeyski",
  fun = function(task, model, pred, feats, extra.args) {
                   measuremulticlass.AvFbeta(pred)
                 }
  )


cv5stratify <- makeResampleDesc("CV", iters = 5L, stratify = TRUE)
cv10stratify <- makeResampleDesc("CV", iters = 10L, stratify = TRUE)
rep2cv10stratify <- makeResampleDesc("RepCV", folds = 10L, reps = 2L, stratify = TRUE)
rep20cv10stratify <- makeResampleDesc("RepCV", folds = 10L, reps = 20L, stratify = TRUE)



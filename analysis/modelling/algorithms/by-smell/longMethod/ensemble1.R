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

makeCQLearner <- function() {
    classif.naiveBayes = myMakeLearner("classif.naiveBayes")
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

    learners <- lapply(listOfBaseLearners, myMakeLearner)
    learners <- lapply(learners, setPredictType, "prob")


    return(mlr::makeStackedLearner(
        base.learners = learners,
        super.learner = classif.naiveBayes,
        predict.type = "prob",
        method = "stack.cv",
        use.feat = TRUE
        ) %>% setLearnerId("ensemble1.learners")
    )

}

buildTunedModel <- function(learner, resampling, measures) {
    myLearnerTuned = learner
    return(train(learner, taskPerSmell))
}

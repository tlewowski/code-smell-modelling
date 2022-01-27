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

    return(mlr::makeStackedLearner(
        base.learners = learnersCombined,
        super.learner = classif.naiveBayes,
        predict.type = "prob",
        method = "stack.cv",
        use.feat = TRUE
        ) %>%
        setLearnerId("ensemble1.learnersCombined")
    )

}

buildTunedModel <- function(learner, resampling, measures) {
    myLearnerTuned = learner
    return(train(learner, taskPerSmell))
}

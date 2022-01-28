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

    listOfNAILearners <- list(
    "classif.earth",
    "classif.evtree",
    "classif.glmnet",
    "classif.ksvm",
    "classif.lda",
    "classif.LiblineaRL1LogReg",
    "classif.LiblineaRL2LogReg",
    "classif.multinom",
    "classif.randomForest",
    "classif.ranger",
    "classif.rpart",
    "classif.sda",
    "classif.sparseLDA",
    "classif.svm"
    )


    learnersNAI <- lapply(listOfNAILearners, myMakeLearnerNAI)
    learnersNAI <- lapply(learnersNAI, setPredictType, "prob")

    return(mlr::makeStackedLearner(
    base.learners = learnersNAI,
    super.learner = classif.lda.NAI,
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

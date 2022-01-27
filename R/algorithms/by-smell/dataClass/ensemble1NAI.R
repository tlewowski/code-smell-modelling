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
    listOfNAILearners <- list(
    "classif.earth",
    "classif.glmnet",
    "classif.kknn",
    "classif.lda", #really good
    "classif.LiblineaRL1LogReg",
    "classif.multinom",
    "classif.randomForest",
    "classif.ranger",
    "classif.sda", #really good
    "classif.sparseLDA" #really good
    )


    learnersNAI <- lapply(listOfNAILearners, myMakeLearnerNAI)
    learnersNAI <- lapply(learnersNAI, setPredictType, "prob")

    return(mlr::makeStackedLearner(
    base.learners = learnersNAI,
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

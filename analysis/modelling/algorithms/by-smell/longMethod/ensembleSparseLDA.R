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

    classif.sparseLDA.NAI = mlr::makeImputeWrapper(
        learner = classif.lda,
        classes = list(
        integer = imputeLearner(makeLearner("regr.rpart")),
        numeric = imputeLearner(makeLearner("regr.rpart")),
        #takes care of double values!
        factor = imputeLearner(makeLearner("classif.rpart"))
        )
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


    learnersNAI <- lapply(listOfNAILearners, myMakeLearnerNAI)
    learnersNAI <- lapply(learnersNAI, setPredictType, "prob")

    return(mlr::makeStackedLearner(
    base.learners = learnersNAI,
    super.learner = classif.sparseLDA.NAI,
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

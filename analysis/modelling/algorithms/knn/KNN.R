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
    return(myMakeLearner("classif.kknn")) #("classif.kknn")
}

buildTunedModel <- function(learner, resampling, measures) {
    #-- Hyperparameter tuning of K ----
    paramSpace <- makeParamSet(makeDiscreteParam("k", values = 1:50))

    print("Starting tuning")
    searchStrategy <- makeTuneControlGrid()

    tunedParams <-
    tuneParams(learner, task = taskPerSmell,
    resampling = resampling,
    measures = measures,
    par.set = paramSpace,
    control = searchStrategy)

    #-- Training final model with tuned params (e.g., K in knn)
    myLearnerTuned <- setHyperPars(learner, par.vals = tunedParams$x)
    return(train(myLearnerTuned, taskPerSmell))
}

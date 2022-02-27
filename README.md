# Code smell modelling

## Features

This repository provides two main functionalities:

 - building machine learning models for code smells,
 - generating model summaries as data tables and boxplots

## How to use

### Modelling

Modelling module is located in `analysis/modelling` directory.

Entrypoint to the modelling is the `Multirun.R` script. 
The expected invocation is:

```R
Rscript Multirun.R
```

The script should be executed from `analysis/modelling` directory, as it uses relative paths in several locations throughout the code.
Relative paths are the only reason for that, modifying them will let you run the scripts from any location.

Before running it, make sure that you have all required libraries installed.
You can install them again by running `Rscript libraries.R`

At some point those may become dependencies managed by Packrat, but right now they aren't.
Detailed list of used dependencies is embedded in metadata for each of the machine learning models.


#### Configuration

The script is not very well-separated - it has several locations in which a hardcoded path is used. Those include the following variables and functionalities:

 - `smells` - list of schemas used for machine learning
 - `commonPath` - root for algorithms that are used as "plugins"
 - `dataSource` - location of base data set
 - `modelsRoot` - target location for built models
 - sourced utilities

Additional parameters that may be adjusted are:

 - `iterations` - number of times a model is built for each algorithm
 - `models` - list of models that should be trained
 - `thresholds` - list of thresholds for severity median for building data set

None of them can be altered without modifying source code.

### Summaries

Summarising module is located in `analysis/diagrams` directory.

Entrypoint is the `tables_and_boxplots.R` file.
The expected invocation is:

```R
Rscript tables_and_boxplots.R
```

The script should be executed from `analysis/diagrams` directory, as it uses relative paths in several locations throughout the code.
Relative paths are the only reason for that, modifying them will let you run the scripts from any location.

All required dependencies are listed in the script, but you need to install them manually,
as no automated script (similar to `libraries.R` or `Packrat.lock`) is provided for this module.

#### Configuration

Five variables in the script are mainly used to control its behavior:

 - `imagesRoot` - path in the local filesystem where scripts results will be stored
 - `modelsRoot` - path in the local filesystem where built models (and their metadata) are stored. Models should be stored in the same directory structure that is created by `Multirun.R`
 - `smellsToAnalyze` - list of smells that will be analyzed and their mapping to "publication name"
 - `datasetsToAnalyze` - list of datasets (threshold levels) that will be analyzed and their mapping to "publication name"
 - `METRIC_NAME_MAPPING` - only metrics present in this mapping are preserved in the output tables and boxplots, visible name is the value of the mapping


None of them can be altered without modifying source code.

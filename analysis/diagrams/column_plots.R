library("rjson")
library("purrr")
library("tidyverse")

METRIC_NAME_MAPPING <- c(
    "acc" = "Accuracy", 
    "bac" = "Balanced accuracy",
    "mcc" = "MCC",
    "ppv" = "Precision",
    "tpr" = "Recall", 
    "f1" = "F-score"
)



mapMetricName <- function(record) {
    print(record)
    record$metric <- METRIC_NAME_MAPPING[[record$metric]];
    record
}

drawPictures <- function(results, title, plot_path, smell) {
    metrics <- names(results) %>% 
    map(function(t) names(results[[t]]) %>% 
        map(function(c) tibble(
            smell=smell, 
            algorithm=t, 
            metric=c, 
            value=as.numeric(results[[t]][[c]]))
        )
    ) %>% 
    flatten() %>% 
    bind_rows()

    plotted <- metrics %>% 
      filter(.$metric %in% c("acc", "bac", "f1", "ppv", "tpr", "mcc"))

    for(name in names(METRIC_NAME_MAPPING)) {
        plotted$metric[plotted$metric == name] <- METRIC_NAME_MAPPING[[name]]
    }
    

    ggplot(data = plotted) +
      ggtitle(title) + 
      xlab("Algorithm") + 
      ylab("Metric value") +
      geom_bar(mapping = aes(x = algorithm, y = value, fill = algorithm), stat = "identity", show.legend = FALSE) +
      theme(axis.text.x = element_text(angle=45, vjust=1, hjust=1)) +
      facet_wrap(~metric, nrow = 2) +
      guides(fill = guide_legend(title = "Algorithms"))

    ggsave(plot_path)
}

resultsRoot <- "../../models/2022-01-07"
targetRoot <- "../../images/2022-01-07"
smellsToAnalyze <- c(
    "blob" = "Blob", 
    "dataclass" = "Data Class", 
    "featureenvy" = "Feature Envy",
    "longmethod" = "Long Method"
    )

datasetsToAnalyze <- c(
    "minor" = "DS1",
    "major" = "DS2"
)

for(smell in names(smellsToAnalyze)) {
    for(ds in names(datasetsToAnalyze)) {
        datafile <- paste(resultsRoot, smell, ds, "totalPerf.json", sep="/")
        targetfile <- paste(targetRoot, paste(paste(smell, ds, sep="_"), ".png", sep = ""), sep="/", collapse="_")
        title <- paste(smellsToAnalyze[[smell]], paste("(", datasetsToAnalyze[[ds]], ")", sep = ""), sep = " ")
        drawPictures(fromJSON(file = datafile), title, targetfile, smell)
    }
}



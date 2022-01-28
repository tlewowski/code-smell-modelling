library("rjson")
library("purrr")
library("tidyverse")
library("tidyjson")
library("xtable")

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

calculateStats <- function(directory) {
    # 1. Read all data files
    # Data is in '<alg>/<num>' directory
    # in 'metadata.json' file in 'performance.metrics' object
    
    # There are no other 'metadata.json' files, so a recursive search will be fine
    metrics <- c()
    datafiles <- list.dirs(directory, recursive = TRUE, full.names = TRUE)
    for(d in datafiles) {
        filename <- paste(d, "metadata.json", sep = "/")
        if(file.exists(filename)) {
            data <- fromJSON(file = filename)
            
            # Originally in the JSON files it's an object where keys are metric names
            filemetrics <- data$performance$metrics
            tbl <- names(filemetrics) %>% map(function(c) tibble(
                id = data$id,  
                metric=c, 
                value=as.numeric(filemetrics[[c]])
            ))
            metrics <- c(metrics, tbl)
        }
    }

    # 2. Let's put all values together to one vector and split smell and algorithm
    # and values are values of the metric (numeric) or "NA" string if cannot be calculated
    # 
    smellmetrics <- metrics %>% 
      bind_rows() %>% 
      separate(id, c("smell", "algorithm", NA), "/")
      
    # 3. Let's calculate some statistics per algorithm:
    # quartiles (Q1, Q2, Q3, Q4)
    # mean and standard deviation
    smellmetrics_summaries <- smellmetrics %>% 
      unite("key", c(smell, algorithm, metric), sep="/")  %>% 
      group_by(key) %>%
      summarize(
          mean=mean(value, na.rm = TRUE), 
          stddev=sd(value, na.rm = TRUE), 
          min = min(value, na.rm = TRUE),
          q1 = quantile(value, .25, na.rm = TRUE),
          q2 = quantile(value, .50, na.rm = TRUE),
          q3 = quantile(value, .75, na.rm = TRUE),
          max = max(value, na.rm = TRUE),
        ) %>%
        separate(key, c("smell", "algorithm", "metric"), sep="/")
    
    return (list(smellmetrics, smellmetrics_summaries))
}


resultsRoot <- "/media/CodeSmells_2022/models/2022-01-16_2"
targetRoot <- "/media/CodeSmells_2022/images/2022-01-17"

smellsToAnalyze <- c(
    "blob" = "Blob", 
    "dataclass" = "Data Class", 
    "featureenvy" = "Feature Envy",
    "longmethod" = "Long Method"
    )

datasetsToAnalyze <- c(
    "minor_major_critical" = "DS1",
    "major_critical" = "DS2"
)

for(smell in names(smellsToAnalyze)) {
    for(ds in names(datasetsToAnalyze)) {
        datadir <- paste(resultsRoot, smell, ds, sep="/")
        stats_full <- calculateStats(datadir)
        stats_table <- stats_full[[2]] %>% 
          filter(metric %in% names(METRIC_NAME_MAPPING))

        tab <- xtable(stats_table, 
          caption = paste("Statistics for", smellsToAnalyze[smell], "detection", sep = " "), 
          label = paste("tab:stats:", smell, ":", ds, sep="")
        )
        print(
            tab, 
            type = "latex",
            file = paste(targetRoot, paste(paste("stats", "table", smell, ds, sep="_"), "tex", sep="."), sep="/")
        )        
        
        stats_boxplot <- stats_full[[1]] %>% 
          filter(metric %in% names(METRIC_NAME_MAPPING))

        for(name in names(METRIC_NAME_MAPPING)) {
            stats_boxplot$metric[stats_boxplot$metric == name] <- METRIC_NAME_MAPPING[[name]]
        }

        title <- paste(smellsToAnalyze[[smell]], paste("(", datasetsToAnalyze[[ds]], ")", sep = ""), sep = " ")
        ggplot(data = stats_boxplot) +
            ggtitle(title) + 
            xlab("Algorithm") + 
            ylab("Metric value") +
            geom_boxplot(mapping = aes(x = algorithm, y = value, fill = algorithm), show.legend = FALSE, na.rm = TRUE) +
            theme(axis.text.x = element_text(angle=45, vjust=1, hjust=1)) +
            facet_wrap(~metric, nrow = 2) +
            guides(fill = guide_legend(title = "Algorithms"))
        

        plot_path <- paste(targetRoot, paste(paste("summaries", "boxplot", smell, ds, sep="_"), "png", sep="."), sep="/")
        ggsave(plot_path)
    }
}



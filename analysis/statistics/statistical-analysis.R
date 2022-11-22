  library("stats")
  library("tidyverse")
  library("PMCMR")
  library("rcompanion")
    
  results_dir <- "../../2022-04-15-package-post-analysis.tar/2022-02-27-results"
  
  datafiles <- list.files(path = results_dir, pattern = "full-results.*")
  
  for(datafile in datafiles) {
    data <- read_csv(paste(results_dir, datafile, sep="/"), show_col_types = FALSE)
    relevant_data <- data %>% filter(metric == "mcc") %>% select(c("algorithm", "value"))
    df <- as.data.frame(relevant_data)
    df$algorithm <- factor(df$algorithm)
  
    print(datafile)  
    print(kruskal.test(value ~ algorithm, data = df))
    print(PMCMRplus::kwAllPairsNemenyiTest(value ~ algorithm, data = df, dist="Chisquare"))
    print(multiVDA(value ~ algorithm, data=df))
  }

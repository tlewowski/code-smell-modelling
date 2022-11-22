library("mlr")

file <- "../../data/mlcq-with-metrics.csv"
data <- read_csv(file, show_col_types = FALSE) %>%
  select(-c(
    col1, 
    id, 
    sample_id, 
    reviewer_id, 
    severity, 
    review_timestamp, 
    name, 
    analysis_timestamp, 
    name..10,
    sample_id..14,
    commit_hash,
    repository,
    language,
    smell
  )
)

funcs <- data %>% filter(type == "function") %>% 
  select(c(
    type,
    abc_size,
    arity,
    block_nesting,
    cyclo,
    lines_of_code,
    pmd_atfd,
    pmd_cyclo,
    pmd_loc,
    pmd_ncss,
    pmd_nlv,
    pmd_nop,
    pmd_nos,
    pmd_npath,
    return_values
  )) %>% 
  mlr::removeConstantFeatures(data, perc = 0.02, dont.rm = "type", na.ignore = TRUE, show.info = FALSE, wrap.tol=0.00001) %>%
  select(-type)
  


classes <- data %>% filter(type == "class") %>% 
  select(-c(
    arity,
    block_nesting,
    cyclo,
    pmd_cyclo,
    pmd_nlv,
    pmd_nop,
    pmd_nos,
    pmd_npath,
    return_values
  )) %>% 
  mlr::removeConstantFeatures(data, perc = 0.02, dont.rm="type", na.ignore = TRUE,, show.info = FALSE, wrap.tol=0.00001) %>%
  select(-type)



classescor <- cor(classes, method="pearson", use="complete.obs")
funcscor <- cor(funcs, method="pearson", use="complete.obs")

write.csv(classescor, "../../statistics/classescor.csv")
write.csv(funcscor, "../../statistics/funcscor.csv")

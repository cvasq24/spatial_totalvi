library(ggplot2)
library(dplyr)
library(stringr)

setwd("./benchmark/other_tools")
res.list <- list.files('../results', pattern = 'runtime_results.csv', include.dirs = TRUE, full.names = TRUE)
names(res.list) <- str_replace_all(str_split_i(res.list, "_", 1), "../results/", "")

df.list <- lapply(res.list, read.csv)

df_concat <- bind_rows(df.list, .id = "model")


p1 <- ggplot(df_concat, aes(x = subset_percent, y = runtime_sec, colour = model)) +
    geom_point(size = 3) +
    geom_line(size = 1) +
    scale_colour_brewer(palette = 'Set2') +
    labs(y = "Runtime (s)", x = 'Dataset size (%)') +
    theme_classic()

ggsave('../results/scalability_plot.pdf', p1, width = 5, height = 4)

library(tidyverse)
library(readxl)
library(patchwork)

read_all_sheets <- function(fp = "", ...) {
  sheet_names <- excel_sheets(fp)
  list_all <- lapply(sheet_names, function(x) {
    as.data.frame(read_excel(fp, sheet = x, ...)) })
  names(list_all) <- sheet_names
  return(list_all)
}

enrich_stdplot <- function(enrichResult_df, pos_neg = "pos", user_title = "Enrichr plot") {
  if (pos_neg == "pos"){
    fill <- "#EF553B"
  } else if (pos_neg == "neg") {
    fill <- "#636EFA"
  } else {
    fill <- "#93e9be"
  }
  enrich_plotdf <- enrichResult_df %>% mutate(neglogadjP = log(`Adjusted P-value`, 10)*(-1))
  enrich_plotdf <- enrich_plotdf %>% filter(`Adjusted P-value` < 0.05) %>% head(10)
  fig <- ggplot(enrich_plotdf, aes(x = neglogadjP, y = reorder(Term, neglogadjP, sum)))+
    geom_col(fill = fill) +
    scale_x_continuous(expand = c(0, NA)) +
    labs(title = user_title, x = "-log10(adjusted p-value)", y = "") +
    theme_classic() +
    theme(plot.title = element_text(hjust = 0.5),
          plot.margin = margin(5,10,5,10, "pt"))
  return(fig)
}

dfs <- read_all_sheets('../results/vanilla_totalVI_enrichr.xlsx')

plot.list <- list()

for (nm in names(dfs)) {
    enr.df <- dfs[[nm]]
    n_sig <- enr.df %>% filter(`Adjusted P-value` < 0.05) %>% nrow()
    if (n_sig > 0) {
        p1 <- enrich_stdplot(dfs[[nm]], pos_neg = 'pos', user_title = str_replace(nm, "_up", ""))
        plot.list[[nm]] <- p1
    }
}

ggsave("../results/vanilla_totalVI_enrichr_sig.pdf", wrap_plots(plot.list, ncol = 2), width = 12, height = 8)

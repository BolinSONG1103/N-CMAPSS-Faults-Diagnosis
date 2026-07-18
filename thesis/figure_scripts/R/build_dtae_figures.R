# 第3章 DTAE 正式出图（图3-11至图3-18）
# 只读取锁定 CSV；不重新训练、不重新选参。

suppressPackageStartupMessages({
  library(ggplot2)
  library(readr)
  library(dplyr)
  library(tidyr)
  library(scales)
  library(patchwork)
})

theme_thesis <- function() {
  theme_bw(base_family = "Microsoft YaHei", base_size = 10.5) +
    theme(
      panel.grid.minor = element_blank(),
      panel.grid.major = element_line(colour = "#E7E7E7", linewidth = 0.3),
      axis.title = element_text(colour = "#222222"),
      axis.text = element_text(colour = "#222222"),
      plot.title = element_text(face = "bold", hjust = 0.5, size = 12),
      plot.subtitle = element_text(hjust = 0.5, colour = "#555555"),
      legend.position = "bottom",
      legend.title = element_blank()
    )
}

save_thesis <- function(plot, out_dir, stem, width = 7.2, height = 4.8) {
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  ggsave(file.path(out_dir, paste0(stem, ".pdf")), plot, width = width, height = height,
         device = cairo_pdf, units = "in", bg = "white")
  ggsave(file.path(out_dir, paste0(stem, ".png")), plot, width = width, height = height,
         dpi = 600, units = "in", bg = "white")
}

read_conf <- function(path) {
  x <- read_csv(path, show_col_types = FALSE, name_repair = "minimal")
  names(x)[1] <- "truth"
  x |> pivot_longer(-truth, names_to = "prediction", values_to = "value")
}

conf_plot <- function(df, title) {
  ggplot(df, aes(prediction, truth, fill = value)) +
    geom_tile(colour = "white", linewidth = 0.8) +
    geom_text(aes(label = percent(value, accuracy = 0.1)), size = 3.2) +
    scale_fill_gradientn(colours = c("#F7FBFF", "#C6DBEF", "#6BAED6", "#2171B5", "#08306B"),
                         limits = c(0, 1), labels = percent) +
    coord_equal() + labs(x = "预测类别", y = "真实类别", title = title) +
    theme_thesis() + theme(axis.text.x = element_text(angle = 28, hjust = 1))
}

build_dtae_figures <- function(repo_root = getwd()) {
  fd <- file.path(repo_root, "improve", "figdata")
  out <- file.path(repo_root, "thesis", "figures", "r_official")
  part_metrics_path <- file.path(repo_root, "improve", "dtae_metrics_part.csv")
  family_metrics_path <- file.path(repo_root, "improve", "dtae_metrics_family.csv")

  # 图3-11：检测混淆 + 关键指标
  det <- read_conf(file.path(fd, "detection_confusion.csv")) |>
    group_by(truth) |> mutate(value = value / sum(value)) |> ungroup()
  p11a <- conf_plot(det, "(a) 逐循环检测混淆矩阵")
  ds <- read_csv(file.path(fd, "detection_summary.csv"), show_col_types = FALSE)
  vals <- c(
    `事件级检出率` = ds$value[ds$metric == "event_detection_rate"],
    `健康循环正确拒绝率` = 1 - ds$value[ds$metric == "healthy_cycle_far"],
    `四部件 macro-F1` = ds$value[ds$metric == "macro_f1"],
    `四部件 micro-F1` = ds$value[ds$metric == "micro_f1"]
  )
  p11b <- tibble(metric = factor(names(vals), levels = names(vals)), value = as.numeric(vals)) |>
    ggplot(aes(metric, value, fill = metric)) + geom_col(width = 0.7, show.legend = FALSE) +
    geom_hline(yintercept = 0.95, linetype = 2, colour = "#777777") +
    geom_text(aes(label = sprintf("%.3f", value)), vjust = -0.35, size = 3.2) +
    scale_y_continuous(limits = c(0, 1.08), expand = expansion(mult = c(0, 0))) +
    labs(x = NULL, y = "指标值", title = "(b) 检测与主隔离关键指标") + theme_thesis() +
    theme(axis.text.x = element_text(angle = 18, hjust = 1))
  p11 <- (p11a | p11b) + plot_annotation(title = "DTAE 两阶段故障检测（分布内在线诊断）")
  save_thesis(p11, out, "图3-11_DTAE故障检测", 10.8, 4.6)

  # 图3-12：四部件混淆
  p12 <- conf_plot(read_conf(file.path(fd, "confusion_part.csv")),
                   "四部件主诊断逐类召回/共现矩阵") +
    labs(caption = "多标签并发退化时行和不要求等于1；对角为逐类召回，非对角含并发共现。")
  save_thesis(p12, out, "图3-12_四部件主诊断混淆矩阵", 6.8, 6.0)

  # 图3-13：四部件 P/R/F1
  pm <- read_csv(part_metrics_path, show_col_types = FALSE) |>
    pivot_longer(c(precision, recall, F1), names_to = "metric", values_to = "value") |>
    mutate(metric = recode(metric, precision = "精确率", recall = "召回率", F1 = "F1"))
  p13 <- ggplot(pm, aes(name, value, fill = metric)) +
    geom_col(position = position_dodge(width = 0.78), width = 0.72) +
    geom_hline(yintercept = 0.95, linetype = 2, colour = "#777777") +
    scale_y_continuous(limits = c(0.90, 1.015)) +
    scale_fill_manual(values = c("精确率"="#6BAED6", "召回率"="#31A354", "F1"="#D62728")) +
    labs(x = NULL, y = "指标值", title = "四部件主诊断逐部件性能（5 seed 聚合）") + theme_thesis()
  save_thesis(p13, out, "图3-13_四部件逐类性能", 8.2, 4.5)

  # 图3-14：t-SNE
  latent <- read_csv(file.path(fd, "latent_tsne.csv"), show_col_types = FALSE)
  p14 <- ggplot(latent, aes(tsne1, tsne2, colour = label)) +
    geom_point(size = 1.05, alpha = 0.58) +
    scale_colour_manual(values = c("正常"="#9E9E9E", "风扇"="#2CA02C",
                                   "高压压气机"="#1F77B4", "低压压气机"="#9467BD", "涡轮"="#D95F02")) +
    labs(x = "t-SNE 维度 1", y = "t-SNE 维度 2", title = "DTAE 潜空间的类别聚集结构") + theme_thesis()
  save_thesis(p14, out, "图3-14_DTAE潜空间tSNE", 7.0, 5.4)

  # 图3-15：诊断时间线
  tl <- read_csv(file.path(fd, "timeline.csv"), show_col_types = FALSE)
  p15a <- ggplot(tl, aes(cycle)) +
    geom_line(aes(y = severity_true, colour = "真实严重度"), linewidth = 0.8) +
    geom_line(aes(y = severity_est, colour = "估计严重度"), linewidth = 0.7) +
    scale_colour_manual(values = c("真实严重度"="#333333", "估计严重度"="#D62728")) +
    labs(x = NULL, y = "量程归一化严重度", title = "(a) 连续退化程度") + theme_thesis()
  p15b <- ggplot(tl, aes(cycle, detected)) + geom_step(colour = "#1F77B4", linewidth = 0.8) +
    scale_y_continuous(breaks = c(0,1), labels = c("正常","报警")) +
    labs(x = NULL, y = NULL, title = "(b) 故障检测") + theme_thesis() + theme(legend.position = "none")
  tl_long <- tl |> select(cycle, truth, pred) |> pivot_longer(c(truth, pred), names_to = "row", values_to = "label") |>
    mutate(row = recode(row, truth = "真实部件", pred = "预测部件"))
  p15c <- ggplot(tl_long, aes(cycle, row, fill = label)) + geom_tile() +
    scale_fill_manual(values = c("正常"="#F0F0F0", "风扇"="#2CA02C", "高压压气机"="#1F77B4",
                                 "低压压气机"="#9467BD", "涡轮"="#D95F02")) +
    labs(x = "飞行循环", y = NULL, title = "(c) 部件隔离") + theme_thesis()
  p15 <- (p15a / p15b / p15c) + plot_layout(heights = c(2.2, 0.7, 0.9)) +
    plot_annotation(title = "代表发动机从退化出现到部件判定的诊断时间线")
  save_thesis(p15, out, "图3-15_代表发动机诊断时间线", 9.5, 6.8)

  # 图3-16：五部件细分混淆
  p16 <- conf_plot(read_conf(file.path(fd, "confusion_family.csv")),
                   "五部件族细分逐类召回/共现矩阵") +
    labs(caption = "HPC-LPC非对角项对应真实并发退化；HPT-LPT项反映涡轮内部混叠。")
  save_thesis(p16, out, "图3-16_五部件族细分混淆矩阵", 6.9, 6.1)

  # 图3-17：HPT/LPT 深度分析
  fm <- read_csv(family_metrics_path, show_col_types = FALSE)
  p17a <- fm |> filter(name %in% c("HPT","LPT")) |>
    pivot_longer(c(precision, recall, F1), names_to = "metric", values_to = "value") |>
    mutate(metric = recode(metric, precision="精确率", recall="召回率", F1="F1")) |>
    ggplot(aes(name, value, fill = metric)) + geom_col(position = "dodge") +
    geom_hline(yintercept = 0.85, linetype = 2, colour = "#777777") +
    scale_fill_manual(values = c("精确率"="#6BAED6", "召回率"="#31A354", "F1"="#D62728")) +
    scale_y_continuous(limits = c(0.60, 1.02)) +
    labs(x = NULL, y = "指标值", title = "(a) 涡轮内部细分性能") + theme_thesis()
  fc <- read_conf(file.path(fd, "confusion_family.csv"))
  mix <- fc |> filter((truth=="HPT" & prediction=="LPT") | (truth=="LPT" & prediction=="HPT")) |>
    mutate(direction = paste0(truth, "→", prediction))
  p17b <- ggplot(mix, aes(direction, value, fill = direction)) + geom_col(width = 0.65, show.legend = FALSE) +
    geom_text(aes(label = percent(value, accuracy = 0.1)), vjust = -0.35) +
    scale_y_continuous(limits = c(0, max(mix$value)*1.45), labels = percent) +
    labs(x = NULL, y = "跨类共现率", title = "(b) 涡轮内相互混叠") + theme_thesis()
  p17 <- (p17a | p17b) + plot_annotation(title = "HPT/LPT 细分的可辨识性深度分析")
  save_thesis(p17, out, "图3-17_涡轮内部细分边界", 9.5, 4.5)

  # 图3-18：四部件合并收益
  part <- read_csv(part_metrics_path, show_col_types = FALSE)
  fam <- read_csv(family_metrics_path, show_col_types = FALSE)
  comp <- tibble(
    component = c("风扇/Fan", "高压压气机/HPC", "低压压气机/LPC", "涡轮"),
    `五部件细分` = c(fam$F1[fam$name=="Fan"], fam$F1[fam$name=="HPC"], fam$F1[fam$name=="LPC"],
                       mean(fam$F1[fam$name %in% c("HPT","LPT")])),
    `四部件主诊断` = c(part$F1[part$name=="风扇"], part$F1[part$name=="高压压气机"],
                         part$F1[part$name=="低压压气机"], part$F1[part$name=="涡轮"])
  ) |> pivot_longer(-component, names_to = "level", values_to = "F1")
  p18 <- ggplot(comp, aes(component, F1, fill = level)) + geom_col(position = "dodge") +
    scale_fill_manual(values = c("五部件细分"="#9ECAE1", "四部件主诊断"="#D62728")) +
    scale_y_continuous(limits = c(0.72, 1.02)) +
    labs(x = NULL, y = "F1", title = "诊断粒度与可辨识性匹配带来的性能收益",
         subtitle = sprintf("macro-F1：%.3f → %.3f", mean(fam$F1), mean(part$F1))) + theme_thesis()
  save_thesis(p18, out, "图3-18_四部件诊断粒度收益", 8.6, 4.6)

  message("DTAE 正式图生成完成：", out)
  invisible(out)
}

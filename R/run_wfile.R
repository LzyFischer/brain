# ========= 配置 =========
xlsx_path <- "glasser.xlsx"
sheet     <- 1
set.seed(42)  # 方便复现，想每次都不一样就删掉这一行

# ========= 依赖 =========
suppressPackageStartupMessages({
  library(readxl)
  library(readr)
  library(dplyr)
  library(stringr)
  library(tidyr)
  library(brainconn)
})

# ========= 读取与清洗 =========
df <- read_xlsx(xlsx_path, sheet = sheet, trim_ws = TRUE)

# 去掉无名索引列（...1 之类）
idx_cols <- grep("^\\.\\.\\.[0-9]+$", names(df), value = TRUE)
if (length(idx_cols)) df <- dplyr::select(df, -all_of(idx_cols))

# 识别列
roi_col <- dplyr::first(intersect(names(df), c("ROI.x","Name.x","roi","ROI","Region","regionName")))
if (is.na(roi_col)) stop("找不到 ROI 名称列（尝试了 ROI.x/Name.x/roi/ROI/Region/regionName）")

net_col <- dplyr::first(intersect(names(df), c("Network","NETWORK","network")))
if (is.na(net_col)) stop("找不到网络列（Network/NETWORK/network）")

need_xyz <- c("x.mni","y.mni","z.mni")
miss_xyz <- setdiff(need_xyz, names(df))
if (length(miss_xyz)) stop("缺少坐标列: ", paste(miss_xyz, collapse = ", "))

# 安全解析坐标（提取数值，兼容全角负号）
fix_minus <- function(x) chartr("\u2212", "-", as.character(x))
num3 <- df %>%
  transmute(
    x.mni = readr::parse_number(fix_minus(`x.mni`)),
    y.mni = readr::parse_number(fix_minus(`y.mni`)),
    z.mni = readr::parse_number(fix_minus(`z.mni`))
  )

atlas <- tibble::tibble(
  ROI.Name = as.character(df[[roi_col]]),
  x.mni = num3$x.mni,
  y.mni = num3$y.mni,
  z.mni = num3$z.mni,
  network = as.character(df[[net_col]])
)

normalize_key <- function(x) {
  x <- tolower(x %||% "")
  x <- stringr::str_replace_all(x, "[^a-z]+", " ")
  stringr::str_squish(x)
}

# 组合 ROI 主名与长名做匹配键
raw_key <- paste0(
  as.character(atlas$ROI.Name), " ",
  if ("Name.x" %in% names(df)) as.character(df$Name.x) else ""
)
atlas$key <- normalize_key(raw_key)

manual_tbl <- tibble::tribble(
  ~key,                   ~x.mni, ~y.mni, ~z.mni,
  "accumbens left",         -10,     12,     -7,
  "accumbens right",         10,     12,     -7,
  "amygdala left",          -23,     -1,    -17,
  "amygdala right",          27,      1,    -18,
  "caudate left",           -11,     11,      9,
  "caudate right",           15,     12,      9,
  "hippocampus left",       -25,    -21,    -10,
  "hippocampus right",       29,    -20,    -10,
  "pallidum left",          -18,      0,      0,
  "pallidum right",          21,      0,      0,
  "putamen left",           -24,      4,      2,
  "putamen right",           28,      5,      2,
  "thalamus left",          -11,    -18,      8,
  "thalamus right",          13,    -18,      8,
  # 小脑：这里给的是常用的 Cerebellum_6 代表点（整半球建议自己从 mask 取质心）
  "cerebellum left",        -22,    -59,    -22,
  "cerebellum right",        26,    -58,    -24,
  # 间脑：常用做法是 Ventral Diencephalon 的近似点（如果你有自己的 aseg/mask，建议用其质心替换）
  "diencephalon left",       -7,    -15,     -6,
  "diencephalon right",       7,    -15,     -6
) %>% dplyr::mutate(key = normalize_key(key))

# 左连接并仅对 NA 的坐标进行补全
atlas <- atlas %>%
  dplyr::left_join(manual_tbl, by = "key", suffix = c("", ".manual")) %>%
  dplyr::mutate(
    x.mni = ifelse(is.na(x.mni) & !is.na(x.mni.manual), x.mni.manual, x.mni),
    y.mni = ifelse(is.na(y.mni) & !is.na(y.mni.manual), y.mni.manual, y.mni),
    z.mni = ifelse(is.na(z.mni) & !is.na(z.mni.manual), z.mni.manual, z.mni)
  ) %>%
  dplyr::select(-dplyr::ends_with(".manual"), -key)

# ========= 半球推断 =========
infer_hemi <- function(roi_name, long_name = NA, x = NA_real_) {
  nm <- paste0(roi_name, " ", long_name)
  if (str_detect(roi_name, "^L[_\\- ]")) return("L")
  if (str_detect(roi_name, "^R[_\\- ]")) return("R")
  if (str_detect(nm, regex("\\bLeft\\b", ignore_case = TRUE))) return("L")
  if (str_detect(nm, regex("\\bRight\\b", ignore_case = TRUE))) return("R")
  if (!is.na(x)) {
    if (x < 0) return("L")
    if (x > 0) return("R")
  }
  return(NA_character_)
}

atlas$hemi <- mapply(infer_hemi,
                     roi_name = atlas$ROI.Name,
                     long_name = if ("Name.x" %in% names(df)) df$Name.x else NA,
                     x = atlas$x.mni)

# ========= 生成用于插补的统计量 =========
# 仅用完整行计算分布
complete_rows <- atlas %>% filter(!is.na(x.mni) & !is.na(y.mni) & !is.na(z.mni))
# 组内（network+hemi）
stats_g <- complete_rows %>%
  mutate(group = paste0(network, "||", hemi)) %>%
  group_by(group) %>%
  summarise(
    mx = mean(x.mni), sx = sd(x.mni),
    my = mean(y.mni), sy = sd(y.mni),
    mz = mean(z.mni), sz = sd(z.mni),
    n = dplyr::n(), .groups = "drop"
  )

# 半球层面
stats_h <- complete_rows %>%
  group_by(hemi) %>%
  summarise(
    mx = mean(x.mni), sx = sd(x.mni),
    my = mean(y.mni), sy = sd(y.mni),
    mz = mean(z.mni), sz = sd(z.mni),
    n = dplyr::n(), .groups = "drop"
  )

# 全局
stats_all <- complete_rows %>%
  summarise(
    mx = mean(x.mni), sx = sd(x.mni),
    my = mean(y.mni), sy = sd(y.mni),
    mz = mean(z.mni), sz = sd(z.mni),
    n = dplyr::n()
  )

# 合理范围（大致的人脑 MNI 边界，保守裁剪）
clip_mni <- function(x, lo, hi) pmax(lo, pmin(hi, x))
bounds <- list(x = c(-80, 80), y = c(-120, 80), z = c(-60, 90))

# 默认均值/方差（当啥统计都没有时）
default_mu <- function(hemi) {
  c(
    x = ifelse(identical(hemi, "L"), -40, ifelse(identical(hemi, "R"), 40, 0)),
    y = -20,
    z = 30
  )
}
default_sd <- c(x = 15, y = 15, z = 15)

# ========= 插补 NA =========
need_imp <- which(is.na(atlas$x.mni) | is.na(atlas$y.mni) | is.na(atlas$z.mni))
imputed_rows <- list()

for (i in need_imp) {
  hemi_i <- atlas$hemi[i]
  grp_i  <- paste0(atlas$network[i], "||", hemi_i)

  # 1) 组内分布
  st_g <- stats_g %>% filter(group == grp_i)
  # 2) 半球分布
  st_h <- stats_h %>% filter(hemi == hemi_i)
  # 3) 全局分布
  st_a <- stats_all

  # 选用的均值与标准差（优先组内，其次半球，最后全局/默认）
  pick_mu_sd <- function(comp, hemi) {
    if (nrow(comp) && comp$n[1] >= 3) {
      mu <- c(x = comp$mx[1], y = comp$my[1], z = comp$mz[1])
      sd <- c(x = comp$sx[1], y = comp$sy[1], z = comp$sz[1])
    } else if (nrow(st_h) && st_h$n[1] >= 3) {
      mu <- c(x = st_h$mx[1], y = st_h$my[1], z = st_h$mz[1])
      sd <- c(x = st_h$sx[1], y = st_h$sy[1], z = st_h$sz[1])
    } else if (!is.na(st_a$n[1]) && st_a$n[1] >= 3) {
      mu <- c(x = st_a$mx[1], y = st_a$my[1], z = st_a$mz[1])
      sd <- c(x = st_a$sx[1], y = st_a$sy[1], z = st_a$sz[1])
    } else {
      mu <- default_mu(hemi)
      sd <- default_sd
    }
    # 防止 sd 为 0 或 NA
    sd[is.na(sd) | sd == 0] <- default_sd[is.na(sd) | sd == 0]
    list(mu = mu, sd = sd)
  }

  ps <- pick_mu_sd(st_g, hemi_i)

  # 只对缺失的坐标采样
  xi <- atlas$x.mni[i]; yi <- atlas$y.mni[i]; zi <- atlas$z.mni[i]
  if (is.na(xi)) xi <- rnorm(1, mean = ps$mu["x"], sd = ps$sd["x"])
  if (is.na(yi)) yi <- rnorm(1, mean = ps$mu["y"], sd = ps$sd["y"])
  if (is.na(zi)) zi <- rnorm(1, mean = ps$mu["z"], sd = ps$sd["z"])

  # 半球约束：左负右正（若能判断半球）
  if (!is.na(hemi_i)) {
    if (hemi_i == "L" && xi > 0) xi <- -abs(xi)
    if (hemi_i == "R" && xi < 0) xi <-  abs(xi)
  }

  # 裁剪到合理范围
  xi <- clip_mni(xi, bounds$x[1], bounds$x[2])
  yi <- clip_mni(yi, bounds$y[1], bounds$y[2])
  zi <- clip_mni(zi, bounds$z[1], bounds$z[2])

  # 记录并写回
  imputed_rows[[length(imputed_rows)+1]] <- data.frame(
    row = i,
    ROI.Name = atlas$ROI.Name[i],
    hemi = hemi_i,
    network = atlas$network[i],
    x.mni = xi, y.mni = yi, z.mni = zi
  )
  atlas$x.mni[i] <- xi
  atlas$y.mni[i] <- yi
  atlas$z.mni[i] <- zi
}

# 展示被插补的行
if (length(imputed_rows)) {
  cat("\n=== 以下行的坐标被随机合理插补（仅供可视化） ===\n")
  print(dplyr::bind_rows(imputed_rows) %>% arrange(row) %>% select(-row))
} else {
  cat("\n没有需要插补的坐标。\n")
}

# 坐标四舍五入并转为整数（满足 brainconn 要求）
atlas <- atlas %>%
  dplyr::mutate(across(c(x.mni, y.mni, z.mni), ~ as.integer(round(.x))))

# 可选：再确认一下类型
str(atlas[c("x.mni","y.mni","z.mni")])




# # ====== 固定“网络 -> 颜色”的映射（第一次跑时生成并保存）======
# suppressPackageStartupMessages({ library(ggplot2) })

# # 用 atlas 里的 network 列建立有序水平（确保可重复）
# net_levels <- sort(unique(as.character(atlas$network)))
# # 选一个稳定的定性调色板；hcl.colors 对同样的 n 恒定
# net_cols <- setNames(grDevices::hcl.colors(length(net_levels), palette = "Dark 3"),
#                      net_levels)

# # 建议保存，今后所有 npz 都加载这份映射，保证跨会话一致
# saveRDS(net_cols, "glasser_network_colors.rds")



# ======== 画图 ========
suppressPackageStartupMessages({
  library(reticulate)
  library(brainconn)
})

# ==== 参数 ====
npz_path <- "/scratch/vjd5zr/project/BrainProject/brain/results/attention_weights/disease_Anx_nosf.npz"
# SC_FC_att_selectedlabel
target_n <- 378       # 只取前 360×360
pct_keep <- 0.0005    # 0.05% 的边
mode_sel <- "positive"    # 或 "abs"（按绝对值筛）

suppressPackageStartupMessages({
  library(reticulate)
  library(brainconn)
})

# 载入 npz -> R 数组
np <- import("numpy", convert = TRUE)
npz <- np$load(npz_path, allow_pickle = TRUE)
arr <- npz[["att_FCs"]]

# 若仍是 Python 对象，转成 R
if (is.null(dim(arr))) arr <- py_to_r(arr)

# 平均掉最后两轴之外的所有轴，得到 2D 矩阵 C
d <- dim(arr)
if (length(d) < 2) stop("att_FCs 维度不对，期望至少是 2D")
if (length(d) == 2) {
  C <- arr
} else {
  # 保留最后两轴，沿其余轴取均值
  keep_axes <- (length(d)-1):length(d)
  C <- apply(arr, keep_axes, mean, na.rm = TRUE)
}

# 截取到 target_n × target_n
n <- min(target_n, nrow(C), ncol(C))
C <- C[1:n, 1:n, drop = FALSE]

# 对称化&去对角
C <- (C + t(C)) / 2
diag(C) <- 0

# ==== 取前 0.05% 的边 ====
U <- upper.tri(C)
w <- C[U]
w_rank <- if (mode_sel == "abs") abs(w) else pmax(w, 0)  # 仅正权或按绝对值
m <- length(w_rank)
k <- max(1, round(m * pct_keep, 0))

ord <- order(w_rank, decreasing = TRUE, na.last = NA)
sel <- ord[seq_len(k)]

A <- matrix(0, n, n)
U_idx <- which(U, arr.ind = TRUE)
A[ U_idx[sel, , drop = FALSE] ] <- w[sel]     # 保留原始权重（含符号）
A <- A + t(A)

# 归一到 [0,1]（便于可视化）
mx <- max(A)
if (mx > 0) A <- A / mx

cat(sprintf("att_FCs dims: %s -> C: %dx%d | kept top %d of %d edges (%.3f%%)\n",
            paste(d, collapse="x"), n, n, k, m, 100*pct_keep))

# ==== 尺寸/atlas 检查并画图 ====
# ===== Export top connection names (and metadata) =====
out_csv <- "top_connections.csv"           # full table
out_txt <- "top_connection_names.txt"      # names only (optional)

# Use the existing selection you computed: U_idx, sel, w, w_rank, A
pairs <- U_idx[sel, , drop = FALSE]  # (i, j) for top edges, in ranked order

edges <- tibble::tibble(
  rank = seq_len(nrow(pairs)),
  i = pairs[, 1],
  j = pairs[, 2],
  ROI1 = as.character(atlas$ROI.Name[pairs[, 1]]),
  ROI2 = as.character(atlas$ROI.Name[pairs[, 2]]),
  hemi1 = atlas$hemi[pairs[, 1]],
  hemi2 = atlas$hemi[pairs[, 2]],
  network1 = as.character(atlas$network[pairs[, 1]]),
  network2 = as.character(atlas$network[pairs[, 2]]),
  weight_raw  = w[sel],                       # original (symmetrized C upper-tri) weight
  weight_rank = w_rank[sel],                  # the metric used for ranking (abs or positive)
  weight_norm = A[cbind(pairs[, 1], pairs[, 2])]  # normalized to [0,1]
) %>%
  dplyr::mutate(connection = paste0(ROI1, " -- ", ROI2)) %>%
  dplyr::relocate(connection, .after = rank)

# Write CSV (full info)
readr::write_csv(edges, out_csv)

# Optionally, also write a simple list of connection names
writeLines(edges$connection, out_txt)

cat(sprintf("Wrote %d top connections to %s\n", nrow(edges), out_csv))
cat(sprintf("Also saved names-only list to %s\n", out_txt))



# ===== Hub node metrics & export =====
suppressPackageStartupMessages({ library(igraph); library(dplyr); library(readr) })

# 用阈后加权无向图（A>0 有边，权重=A_ij）
g <- igraph::graph_from_adjacency_matrix(A, mode = "undirected",
                                         weighted = TRUE, diag = FALSE)

# 1) 基本中心性（注意：加权最短路把“权重大=连通强”转成“距离小”）
w <- E(g)$weight
eps <- .Machine$double.eps
dist_w <- 1 / pmax(w, eps)   # 距离 = 1/权重

deg_bin    <- igraph::degree(g, mode = "all")
strength_w <- igraph::strength(g, vids = V(g), mode = "all", weights = w)
eig        <- igraph::eigen_centrality(g, weights = w, directed = FALSE)$vector
btw        <- igraph::betweenness(g, directed = FALSE, weights = dist_w, normalized = TRUE)
clo        <- igraph::closeness(g, normalized = TRUE, weights = dist_w)

# 2) Participation coefficient (P) 与 within-module degree z
modules <- as.character(atlas$network)
modules[is.na(modules) | modules == ""] <- "Unknown"
mod_levels <- unique(modules)

S <- A
k_tot <- rowSums(S)
# 各节点连到每个模块的加权强度
Kis <- matrix(0, nrow(S), length(mod_levels), dimnames = list(NULL, mod_levels))
for (m in mod_levels) {
  idx <- which(modules == m)
  if (length(idx)) Kis[, m] <- rowSums(S[, idx, drop = FALSE])
}

# Participation coefficient
pc <- rep(NA_real_, nrow(S))
nz <- which(k_tot > 0)
pc[nz] <- 1 - rowSums( (Kis[nz, , drop = FALSE] / k_tot[nz])^2 )

# Within-module degree z（加权版）
k_within <- numeric(nrow(S)); z_within <- rep(NA_real_, nrow(S))
for (m in mod_levels) {
  idx <- which(modules == m)
  if (length(idx) >= 2) {
    kw  <- rowSums(S[idx, idx, drop = FALSE])           # 节点在本模块内的强度
    mu  <- mean(kw); sdv <- stats::sd(kw)
    z   <- if (sdv > 0) (kw - mu) / sdv else rep(0, length(kw))
    k_within[idx] <- kw
    z_within[idx] <- z
  } else if (length(idx) == 1) {
    k_within[idx] <- 0
    z_within[idx] <- 0
  }
}

# 3) 组合成表 & 计算综合 hub 分数（标准化后求均值）
zscore <- function(x) {
  x <- as.numeric(x); m <- mean(x, na.rm = TRUE); s <- stats::sd(x, na.rm = TRUE)
  if (is.na(s) || s == 0) return(rep(0, length(x)))
  (x - m) / s
}

df_nodes <- tibble::tibble(
  node       = seq_len(nrow(A)),
  ROI        = as.character(atlas$ROI.Name),
  hemi       = atlas$hemi,
  network    = modules,
  degree_bin = deg_bin,
  strength   = strength_w,
  eigenvector = eig,
  betweenness = btw,
  closeness   = clo,
  part_coeff  = pc,
  z_within    = z_within
) %>%
  mutate(
    z_strength = zscore(strength),
    z_eigen    = zscore(eigenvector),
    z_betw     = zscore(betweenness),
    # 参与系数通常用来识别“connector hubs”，不直接进综合分，可单独参考；
    # 若你想纳入综合分，可把 zscore(part_coeff) 加进去。
    hub_score  = rowMeans(cbind(z_strength, z_eigen, z_betw, z_within), na.rm = TRUE)
  ) %>%
  arrange(desc(hub_score))

# 4) 导出
top_n <- 20
top_hubs <- df_nodes %>% slice_head(n = min(top_n, nrow(.)))

readr::write_csv(df_nodes,  "node_hub_metrics.csv")   # 全部节点
readr::write_csv(top_hubs, "top_hubs.csv")            # Top-N
writeLines(top_hubs$ROI,   "top_hub_labels.txt")      # 只要名字

cat(sprintf("Computed hub metrics for %d nodes. Saved node_hub_metrics.csv and top_hubs.csv (top %d in top_hubs.csv)\n",
            nrow(df_nodes), nrow(top_hubs)))

# stopifnot(nrow(atlas) == n)
check_atlas(atlas)
# brainconn(atlas = atlas, conmat = A, node.size = 7)

# ====== 每次绘图前（加载固定映射）======
net_cols <- readRDS("glasser_network_colors.rds")

# 把 atlas$network 锁成固定水平，避免因缺少/增多某些网络导致颜色错位
atlas$network <- factor(as.character(atlas$network), levels = names(net_cols))

# 你的 A 矩阵按原逻辑算好以后：
p <- brainconn(atlas = atlas,
               conmat = A,
               node.size = 7,
               node.color = "network",  # 按 network 着色
               show.legend = TRUE)

# 用同一映射固定颜色（同时管住 colour 和 fill，稳妥）
p <- p +
  ggplot2::scale_color_manual(values = net_cols, limits = names(net_cols), drop = FALSE) +
  ggplot2::scale_fill_manual(values = net_cols, limits = names(net_cols), drop = FALSE)

print(p)





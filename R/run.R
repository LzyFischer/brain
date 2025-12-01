# 必要包
if (!requireNamespace("brainconn", quietly = TRUE)) {
  # 先尝试 CRAN
  install.packages("brainconn")
  # 如果报“找不到包”或不在 CRAN，请按 brainconn 官方仓库说明用 remotes::install_github("作者/仓库")
}
library(brainconn)
library(readxl)
library(readr)  # 你代码里用到了 read_csv / read_table2

# ---- 读入数据 ----
CAB_NP_v1_1_Labels_ReorderedbyNetworks <- read_excel(
  "glasser_roi2aal.xlsx"
)

glassermni <- read_csv("/Users/aiying/Documents/CCA/glassermni.txt")
m_g <- nrow(glassermni)

# readr 2.0+ 里 read_table2() 被软弃用；若你的版本没有它，就用 read_table()
read_table2_safe <- if (exists("read_table2")) read_table2 else read_table

glasser <- read_table2_safe(
  "/Users/aiying/Documents/CCA/glasser.node",
  col_names = FALSE
)

# ---- 组装 atlas ----
glassertemp <- data.frame(
  ROI.Name = glassermni$regionName,
  x.mni = as.integer(round(glasser$X1)),
  y.mni = as.integer(round(glasser$X2)),
  z.mni = as.integer(round(glasser$X3)),
  network = CAB_NP_v1_1_Labels_ReorderedbyNetworks$NETWORK[1:m_g]
)

# 基本完整性检查（可选但建议）
check_atlas(glassertemp)

# ---- 准备你的连通性矩阵 ----
# 你原代码里用到了 com_edge / x_hub / z_hub，这里只是示例占位：
# 实际使用时请把这三行替换为读取你自己的矩阵（大小必须是 m_g x m_g）
# 例如：com_edge <- as.matrix(read_csv(".../com_edge.csv", col_names = FALSE))
set.seed(1)
symm <- function(M) { M[lower.tri(M)] <- t(M)[lower.tri(M)]; diag(M) <- 0; M }
com_edge <- symm(matrix(runif(m_g*m_g), m_g, m_g))
x_hub   <- symm(matrix(runif(m_g*m_g), m_g, m_g))
z_hub   <- symm(matrix(runif(m_g*m_g), m_g, m_g))

# ---- 绘图/输出 ----
brainconn(atlas = glassertemp, conmat = com_edge)
brainconn(atlas = glassertemp, conmat = x_hub)
brainconn(atlas = glassertemp, conmat = z_hub)
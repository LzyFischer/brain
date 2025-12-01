# ======== 配置：把路径改成你的 xlsx 路径 ========
xlsx_path <- "glasser_roi2aal.xlsx"
sheet     <- 1   # 如果不是第一个工作表，请改成相应 sheet 名或序号

# ======== 包 ========
if (!requireNamespace("readxl", quietly = TRUE)) install.packages("readxl")
if (!requireNamespace("brainconn", quietly = TRUE)) install.packages("brainconn")
suppressPackageStartupMessages({
  library(readxl)
  library(brainconn)
})

# ======== 读取 xlsx 并映射到 brainconn 需要的列名 ========
df <- read_xlsx(xlsx_path, sheet = sheet)

# 选择一个 ROI 名称列：优先用短名 ROI.x，其次 Name.x，再次 roi（数值则转成字符串）
roi_col <- intersect(names(df), c("ROI.x","Name.x","roi"))
if (length(roi_col) == 0) stop("在表里找不到 ROI 名称列（期望 ROI.x / Name.x / roi）")
roi_col <- roi_col[1]

roi_name <- df[[roi_col]]
if (roi_col == "roi") roi_name <- as.character(roi_name)  # 数值索引转字符以避免重复/混淆

# 必需坐标列检查
need_xyz <- c("x.mni","y.mni","z.mni")
miss_xyz <- setdiff(need_xyz, names(df))
if (length(miss_xyz)) stop("缺少坐标列: ", paste(miss_xyz, collapse = ", "))

# 网络列（大小写都兼容）
net_col <- intersect(names(df), c("Network","NETWORK","network"))
if (length(net_col) == 0) stop("在表里找不到网络列（期望 Network/NETWORK/network）")
net_col <- net_col[1]

# 组装成 brainconn 规范的 atlas 数据框
atlas <- data.frame(
  ROI.Name = roi_name,
  x.mni = as.integer(round(as.numeric(df[["x.mni"]]))),
  y.mni = as.integer(round(as.numeric(df[["y.mni"]]))),
  z.mni = as.integer(round(as.numeric(df[["z.mni"]]))),
  network = as.character(df[[net_col]]),
  stringsAsFactors = FALSE
)

# 基本质量检查
if (any(is.na(atlas$ROI.Name))) stop("ROI.Name 里有 NA")
if (any(is.na(atlas$network))) stop("network 里有 NA（请检查 Network 列）")
if (any(is.na(atlas$x.mni) | is.na(atlas$y.mni) | is.na(atlas$z.mni)))
  stop("坐标列里有 NA，请检查 x.mni/y.mni/z.mni")

# 可选：检查是否有重复 ROI 名（不一定是错误，但可能意味着短名/长名混用）
if (anyDuplicated(atlas$ROI.Name)) {
  dupn <- unique(atlas$ROI.Name[duplicated(atlas$ROI.Name)])
  warning("发现重复 ROI 名：", paste(dupn, collapse = ", "))
}

# 让 brainconn 先检验一遍
check_atlas(atlas)

# ======== 读取/准备你的连通性矩阵 ========
# 你已有 com_edge / x_hub / z_hub 就直接读；示例写法如下（去掉下面的示例矩阵）：
# com_edge <- as.matrix(read.csv("/path/to/com_edge.csv", header = FALSE))
# x_hub   <- as.matrix(read.csv("/path/to/x_hub.csv",   header = FALSE))
# z_hub   <- as.matrix(read.csv("/path/to/z_hub.csv",   header = FALSE))

# --- 占位示例：用随机对称矩阵跑通流程（跑正式图时请删掉这段） ---
set.seed(1)
m_g <- nrow(atlas)
symm <- function(M){ M[lower.tri(M)] <- t(M)[lower.tri(M)]; diag(M) <- 0; M }
com_edge <- symm(matrix(runif(m_g*m_g), m_g, m_g))
x_hub   <- symm(matrix(runif(m_g*m_g), m_g, m_g))
z_hub   <- symm(matrix(runif(m_g*m_g), m_g, m_g))
# --------------------------------------------------------------------

# 如果你的矩阵自带行列名，并且和 ROI.Name 对得上，建议按名字对齐一下
align_to_mat <- function(atlas, mat) {
  rn <- rownames(mat); cn <- colnames(mat)
  if (!is.null(rn) && !is.null(cn) && identical(rn, cn)) {
    idx <- match(rn, atlas$ROI.Name)
    if (anyNA(idx)) stop("矩阵的行名有在 atlas$ROI.Name 里找不到的项")
    atlas[idx, , drop = FALSE]
  } else {
    if (nrow(mat) != nrow(atlas)) stop("矩阵大小与 atlas 行数不一致，且没有可用的行列名对齐")
    atlas
  }
}

atlas_ce <- align_to_mat(atlas, com_edge)
atlas_xh <- align_to_mat(atlas, x_hub)
atlas_zh <- align_to_mat(atlas, z_hub)

# ======== 作图 ========
brainconn(atlas = atlas_ce, conmat = com_edge)
brainconn(atlas = atlas_xh, conmat = x_hub)
brainconn(atlas = atlas_zh, conmat = z_hub)
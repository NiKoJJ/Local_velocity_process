#!/bin/bash
#===============================================================================
# ITS_LIVE NC 文件批量下载脚本 (aria2 超高速版)
# aria2 支持多线程、断点续传，速度更快
# 用法：./download_nc_aria2.sh [url列表文件] [输出目录]
#===============================================================================

# 配置
URL_FILE="${1:-nc_url.txt}"
OUTPUT_DIR="${2:-./nc_downloads}"
CONNECTIONS_PER_FILE=4        # 每个文件连接数
MAX_CONCURRENT=4              # 最大并发下载数
MIN_SPEED=1K                  # 最小速度限制

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

# 检查 aria2
if ! command -v aria2c &> /dev/null; then
    echo -e "${RED}[错误] aria2 未安装${NC}"
    echo -e "  安装命令：${YELLOW}sudo apt install aria2${NC} 或 ${YELLOW}conda install -c conda-forge aria2${NC}"
    exit 1
fi

# 检查文件
if [ ! -f "$URL_FILE" ]; then
    echo -e "${RED}[错误] 文件不存在：$URL_FILE${NC}"
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

echo -e "${BLUE}===============================================================${NC}"
echo -e "${BLUE}  ITS_LIVE NC 文件批量下载 (aria2 高速版)${NC}"
echo -e "${BLUE}===============================================================${NC}"
echo -e "  URL 列表   : $URL_FILE"
echo -e "  输出目录   : $(realpath "$OUTPUT_DIR")"
echo -e "  并发连接   : $CONNECTIONS_PER_FILE / 文件"
echo -e "  最大并发   : $MAX_CONCURRENT 文件"
echo -e "${BLUE}===============================================================${NC}"
echo ""

# 开始下载
aria2c \
    -i "$URL_FILE" \
    -d "$OUTPUT_DIR" \
    -x "$CONNECTIONS_PER_FILE" \
    -s "$CONNECTIONS_PER_FILE" \
    -j "$MAX_CONCURRENT" \
    -k 1M \
    --min-split-size=1M \
    --max-tries=2 \
    --retry-wait=2 \
    --timeout=120 \
    --continue=true \
    --console-log-level=warn \
    --summary-interval=1 \
    --check-integrity=true

# 统计
echo ""
echo -e "${BLUE}===============================================================${NC}"
echo -e "${GREEN}  下载完成！${NC}"
echo -e "${BLUE}===============================================================${NC}"
echo -e "  输出目录   : $(realpath "$OUTPUT_DIR")"
echo -e "  NC 文件数  : $(ls -1 "$OUTPUT_DIR"/*.nc 2>/dev/null | wc -l)"
echo -e "${BLUE}===============================================================${NC}"

# 检查失败
if [ -f "$OUTPUT_DIR/aria2_session.txt" ]; then
    echo -e "${YELLOW}  会话文件   : $OUTPUT_DIR/aria2_session.txt${NC}"
fi
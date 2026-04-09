#!/bin/bash
#===============================================================================
# ITS_LIVE NC 文件批量下载脚本 (并发加速版)
# 用法：./download_nc_parallel.sh [url列表文件] [输出目录] [并发数]
#===============================================================================

# 配置
URL_FILE="${1:-nc_url.txt}"
OUTPUT_DIR="${2:-./nc_downloads}"
MAX_JOBS="${3:-4}"                    # 并发下载数
MAX_RETRIES=2
TIMEOUT=120

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 检查依赖
if ! command -v wget &> /dev/null; then
    echo -e "${RED}[错误] wget 未安装${NC}"
    exit 1
fi

# 检查文件
if [ ! -f "$URL_FILE" ]; then
    echo -e "${RED}[错误] 文件不存在：$URL_FILE${NC}"
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"
cd "$OUTPUT_DIR" || exit 1

# 统计
total=$(grep -v '^#' "$URL_FILE" | grep -v '^$' | wc -l)
echo -e "${BLUE}===============================================================${NC}"
echo -e "${BLUE}  ITS_LIVE NC 文件批量下载 (并发版)${NC}"
echo -e "${BLUE}===============================================================${NC}"
echo -e "  URL 列表   : $URL_FILE"
echo -e "  输出目录   : $(pwd)"
echo -e "  文件总数   : $total"
echo -e "  并发数     : $MAX_JOBS"
echo -e "${BLUE}===============================================================${NC}"
echo ""

# 下载函数
download_file() {
    local url="$1"
    local filename=$(basename "$url")
    
    # 跳过已存在文件
    if [ -f "$filename" ]; then
        echo -e "${YELLOW}[跳过] $filename${NC}"
        return 2
    fi
    
    # 下载
    local attempt=1
    while [ $attempt -le $MAX_RETRIES ]; do
        if wget -q --timeout=$TIMEOUT --tries=1 -c "$url" -O "$filename" 2>/dev/null; then
            echo -e "${GREEN}[成功] $filename${NC}"
            return 0
        fi
        ((attempt++))
        sleep 1
    done
    
    echo -e "${RED}[失败] $filename${NC}"
    echo "$url" >> "../failed_urls.txt"
    return 1
}

export -f download_file
export MAX_RETRIES TIMEOUT RED GREEN YELLOW NC

# 开始下载 (并发)
start_time=$(date +%s)

grep -v '^#' "$URL_FILE" | grep -v '^$' | \
    xargs -P "$MAX_JOBS" -I {} bash -c 'download_file "$@"' _ {}

# 统计结果
end_time=$(date +%s)
duration=$((end_time - start_time))
success=$(ls -1 *.nc 2>/dev/null | wc -l)
failed=$(wc -l < "../failed_urls.txt" 2>/dev/null || echo 0)
skipped=$((total - success - failed))

echo ""
echo -e "${BLUE}===============================================================${NC}"
echo -e "${GREEN}  下载完成！${NC}"
echo -e "${BLUE}===============================================================${NC}"
echo -e "  成功       : ${GREEN}$success${NC}"
echo -e "  失败       : ${RED}$failed${NC}"
echo -e "  跳过       : ${YELLOW}$skipped${NC}"
echo -e "  总耗时     : $((duration/60)) 分 $((duration%60)) 秒"
echo -e "  平均速度   : $((total * 60 / (duration + 1))) 秒/文件"
echo -e "${BLUE}===============================================================${NC}"

[ $failed -gt 0 ] && echo -e "${RED}  失败列表   : ../failed_urls.txt${NC}"
exit 0
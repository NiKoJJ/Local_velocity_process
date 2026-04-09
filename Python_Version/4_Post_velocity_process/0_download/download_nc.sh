#!/bin/bash
#===============================================================================
# ITS_LIVE NC 文件批量下载脚本 (基础版)
# 用法：./download_nc.sh [url列表文件] [输出目录]
#===============================================================================

# 配置
URL_FILE="${1:-nc_url.txt}"           # URL 列表文件
OUTPUT_DIR="${2:-./nc_downloads}"     # 输出目录
MAX_RETRIES=3                         # 最大重试次数
TIMEOUT=120                           # 超时时间 (秒)

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 检查文件
if [ ! -f "$URL_FILE" ]; then
    echo -e "${RED}[错误] 文件不存在：$URL_FILE${NC}"
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"
cd "$OUTPUT_DIR" || exit 1

# 统计
total=$(wc -l < "$URL_FILE")
success=0
failed=0
skipped=0

echo -e "${BLUE}===============================================================${NC}"
echo -e "${BLUE}  ITS_LIVE NC 文件批量下载${NC}"
echo -e "${BLUE}===============================================================${NC}"
echo -e "  URL 列表   : $URL_FILE"
echo -e "  输出目录   : $(pwd)"
echo -e "  文件总数   : $total"
echo -e "  最大重试   : $MAX_RETRIES"
echo -e "${BLUE}===============================================================${NC}"
echo ""

# 开始下载
start_time=$(date +%s)

while IFS= read -r url || [ -n "$url" ]; do
    # 跳过空行和注释
    [[ -z "$url" || "$url" =~ ^# ]] && continue
    
    # 提取文件名
    filename=$(basename "$url")
    
    # 检查是否已存在
    if [ -f "$filename" ]; then
        echo -e "${YELLOW}[跳过] $filename (已存在)${NC}"
        ((skipped++))
        continue
    fi
    
    # 下载
    attempt=1
    while [ $attempt -le $MAX_RETRIES ]; do
        echo -e "${BLUE}[下载] [$((success+failed+skipped+1))/$total] $filename (尝试 $attempt/$MAX_RETRIES)${NC}"
        
        if wget -q --show-progress --timeout=$TIMEOUT --tries=1 -c "$url" -O "$filename"; then
            echo -e "${GREEN}[成功] $filename${NC}"
            ((success++))
            break
        else
            echo -e "${RED}[失败] $filename (尝试 $attempt)${NC}"
            ((attempt++))
            [ $attempt -le $MAX_RETRIES ] && sleep 2
        fi
    done
    
    # 超过最大重试次数
    if [ $attempt -gt $MAX_RETRIES ]; then
        ((failed++))
        echo "$url" >> "../failed_urls.txt"
    fi
    
done < "$URL_FILE"

# 结束统计
end_time=$(date +%s)
duration=$((end_time - start_time))

echo ""
echo -e "${BLUE}===============================================================${NC}"
echo -e "${GREEN}  下载完成！${NC}"
echo -e "${BLUE}===============================================================${NC}"
echo -e "  成功       : ${GREEN}$success${NC}"
echo -e "  失败       : ${RED}$failed${NC}"
echo -e "  跳过       : ${YELLOW}$skipped${NC}"
echo -e "  总耗时     : $((duration/60)) 分 $((duration%60)) 秒"
echo -e "  输出目录   : $(pwd)"
[ $failed -gt 0 ] && echo -e "  失败列表   : ../failed_urls.txt"
echo -e "${BLUE}===============================================================${NC}"

# 返回失败状态
[ $failed -gt 0 ] && exit 1
exit 0
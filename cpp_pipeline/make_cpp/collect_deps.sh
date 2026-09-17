#!/bin/bash
# collect_deps.sh — 递归收集可执行文件的全部非系统依赖库, 生成自包含部署包
#
# 用法:
#   ./collect_deps.sh <可执行文件> [输出目录] [引擎文件]
#   ./collect_deps.sh infer_e2e_b8_cpp ./deploy /path/to/xxx.engine
#
# 产出:
#   <输出目录>/infer_e2e_b8_cpp        可执行文件
#   <输出目录>/lib/*.so                全部非系统依赖库
#   <输出目录>/engine/                 (可选) 引擎文件
#   <输出目录>/run.sh                  运行入口 (自动设置 LD_LIBRARY_PATH)
#
# 原理: 从可执行文件出发 BFS 遍历 ldd 依赖树, 跳过 glibc 核心(不可携带, 必须用
#       目标机自带)与 libcuda(必须用目标机驱动自带), 其余全部拷贝打包。

set -e
BIN=$(readlink -f "$1")
OUT=${2:-./deploy}
ENGINE=$3

# glibc 核心库: 跟随目标机 libc, 严禁携带 (破坏 NSS/动态加载器一致性)
# libcuda: 由目标机 NVIDIA 驱动提供, 必须用目标机版本
EXCLUDE='(linux-vdso|ld-linux-x86-64|libc\.so\.6$|libm\.so\.6$|libpthread|libdl\.so|librt\.so\.1$|libresolv|libutil\.so|libanl\.so|libnss_compat|libnss_files|libnss_dns|libcuda\.so)'

mkdir -p "$OUT/lib"

# BFS 遍历 ldd 依赖树 (单进程数组队列, 避免递归子壳)
declare -A VISITED
count=0
queue=("$BIN")
while [ ${#queue[@]} -gt 0 ]; do
    obj="${queue[0]}"
    queue=("${queue[@]:1}")
    libs=$(ldd "$obj" 2>/dev/null | awk '/=> \// {print $3} /^ *\/[^ ]*$/ {print $1}')
    for lib in $libs; do
        [ -f "$lib" ] || continue
        base=$(basename "$lib")
        if echo "$base" | grep -qE "$EXCLUDE"; then continue; fi
        [ -n "${VISITED[$base]:-}" ] && continue
        VISITED[$base]=1
        if [ ! -e "$OUT/lib/$base" ]; then
            cp -L "$lib" "$OUT/lib/"
            count=$((count+1))
            echo "  收集 $base"
        fi
        queue+=("$lib")
    done
done
echo "  共收集 $count 个库"

cp "$BIN" "$OUT/"
chmod +x "$OUT/$(basename "$BIN")"

if [ -n "$ENGINE" ] && [ -f "$ENGINE" ]; then
    mkdir -p "$OUT/engine"
    cp "$ENGINE" "$OUT/engine/"
    echo "  引擎已放入 $OUT/engine/"
fi

BINNAME=$(basename "$BIN")
cat > "$OUT/run.sh" <<EOF
#!/bin/bash
# 自包含部署包运行入口
DIR=\$(cd "\$(dirname "\$0")" && pwd)
export LD_LIBRARY_PATH="\$DIR/lib:\$LD_LIBRARY_PATH"
exec "\$DIR/$BINNAME" "\$@"
EOF
chmod +x "$OUT/run.sh"

echo
echo "✅ 部署包已生成: $OUT ($(du -sh "$OUT" | awk '{print $1}'))"
echo "   迁移: 整个目录拷到目标机任意位置 → ./run.sh --source ... 直接用"

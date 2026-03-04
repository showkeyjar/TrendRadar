#!/bin/bash
set -e

# 检查配置文件
if [ ! -f "/app/config/config.yaml" ] || [ ! -f "/app/config/frequency_words.txt" ]; then
    echo "❌ 配置文件缺失"
    exit 1
fi

# 保存环境变量
env >> /etc/environment

WEBAPP_PID=""
MCP_PID=""

start_optional_services() {
    if [ "${RUN_WEBAPP:-false}" = "true" ]; then
        local webapp_port="${WEBAPP_PORT:-8899}"
        echo "🌐 启动 Web 控制台: 0.0.0.0:${webapp_port}"
        /usr/local/bin/python -m trendradar.webapp --host 0.0.0.0 --port "${webapp_port}" &
        WEBAPP_PID=$!
    fi

    if [ "${RUN_MCP:-false}" = "true" ]; then
        local mcp_port="${MCP_PORT:-3333}"
        echo "🧠 启动 MCP 服务: 0.0.0.0:${mcp_port}"
        /usr/local/bin/python -m mcp_server.server --transport http --host 0.0.0.0 --port "${mcp_port}" &
        MCP_PID=$!
    fi
}

case "${RUN_MODE:-cron}" in
"once")
    echo "🔄 单次执行"
    start_optional_services
    exec /usr/local/bin/python -m trendradar
    ;;
"cron")
    # 生成 crontab（默认：抓取后自动执行情报分析）
    CRON_CMD="cd /app && /usr/local/bin/python -m trendradar"
    if [ "${RUN_INTEL_AFTER_CRAWL:-true}" = "true" ]; then
        CRON_CMD="${CRON_CMD} && /usr/local/bin/python -m trendradar.intel run --top ${INTEL_TOP_K:-30} --output output/intel/latest_digest.md --json-output output/intel/latest_digest.json"
    fi
    echo "${CRON_SCHEDULE:-*/30 * * * *} ${CRON_CMD}" > /tmp/crontab
    
    echo "📅 生成的crontab内容:"
    cat /tmp/crontab

    if ! /usr/local/bin/supercronic -test /tmp/crontab; then
        echo "❌ crontab格式验证失败"
        exit 1
    fi

    # 立即执行一次（如果配置了）
    if [ "${IMMEDIATE_RUN:-false}" = "true" ]; then
        echo "▶️ 立即执行一次"
        /usr/local/bin/python -m trendradar
    fi

    start_optional_services

    # 启动 Web 服务器（如果配置了）
    if [ "${ENABLE_WEBSERVER:-false}" = "true" ]; then
        echo "🌐 启动 Web 服务器..."
        /usr/local/bin/python manage.py start_webserver
    fi

    echo "⏰ 启动supercronic: ${CRON_SCHEDULE:-*/30 * * * *}"
    echo "🎯 supercronic 将作为 PID 1 运行"

    exec /usr/local/bin/supercronic -passthrough-logs /tmp/crontab
    ;;
*)
    exec "$@"
    ;;
esac

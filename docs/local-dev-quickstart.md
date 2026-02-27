# 本地开发快速启动（Windows / macOS / Linux）

适用目标：在本地快速跑起 TrendRadar，进入可调试状态。

## 1. 环境要求
- Python `>=3.10`
- Git

## 2. 初始化项目
在项目根目录执行：

```powershell
# Windows PowerShell
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

```bash
# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 3. 最小启动（主程序）
```bash
python -m trendradar --show-schedule
python -m trendradar
```

说明：
- 首次运行会在 `output/` 下生成本地数据文件。
- 默认未配置通知渠道时，只做本地处理，不会推送。

## 4. 行业情报模块启动（你当前新增功能）
```bash
python -m trendradar.intel run --top 30 --output output/intel/latest_digest.md --json-output output/intel/latest_digest.json
```

反馈学习测试：
```bash
python -m trendradar.intel feedback --rating 5 --keywords "碳监测,生态数据平台" --sources "生态环境部" --add-keywords "雷击火预警" --note "local dev"
```

## 4.1 一体化本地 Web 控制台（推荐）
```bash
python -m trendradar.webapp --host 127.0.0.1 --port 8899
```

打开浏览器访问：`http://127.0.0.1:8899`

多页面支持：
- `/dashboard`：任务执行与状态
- `/intel`：情报结果列表
- `/feedback`：反馈学习（强相关/一般/噪声 + 加词减词）
- `/keywords`：关键词分组管理（按业务流程分类）
- `/llm`：大模型配置管理（模型、API、分析开关）
- `/sources`：数据源管理（平台/RSS 手工维护 + 自动发现）
- `/notification`：通知渠道配置（企业微信/飞书/钉钉/Telegram/邮件/Slack/Webhook）
- `/schedule`：定时调度配置（开关 + preset）

说明：
- Web 控制台会把通知配置写入 `output/webapp/settings.json`。
- 每次点击“执行一次抓取”时，会生成 `output/webapp/runtime_config.yaml` 并注入运行，不改动原始 `config/config.yaml`。

## 4.2 Docker 打包与运行
本项目已支持 Docker 本地打包（主程序 + MCP + Web 控制台）。

本地构建镜像：
```bash
# macOS / Linux
bash docker/build-local.sh

# Windows
docker\\build-local.bat
```

使用 Compose 启动（包含 `trendradar`、`trendradar-mcp`、`trendradar-webapp`）：
```bash
cd docker
# 建议先在 docker/.env 设置外部持久化目录：
# HOST_CONFIG_DIR=/data/trendradar/config
# HOST_OUTPUT_DIR=/data/trendradar/output
docker compose -f docker-compose-build.yml up -d --build
```

访问地址：
- Web 控制台：`http://127.0.0.1:8899`
- MCP 服务：`http://127.0.0.1:3333`

## 5. VS Code 调试（推荐）
在 `.vscode/launch.json` 使用以下配置：

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "TrendRadar Main",
      "type": "python",
      "request": "launch",
      "module": "trendradar",
      "cwd": "${workspaceFolder}",
      "console": "integratedTerminal",
      "justMyCode": false
    },
    {
      "name": "TrendRadar Main (show-schedule)",
      "type": "python",
      "request": "launch",
      "module": "trendradar",
      "args": ["--show-schedule"],
      "cwd": "${workspaceFolder}",
      "console": "integratedTerminal",
      "justMyCode": false
    },
    {
      "name": "TrendRadar Intel Run",
      "type": "python",
      "request": "launch",
      "module": "trendradar.intel",
      "args": [
        "run",
        "--top",
        "30",
        "--output",
        "output/intel/latest_digest.md",
        "--json-output",
        "output/intel/latest_digest.json"
      ],
      "cwd": "${workspaceFolder}",
      "console": "integratedTerminal",
      "justMyCode": false
    },
    {
      "name": "TrendRadar Web Console",
      "type": "python",
      "request": "launch",
      "module": "trendradar.webapp",
      "args": ["--host", "127.0.0.1", "--port", "8899"],
      "cwd": "${workspaceFolder}",
      "console": "integratedTerminal",
      "justMyCode": false
    }
  ]
}
```

## 6. 常见问题
- 报错 `ModuleNotFoundError: fastmcp`：未安装依赖，重新执行 `pip install -r requirements.txt`。
- 命令 `python -m trendradar.intel` 无法找到：确认已在项目根目录 `d:\codes\TrendRadar` 执行。
- 中文输出乱码（Windows）：优先用 PowerShell 7；必要时执行 `chcp 65001`。

## 7. 推荐本地开发流程
1. 改配置：`config/config.yaml`、`config/frequency_words.txt`、`config/intelligence.yaml`
2. 跑主流程：`python -m trendradar`
3. 跑情报：`python -m trendradar.intel run ...`
4. 做反馈：`python -m trendradar.intel feedback ...`
5. 查看输出：`output/html/`、`output/news/`、`output/intel/`

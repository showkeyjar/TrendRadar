# TrendRadar 行业情报系统（MVP）

## 目标
- 在现有热点聚合基础上，新增行业情报层：政策节奏、投融资信号、窗口期信号。
- 支持反馈学习：评分、加词、减词后，自动调整后续推送优先级。

## 新增能力
- 新配置文件：`config/intelligence.yaml`
- 新模块：`trendradar/intel/`
- 新 CLI：
  - `python -m trendradar.intel run`
  - `python -m trendradar.intel feedback`

## 运行
```bash
python -m trendradar.intel run --top 30 --output output/intel/latest_digest.md --json-output output/intel/latest_digest.json
```

## 反馈学习
```bash
python -m trendradar.intel feedback --rating 5 --keywords "碳监测,生态数据平台" --sources "生态环境部" --add-keywords "雷击火预警"
python -m trendradar.intel feedback --rating 2 --keywords "生态旅游" --block-keywords "营销活动"
```

反馈状态写入 `output/intel/feedback_state.json`，内容包括：
- `keyword_adjustments`: 关键词权重系数
- `source_adjustments`: 来源权重系数
- `added_keywords`: 用户新增词
- `blocked_keywords`: 用户屏蔽词

## 推荐接入主流程
- 先独立跑 `intel run`，确认标签和评分符合预期。
- 然后把 `output/intel/latest_digest.md` 作为附加区域并入现有通知渠道（飞书/企微/Telegram）。
- 最后在前端或机器人交互里接 `feedback` 命令，实现闭环学习。


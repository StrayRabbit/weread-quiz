# weread-quiz

![Python](https://img.shields.io/badge/python-3.10%2B-blue) ![Platform](https://img.shields.io/badge/platform-Windows%2010%2F11-lightgrey) ![License](https://img.shields.io/badge/license-MIT-green) ![Model](https://img.shields.io/badge/model-OpenAI%20%E5%85%BC%E5%AE%B9-informational) ![Release](https://img.shields.io/github/v/release/StrayRabbit/weread-quiz)

微信读书「每日一答」/「问答 PK」自动答题工具（Windows）。

抓小程序界面截图 → 本地 OCR 读题 → 模型判答案 → 定位选项按钮 → 点击前复核 → 逐题答完整局。
只认明确证据：认不出页面、读不出题、模型给不出合法下标时，程序会停下并留证，而不是猜着点。

**模型只判哪个选项对，其余全在本地跑** —— 一局实测约 3.6K token、12 次请求（10 题）。

> **仅供学习研究使用，如有侵权，请联系删除！**

<img src="assets/demo-pk.gif" width="480" alt="问答 PK 自动答题过程录屏（节选）">
<p align="center"><sub>问答 PK 自动答题过程录屏（节选）</sub></p>

## 快速开始

### 方式一：下载免安装包（不用装 Python）

从 [Releases](https://github.com/StrayRabbit/weread-quiz/releases) 下 `weread-quiz-v*-win64.zip`：

1. 解压
2. 双击 **`首次配置.bat`** —— 它把 `.env.example` 复制成 `.env` 并用记事本打开，填 `MODEL_BASE_URL` / `MODEL_API_KEY` / `MODEL_NAME` 三行
3. 在微信里打开微信读书小程序（游戏中心入口页）
4. 双击 **`启动PK.bat`**

包里自带便携 Python，**不用装 Python，也不用装 uv**。

### 方式二：源码运行（需要 Python 3.10+）

```bash
uv sync
cp .env.example .env      # 编辑 .env，填 MODEL_BASE_URL / MODEL_API_KEY / MODEL_NAME
```

然后在微信里打开微信读书小程序（游戏中心入口页），双击项目根目录的 **`启动PK.bat`**。

### 小程序直达链接

复制到微信「文件传输助手」再点开（`#小程序://` 协议在浏览器里点不开）：

| 入口 | 链接 |
|------|------|
| 问答 PK | `#小程序://微信读书/bJK5TaSZ3rUpu1D` |
| 每日一答 | `#小程序://微信读书/lI9387D9hnbmJnk` |

## 启动器

双击即可：

| 启动器 | 作用 |
|--------|------|
| `启动PK.bat` | 问答 PK：自动匹配 → 答完整局（常规 10 题，平局加时赛会接着答） |
| `启动每日一答-瓜分天数.bat` | 每日一答：瓜分天数路线 → 答完 12 题 |
| `启动每日一答-瓜分书币.bat` | 每日一答：瓜分书币路线 → 答完 12 题 |
| `首次配置.bat` | 生成并打开 `.env`（免安装包用它做第一步） |

免安装包用随包的 `python/`，源码运行用项目的 `.venv/`，命令行形式等价：

```bash
weread-quiz pk --dry-run    # 只识别不点击
weread-quiz pk --trace      # 详细日志打到终端
weread-quiz daily           # 每日一答（--route daily-days / daily-coins）
```

## 工作流程

```mermaid
flowchart TD
    A["小程序窗口（前台可见）"] --> B["本地：白名单文字推进<br/>「问答PK → 自动匹配」<br/>「开始游戏」"]
    B --> C["本地：截图 + RapidOCR 读题<br/>题干 / 选项 / 置信度门控"]
    C -->|"门控通过"| D["模型 ① 纯文本判下标<br/>只发题干和编号选项，收一个下标"]
    C -->|"门控不通过"| E["模型 ② 整图兜底<br/>读题 + 判答案一次做完"]
    D --> F["本地：几何定位选项按钮 + 点击前复核"]
    E --> F
    F --> G["本地：点击"]
    G --> H{"还有下一题？"}
    H -->|"是"| C
    H -->|"否"| I["本地：结算页判读<br/>模型 ③ 兜底（通常用不到）"]

    classDef local fill:#f4f7f4,stroke:#8aa88a,color:#203020
    classDef model fill:#fff3e0,stroke:#d07f00,color:#402a00,stroke-width:2px
    class A,B,C,F,G,H,I local
    class D,E model
```

**模型只干一件事：判哪个选项对。** 找窗口、抓屏、读题、定位按钮、点击前复核、点击 —— 全在本地跑完，不消耗 token。

| 调用 | 什么时候 | 发给模型的东西 | 频率 |
|------|---------|---------------|------|
| ① 判下标 | 每道题的主路径 | 题干 + 编号选项，纯文本 | 每题 1 次 |
| ② 整图兜底 | 本地 OCR 门控不通过（置信度不足、选项数与按钮数对不上、没读出） | 整张截图 | 偶发，10~12 题里通常 0~1 次 |
| ③ 结算页判读 | 本地 OCR 没读到明确的胜负或积分 | 结算截图 | 每局最多 1 次 |

一局实测（`data/runs/*/run.log` 里的真实记录）：PK 10 题 12 次请求 / 3.68K token；每日一答 12 题 13 次 / 3.57K。重试只会把同一道题的请求数翻一倍（`VISION_RETRIES`，默认 1），不会因为复核、点击重来而放大。

## 配置

全部通过 `.env`，下面是全部可调项。**主模型必填，视觉模型可选**：

| 变量 | 默认 | 说明 |
|------|------|------|
| `MODEL_BASE_URL` / `MODEL_API_KEY` / `MODEL_NAME` | 必填 | 主模型端点。**换供应商只改这三行**，代码不用动 |
| `VISION_BASE_URL` / `VISION_API_KEY` / `VISION_NAME` | 可选 | 视觉模型（能读图的那个）。主模型是决策模型时建议配上 —— 见下面「选哪个模型」 |
| `MODEL_THINKING` | `1` | 是否让视觉模型输出思考块。**PK 不要关**：实测快 312ms/题，但 10 道真实题里 2 道的答案会变。**不影响决策模型**（它没有 thinking 概念）。MiniMax-M3.1-Flash 强制要思考，关了会被服务端拒（400） |
| `MODEL_MIN_CONFIDENCE` | 留空 | 复核门槛：决策模型置信度低于它就转给视觉模型复核。**留空 = 自动**（配了视觉模型用 0.8，没配就不复核）；`0` = 不复核 |
| `VISION_RETRIES` | `1` | 视觉模型失败后的重试次数。PK 时间预算紧，重试 2 次必然超时 |
| `GUESS_ON_FAILURE` | `1` | 模型失败时是否点第一个按钮（超时惩罚与答错相同，蒙有 25~33% 机会） |
| `OCR_MIN_CONFIDENCE` | `0.7` | PK 本地 OCR 最低置信度，低于阈值转图片路径 |
| `DAILY_OCR_MIN_CONFIDENCE` | `0.85` | daily 的阈值，独立于 PK |
| `PK_NEXT_QUESTION_DELAY` | `1.0` | PK 点击后开始检查下一题的固定延迟（秒） |
| `PK_OCR_FALLBACK_SECONDS` | `3.0` | PK 本地 OCR 连续判否后，允许一次视觉兜底的等待时间（秒） |
| `PK_TEXT_TIMEOUT` | `5.0` | PK 文本阶段（判下标，含复核）的最大用时（秒）。有图片能力时还会动态缩短，为图片回退留出预算 |

### 模型慢怎么办

PK 每题 12 秒，识别和点击共用这份预算。程序会扣掉本地 OCR 耗时和点击预留；有图片能力时，文本阶段还会为图片回退预留 3.5 秒及 0.3 秒超时退出余量，避免文本请求一超时就只能蒙答。

- **调高文本上限**：例如设置 `PK_TEXT_TIMEOUT=6.5`。这只是最大值，实际用时仍受当题剩余预算限制；没有图片能力时，不预留图片预算。
- **换快模型**：慢模型即使拿到回退机会，也不保证来得及返回；优先选择响应稳定、速度合适的模型。

### 选哪个模型

配置里只有一组模型变量 —— **协议是实现细节，不该让你填**。程序从地址和模型名判断：

| 判据 | 协议 | 能力 |
|------|------|------|
| 地址以 `/systemone` 结尾，或模型名像 `decision-model-*` / `jev-*` | System One（决策模型） | 判下标**极快**（百毫秒级），但**读不了图** |
| 其余 | OpenAI 兼容 `chat/completions` | 判下标 + 读图 |

填一个普通多模态模型 → 全部走它；填一个决策模型 → 判下标走它，但整图兜底不可用。

配一个模型也行，配两个更好：

- **决策模型**：判下标极快（实测 78ms）但会答错（实测 85.5%），且读不了图
- **视觉模型**：慢一点但更准（实测 100%）
- **两个都配**：拿不准的题转视觉模型复核 → 答对率 98.6%

**推荐**：主模型填阿里百炼的 `decision-model-preview`（实测 78ms、目前限免），再配一个 `VISION_*` 视觉模型兜底。分工细节、实测数据、复核门槛依据见 [docs/design.md](docs/design.md)。

## 使用与注意

前提：微信里已打开小程序并停在游戏中心入口页 → 双击对应启动器，程序自己推进到答题页并开始作答。**Esc 键紧急中止**。

- **小程序窗口不能被遮挡，要留在最外层。** 程序启动后会把它抬到前面并置顶（跑完还原）；小程序失去焦点会停止渲染，程序会重新激活（白屏自愈）。
- **Windows 显示缩放不用调。** 100%/125%/150%/200% 都支持。
- **不要同时跑两个启动器。** 没有跨进程互斥，两个进程抢桌面会互相干扰。
- **正式每日一答答错即当日机会用完**，不可恢复。
- **不会使用任何道具**（跳题卡 / 换题卡 / 续命卡一律不碰），也不会自动点「取消匹配」或「邀请好友」。
- 每日一答活动页上有两张卡片，两个启动器各走一条路线，**互不干扰**。
- PK 的平局加时赛会继续出题，程序接着答到结算页出现为止。
- 仅支持 Windows，依赖 mss 抓屏 / RapidOCR 本地识别 / pyautogui 点击。

## 自己构建免安装包

```bash
uv run python tools/build_release.py
```

产物在 `dist/weread-quiz-v*-win64.zip`。打 `v*` tag 推上去，GitHub Actions 会自动构建并发布。

离线回归测试（不需要真微信和网络）：`uv sync --extra dev && pytest`

## 支持与反馈

如果它对你有用，欢迎点个 ⭐（Star）；也欢迎提 Issue —— 界面判据、预算策略、模型调用边界这几块都还有优化空间。

## License

MIT，见 [LICENSE](LICENSE)。

本项目仅供学习研究使用，如有侵权，请联系删除！第三方模型和微信相关素材的权利不因本项目 LICENSE 而改变。

## 社区

本项目分享于 [LINUX DO 社区](https://linux.do/)，欢迎社区佬友交流、反馈和参与贡献。

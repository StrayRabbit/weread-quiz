# 调试与排查

面向改代码、调阈值的人。日常使用不需要看这一篇。

## 看时间花在哪（`--trace`）

平时那些分段耗时、请求时延**只落进 `run.log`**，终端上看不见。加 `--trace`
之后同样的行会**同步打到终端**（内容与 `run.log` 逐字相同），另外还多一行
等待阶段的耗时归因：

```
[09:56:20] [pk] 请求 #1 时延（完成）：...HTTP总计 1112ms...
[09:56:20] [pk] 第 2 题（含等待下一题） 耗时 4913ms：模型识别 1792ms 定位按钮 3ms...
[09:56:20] [pk] 等待阶段明细：抓屏+门控 6轮 28ms｜题号行OCR 5次 4003ms｜本地OCR 1次 630ms｜
                    模型请求 1次 1161ms｜轮询等待 5次 505ms（已归因 6326ms/共 6326ms）
```

最后那行把等待阶段拆成五类，耗时大头一眼可见：

| 类别 | 是什么 |
|------|--------|
| `抓屏+门控` | 每轮的抓屏 + 模板预筛 + 像素门控 + 题干渲染判据 |
| `题号行OCR` | 模板判否时读题号行的那次 OCR（匹配页上会反复发生） |
| `本地OCR` | 整图路 + 拼接路 + 合并带（本地读题干与选项） |
| `模型请求` | 发给模型的网络往返（文本路径与图片路径都算） |
| `轮询等待` | 每轮之间的 `sleep(_POLL_INTERVAL)` —— 纯等待，不产生进展 |

## 逐层看读题链路（`tools/debug_read.py`）

调阈值（`OCR_MIN_CONFIDENCE`）、改几何、改 prompt 时，需要知道**是哪一层、
因为什么原因**变了。`--trace` 只给耗时归因，不给读数；这个脚本给读数：

```bash
# 离线：像素层 + OCR 两路 + 门控，不联网不花钱
python tools/debug_read.py --image 答题页.png --mode pk
python tools/debug_read.py --dir 每日一答截图目录 --mode daily --out out/daily.json

# 带模型（花钱、慢）
python tools/debug_read.py --image 答题页.png --mode pk --model --model-path image
```

逐层输出：像素判定 → 按钮定位 → OCR 整图路 → OCR 拼接路 → 门控裁决 →
模型。任一层失败都如实打印原因。

**`--out` 的 JSON 是给 diff 用的** —— 改代码前后跑同一批截图，两份 JSON
直接 diff 就能定位变化（耗时字段除外；同一批图的读数是确定的）。

`--mode auto`（默认）按按钮检测器自动判别链路；**判不出来就跳过该图**
（如结算页），不瞎猜。PK 与每日一答的按钮检测器互斥 —— 用错一张都找不到。

注意：**中文输出需要 `PYTHONIOENCODING=utf-8`**（Windows 默认 stdout 是
gbk），脚本会检测并提示。

## 阈值校准

`OCR_MIN_CONFIDENCE`（PK，默认 0.7）与 `DAILY_OCR_MIN_CONFIDENCE`
（daily，默认 0.85）的依据、实测数据和调法见
[design.md 的「阈值怎么调」](design.md#设计要点)。日常只看每局结束终端打的
一行汇总：

```
OCR 快路径：命中 9/10，回退图片 1 次（原因：本地 OCR 置信度不足）
```

完整读数（每个选项的置信度）在 `data/runs/<子命令>/<时间戳>/run.log` 的
「图片路径 回退原因」行里。

## tools/ 脚本

| 脚本 | 用途 |
|------|------|
| `debug_read.py` | 逐层看读题链路：像素判定 → 按钮定位 → OCR 两路 → 门控 → 模型。调阈值、改几何、改 prompt 时用它 |
| `capture_watch.py` | 监视微信小程序窗口，检测到答题界面就自动存图 |
| `segment_questions.py` | 按帧差把连续截图切成「每题一段」 |
| `verify_vision.py` | 拿真实截图验证视觉识别 + 按钮定位 |
| `write_launchers.py` | 生成三个 `.bat` 启动器（纯 ASCII + CRLF，改启动器时用生成器而不是手改） |
| `bench_thinking_ab.py` | `MODEL_THINKING` 开/关的 A/B 实测，需要自备 PK 截图数据集 |

仓库不含真实截图样本。上述脚本可直接对任意截图或实时窗口使用，
`bench_thinking_ab.py` 需要自备 PK 截图数据集。

## 测试

```bash
uv sync --extra dev  # 仅开发 / 测试时安装 pytest
uv run --extra dev pytest  # 离线回归测试，不需要微信、不需要网络
```

测试全部离线：注入假时钟、模型、截图与点击器验证流程，默认禁止外网
请求（`tests/conftest.py` 直接打断 socket）。
`tests/test_daily_pipeline.py`、`tests/test_daily_safety.py` 覆盖 daily 的本地
确认、预读复用、预算、身份校验、终态、dry-run 和资源关闭；其余用例覆盖 PK
预算、题号缓存、答后取色、选项下标映射、模型预热及 RapidOCR 接口。
测试不会消耗真实答题机会。

依赖升级的验证记录见 `2026-11-01-rapidocr-3.9-validation.md`。

"""决策模型 vs 现有模型：拿真实历史题的标准答案逐题对照。

**标准答案是白送的** —— run.log 里的「答后反馈：答对/答错」是程序读游戏
画面颜色判出来的，不是我们自己标的。所以这份语料可信。

题面和选项用本地 OCR 从截图里再读一遍（和当年那一局用的是同一套代码、
同一张图，RapidOCR 是确定性的，所以读出来的是同一批选项）。

评估口径是**答案对不对**，不是解析成功率 —— 关掉 thinking 时解析成功率
反而更好，但答案是错的。判错的下标要和标准答案的下标比，不是比字符串。

用法::

    uv run python tools/bench_decision.py                 # 只跑决策模型
    uv run python tools/bench_decision.py --mode both     # 两个模型对照
    uv run python tools/bench_decision.py --limit 20      # 只跑前 20 题

多个决策模型供应商用 ``--providers`` 指向一个 JSON 文件（默认
``data/decision-providers.json``，该目录被 gitignore，key 不会进仓库）::

    [
      {"name": "阿里 decision-model-preview",
       "base_url": "https://<workspace>.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
       "api_key": "sk-...", "model": "decision-model-preview"},
      {"name": "Jev",
       "base_url": "https://<host>/v1", "api_key": "...", "model": "jev-latest"}
    ]
"""
from __future__ import annotations

import argparse
import re
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from weread_quiz.config import AppConfig  # noqa: E402
from weread_quiz.decision import DecisionService  # noqa: E402
from weread_quiz.model_service import (  # noqa: E402
    PROTOCOL_CHAT,
    PROTOCOL_SYSTEMONE,
    detect_protocol,
)
from weread_quiz.pk.ocr import RapidOcrReader  # noqa: E402
from weread_quiz.vision import VisionService  # noqa: E402

RUNS = ROOT / "data" / "runs" / "pk"

# 「第 N 题 题干 → 答案，已点击」
_ANSWERED = re.compile(r"第 (\d+) 题 (.+?) → (.+?)，已点击")
# 「第 N 题 答后反馈：答对」
_VERDICT = re.compile(r"第 (\d+) 题 答后反馈：(答对|答错|未确认)")


class Item:
    """一道有标准答案的题。"""

    def __init__(self, run_dir: Path, number: int, stem: str, answer: str,
                 verdict: str):
        self.run_dir = run_dir
        self.number = number
        self.stem = stem
        self.answer = answer
        self.verdict = verdict
        self.options: list[str] = []
        self.truth_index: int | None = None

    @property
    def screenshot(self) -> Path | None:
        """该题点击前的截图。"""
        pattern = f"*_q{self.number:02d}_before.png"
        found = sorted(self.run_dir.glob(pattern))
        return found[0] if found else None


def collect() -> list[Item]:
    """从 run.log 里收集有对错判定的题。"""
    items: list[Item] = []
    for log in sorted(RUNS.glob("*/run.log")):
        run_dir = log.parent
        answered: dict[int, tuple[str, str]] = {}
        verdicts: dict[int, str] = {}
        for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
            match = _ANSWERED.search(line)
            if match:
                answered[int(match.group(1))] = (match.group(2), match.group(3))
            match = _VERDICT.search(line)
            if match:
                verdicts[int(match.group(1))] = match.group(2)
        for number, (stem, answer) in answered.items():
            verdict = verdicts.get(number)
            if verdict is None or verdict == "未确认":
                continue
            items.append(Item(run_dir, number, stem, answer, verdict))
    return items


def read_options(ocr: RapidOcrReader, item: Item) -> bool:
    """用本地 OCR 重读截图，拿到选项并定位标准答案的下标。"""
    path = item.screenshot
    if path is None:
        return False
    from PIL import Image

    read = ocr.read(Image.open(path).convert("RGB"))
    if read is None or not read.options:
        return False
    item.options = list(read.options)
    # 标准答案就是当年模型选中的那个选项文本 —— 在同一批 OCR 结果里找它。
    for index, text in enumerate(item.options):
        if text.strip() == item.answer.strip():
            item.truth_index = index
            return True
    # OCR 结果可能和当年那次有细微差别，退一步做包含匹配。
    for index, text in enumerate(item.options):
        if item.answer.strip() in text.strip() or text.strip() in item.answer.strip():
            item.truth_index = index
            return True
    return False


def run_decision(service: DecisionService, item: Item) -> tuple[int | None, float]:
    started = time.monotonic()
    index = service.answer_options(item.stem, item.options)
    return index, (time.monotonic() - started) * 1000


def run_vision(service: VisionService, item: Item) -> tuple[int | None, float]:
    from weread_quiz.pk.ocr import TEXT_PROMPT

    options = "\n".join(f"{i}. {t}" for i, t in enumerate(item.options))
    started = time.monotonic()
    index = service.answer_from_text(TEXT_PROMPT % (item.stem, options),
                                     len(item.options))
    return index, (time.monotonic() - started) * 1000


def report(name: str, results: list[tuple[Item, int | None, float]]) -> None:
    answered = [(item, index, ms) for item, index, ms in results if index is not None]
    correct = [(item, index, ms) for item, index, ms in answered
               if index == item.truth_index]
    times = [ms for _, _, ms in answered]
    print()
    print(f"== {name} ==")
    print(f"  有答案 {len(answered)}/{len(results)}"
          f"（其余是没给出合法下标）")
    if answered:
        accuracy = len(correct) / len(answered)
        print(f"  答对 {len(correct)}/{len(answered)} = {accuracy:.1%}")
        times_sorted = sorted(times)
        print(f"  时延 中位 {statistics.median(times):.0f}ms"
              f"，p90 {times_sorted[int(len(times_sorted) * 0.9) - 1]:.0f}ms"
              f"，最大 {max(times):.0f}ms")
    wrong = [(item, index) for item, index, _ in answered if index != item.truth_index]
    if wrong:
        print(f"  答错的题：")
        for item, index in wrong:
            print(f"    第{item.number}题 {item.stem[:28]}"
                  f" → 选了「{item.options[index]}」，正确「{item.options[item.truth_index]}」")


def load_providers(path: Path | None) -> list[dict]:
    """读决策模型供应商列表。

    有 providers 文件就用它（可以一次测多个）；没有就退回 .env 里的模型 ——
    前提是它走的是 System One 协议（决策模型）。
    """
    if path is not None and path.is_file():
        import json

        providers = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(providers, list) or not providers:
            raise SystemExit(f"{path} 里没有供应商配置")
        return providers
    cfg = AppConfig.from_env()
    if not cfg.model_configured:
        raise SystemExit("没配模型（MODEL_BASE_URL / MODEL_API_KEY / MODEL_NAME）")
    if detect_protocol(cfg.model_base_url, cfg.model_name) != PROTOCOL_SYSTEMONE:
        raise SystemExit(
            f"当前模型（{cfg.model_name}）不是决策模型 —— "
            f"要么换一个 System One 协议的模型，要么用 --providers 指定"
        )
    return [{
        "name": cfg.model_name,
        "base_url": cfg.model_base_url,
        "api_key": cfg.model_api_key,
        "model": cfg.model_name,
    }]


def main() -> int:
    parser = argparse.ArgumentParser(description="决策模型与现有模型逐题对照")
    parser.add_argument("--mode", choices=["decision", "vision", "both"],
                        default="decision")
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 题")
    parser.add_argument("--min-confidence", type=float, default=0.0,
                        help="决策模型置信度门槛")
    parser.add_argument("--providers", type=Path,
                        default=ROOT / "data" / "decision-providers.json",
                        help="决策模型供应商列表（JSON）")
    args = parser.parse_args()

    items = collect()
    if args.limit:
        items = items[:args.limit]
    print(f"有标准答案的题：{len(items)} 道", flush=True)

    ocr = RapidOcrReader(min_confidence=0.0)
    usable: list[Item] = []
    for item in items:
        if read_options(ocr, item):
            usable.append(item)
    print(f"本地 OCR 能读出选项的：{len(usable)} 道", flush=True)
    if not usable:
        print("没有可用的题 —— 先跑几局真实对局攒语料")
        return 1

    if args.mode in ("decision", "both"):
        for provider in load_providers(args.providers):
            print(f"\n—— 决策模型：{provider.get('name', provider['model'])} ——",
                  flush=True)
            with DecisionService(
                base_url=provider["base_url"],
                api_key=provider["api_key"],
                model=provider["model"],
                min_confidence=args.min_confidence,
            ) as service:
                results = []
                for item in usable:
                    index, ms = run_decision(service, item)
                    results.append((item, index, ms))
                    print(f"  [{len(results)}/{len(usable)}] "
                          f"置信度 {service.last_confidence} {ms:.0f}ms "
                          f"{item.stem[:20]}", flush=True)
            report(provider.get("name", provider["model"]), results)

    if args.mode in ("vision", "both"):
        cfg = AppConfig.from_env()
        if detect_protocol(cfg.model_base_url, cfg.model_name) != PROTOCOL_CHAT:
            print("当前模型不是 chat 协议（读不了图）—— 跳过对照")
            return 0
        with VisionService(
            base_url=cfg.model_base_url,
            api_key=cfg.model_api_key,
            model=cfg.model_name,
            retries=cfg.vision_retries,
            thinking=cfg.model_thinking,
        ) as service:
            vision_results = []
            for item in usable:
                index, ms = run_vision(service, item)
                vision_results.append((item, index, ms))
                print(f"  [{len(vision_results)}/{len(usable)}] {ms:.0f}ms "
                      f"{item.stem[:20]}", flush=True)
        report("现有模型", vision_results)
    return 0


if __name__ == "__main__":
    sys.exit(main())

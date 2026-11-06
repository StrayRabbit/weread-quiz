"""配置加载。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from .pk_budget import _TEXT_PATH_TIMEOUT

def _project_root() -> Path:
    """定位 .env 和 data/ 所在的目录。

    源码布局下是仓库根目录（src/weread_quiz/config.py 的上两级）。
    免安装版把程序装进 site-packages，那条路径推导会指向 Python 的 Lib 目录，
    所以启动器用 WEREQ_HOME 指向解压后的目录；没设时仍按源码布局推导。
    """
    home = os.environ.get("WEREQ_HOME")
    if home:
        return Path(home)
    return Path(__file__).resolve().parents[2]


PROJECT_ROOT = _project_root()
DATA_DIR = PROJECT_ROOT / "data"
RUNS_DIR = DATA_DIR / "runs"

_TRUTHY = {"1", "true", "yes", "on"}
_FALSY = {"0", "false", "no", "off"}

# .env.example 里的占位符。用户没改就跑了 —— 得说清楚，而不是让每题都蒙。
_PLACEHOLDERS = ("在这里填", "在这里换", "xxxx", "your-key", "your_key", "<你的")


def _is_placeholder(value: str) -> bool:
    """这个值是不是模板里没改过的占位符。

    地址里带尖括号（``<workspace-id>`` 这种）也算 —— 那不是能用的地址。
    """
    text = (value or "").strip().lower()
    if not text:
        return False
    if "<" in text or ">" in text:
        return True
    return any(marker in text for marker in _PLACEHOLDERS)


def _env_flag(name: str, default: bool) -> bool:
    """读布尔型环境变量。未设置返回默认值。"""
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in _TRUTHY:
        return True
    if value in _FALSY:
        return False
    raise ValueError(
        f"{name} 只能是 {sorted(_TRUTHY)} 或 {sorted(_FALSY)} 之一，收到 {raw!r}"
    )


def _load_dotenv(path: Path) -> None:
    """读 .env —— UTF-8 读不了就换 GBK。

    中文 Windows 的记事本默认存 ANSI（GBK）。用 UTF-8 读会抛
    UnicodeDecodeError，用户看到的是一串看不懂的报错。这里换 GBK 再读
    一次（GBK 也能读纯 ASCII，对 UTF-8 文件没有副作用）。
    """
    try:
        load_dotenv(path, override=False)
    except UnicodeDecodeError:
        load_dotenv(path, override=False, encoding="gbk")


@dataclass
class AppConfig:
    """运行时配置，来自 .env / 环境变量。"""

    model_base_url: str
    model_api_key: str
    model_name: str
    # 可选的第二个模型 —— 只负责整图路径（读截图）。
    #
    # 主模型可以是决策模型（快，但读不了图）。配上视觉模型之后：
    # 文本路径先用决策模型（实测 78ms），拿不准就转给视觉模型复核；
    # 整图路径（本地 OCR 没读出题时）由视觉模型接手。
    #
    # 三项要么都填、要么都不填 —— 填一半是配置错误，当场报错。
    vision_base_url: str = ""
    vision_api_key: str = ""
    vision_name: str = ""
    vision_retries: int = 1
    # 重试仍失败时是否蒙一个答案。PK 里超时和答错的惩罚一样（这题不得分），
    # 所以"选第一个"有 25~33% 机会，不得分是 0% —— 开了不亏。
    guess_on_failure: bool = True
    model_thinking: bool = True
    ocr_min_confidence: float = 0.7
    # 每日一答 OCR 快路径的置信度门控阈值。
    #
    # 0.85 落在"正常帧全部放行"和"过渡帧被拦"之间：daily 唯一低于
    # 0.9 的两帧是按钮正在消失的过渡帧，正好该被挡掉。
    daily_ocr_min_confidence: float = 0.85
    pk_next_question_delay: float = 1.0
    pk_ocr_fallback_seconds: float = 3.0
    # PK 文本路径（本地 OCR 读题 + 模型只判下标）的单次上限（秒）。
    #
    # 默认 5 秒是给快模型留的。慢模型要调高 —— 实测 MiniMax-M3.1-Flash
    # 文本路径 3.2~4.6 秒、偶尔超过 5 秒，被切掉就只能蒙答。
    #
    # 调高要一起看预算：倒计时 12 秒减去点击预留是 10.5 秒，文本路径
    # 占 6.5 秒就只剩 4 秒给图片路径（读图要 5.5 秒以上）。
    pk_text_timeout: float = _TEXT_PATH_TIMEOUT

    # 决策模型拿不准时的**复核门槛** —— 置信度低于它就转给视觉模型复核。
    #
    # * None（没配）= 自动：配了视觉模型就用 0.8，没配就不复核。
    # * 0 = 明确不复核，决策模型的答案直接用。
    # * >0 = 用这个值。
    #
    # 实测（69 道带标准答案的真实题）：决策模型答错的 10 道里 9 道置信度
    # ≤ 0.76。门槛 0.8 → 30% 的题转给视觉模型复核，答对率 85.5% → 98.6%。
    model_min_confidence: float | None = None

    @property
    def model_configured(self) -> bool:
        """模型三项配齐了没有。"""
        return bool(self.model_base_url and self.model_api_key and self.model_name)

    @property
    def vision_model_configured(self) -> bool:
        """视觉模型三项配齐了没有。"""
        return bool(self.vision_base_url and self.vision_api_key and self.vision_name)

    @classmethod
    def from_env(cls) -> "AppConfig":
        _load_dotenv(PROJECT_ROOT / ".env")

        # 主模型 —— 协议（OpenAI 兼容 / System One）由代码自己认，
        # 用户不该在配置里看见它。
        model_base_url = os.getenv("MODEL_BASE_URL", "").strip()
        model_api_key = os.getenv("MODEL_API_KEY", "").strip()
        model_name = os.getenv("MODEL_NAME", "").strip()

        # 视觉模型（可选）—— 整图路径 + 决策模型拿不准时的复核。
        # 三项要么都填，要么都不填。
        vision_base_url = os.getenv("VISION_BASE_URL", "").strip()
        vision_api_key = os.getenv("VISION_API_KEY", "").strip()
        vision_name = os.getenv("VISION_NAME", "").strip()
        if any(_is_placeholder(value)
               for value in (vision_base_url, vision_api_key, vision_name)):
            # 模板里给了示例行，用户没改 —— 当作没配，别报错。
            if any((vision_base_url, vision_api_key, vision_name)):
                print("提示：VISION_* 还是模板里的示例 —— 视觉模型按「没配」处理。"
                      "想用就把那三行改掉。")
            vision_base_url = vision_api_key = vision_name = ""
        if any((vision_base_url, vision_api_key, vision_name)):
            missing = [
                name for name, value in (
                    ("VISION_BASE_URL", vision_base_url),
                    ("VISION_API_KEY", vision_api_key),
                    ("VISION_NAME", vision_name),
                )
                if not value
            ]
            if missing:
                raise ValueError(
                    f"视觉模型配置不完整：缺少 {', '.join(missing)}"
                    f"（VISION_BASE_URL / VISION_API_KEY / VISION_NAME "
                    f"三项要么都填，要么都不填）"
                )

        # 旧配置迁移：DECISION_* 是"决策模型"时代的三项，现在并进 MODEL_* ——
        # 决策模型只是另一种协议，不再是单独的一个模型。
        legacy = {
            "base_url": os.getenv("DECISION_BASE_URL", "").strip(),
            "api_key": os.getenv("DECISION_API_KEY", "").strip(),
            "model": os.getenv("DECISION_MODEL", "").strip(),
        }
        if any(legacy.values()):
            if not (model_base_url and model_api_key and model_name):
                # 只配了旧的决策模型 → 它就是那个模型。
                model_base_url = legacy["base_url"] or model_base_url
                model_api_key = legacy["api_key"] or model_api_key
                model_name = legacy["model"] or model_name
                print("提示：检测到旧的 DECISION_* 配置 —— 已并进 MODEL_*，"
                      "以后只配一组就行。")
            else:
                print("提示：DECISION_* 已经并进 MODEL_* —— 现在只配一个模型，"
                      "DECISION_* 不再生效（当前用 MODEL_*）。"
                      "想让决策模型生效，把它那三行挪到 MODEL_* 即可。")

        try:
            vision_retries = int(os.getenv("VISION_RETRIES", "1"))
        except ValueError as exc:
            raise ValueError(
                f"VISION_RETRIES 必须是整数，收到 {os.getenv('VISION_RETRIES')!r}"
            ) from exc
        if vision_retries < 0:
            raise ValueError(f"VISION_RETRIES 不能为负数，收到 {vision_retries}")

        try:
            ocr_min_confidence = float(os.getenv("OCR_MIN_CONFIDENCE", "0.7"))
        except ValueError as exc:
            raise ValueError(
                f"OCR_MIN_CONFIDENCE 必须是 0~1 的小数，收到 {os.getenv('OCR_MIN_CONFIDENCE')!r}"
            ) from exc
        if not (0.0 <= ocr_min_confidence <= 1.0):
            raise ValueError(
                f"OCR_MIN_CONFIDENCE 必须在 0~1 之间，收到 {ocr_min_confidence}"
            )

        try:
            daily_ocr_min_confidence = float(
                os.getenv("DAILY_OCR_MIN_CONFIDENCE", "0.85")
            )
        except ValueError as exc:
            raise ValueError(
                f"DAILY_OCR_MIN_CONFIDENCE 必须是 0~1 的小数，"
                f"收到 {os.getenv('DAILY_OCR_MIN_CONFIDENCE')!r}"
            ) from exc
        if not (0.0 <= daily_ocr_min_confidence <= 1.0):
            raise ValueError(
                f"DAILY_OCR_MIN_CONFIDENCE 必须在 0~1 之间，"
                f"收到 {daily_ocr_min_confidence}"
            )

        try:
            pk_next_question_delay = float(os.getenv("PK_NEXT_QUESTION_DELAY", "1.0"))
        except ValueError as exc:
            raise ValueError(
                f"PK_NEXT_QUESTION_DELAY 必须是数字，收到 {os.getenv('PK_NEXT_QUESTION_DELAY')!r}"
            ) from exc
        if pk_next_question_delay < 0:
            raise ValueError(
                f"PK_NEXT_QUESTION_DELAY 不能为负数，收到 {pk_next_question_delay}"
            )

        try:
            pk_ocr_fallback_seconds = float(
                os.getenv("PK_OCR_FALLBACK_SECONDS", "3.0")
            )
        except ValueError as exc:
            raise ValueError(
                f"PK_OCR_FALLBACK_SECONDS 必须是数字，"
                f"收到 {os.getenv('PK_OCR_FALLBACK_SECONDS')!r}"
            ) from exc
        if pk_ocr_fallback_seconds < 0:
            raise ValueError(
                f"PK_OCR_FALLBACK_SECONDS 不能为负数，收到 {pk_ocr_fallback_seconds}"
            )

        try:
            pk_text_timeout = float(
                os.getenv("PK_TEXT_TIMEOUT", str(_TEXT_PATH_TIMEOUT))
            )
        except ValueError as exc:
            raise ValueError(
                f"PK_TEXT_TIMEOUT 必须是数字，收到 {os.getenv('PK_TEXT_TIMEOUT')!r}"
            ) from exc
        if pk_text_timeout <= 0:
            raise ValueError(
                f"PK_TEXT_TIMEOUT 必须大于 0，收到 {pk_text_timeout}"
            )

        raw_confidence = os.getenv("MODEL_MIN_CONFIDENCE", "").strip()
        if raw_confidence:
            try:
                model_min_confidence = float(raw_confidence)
            except ValueError as exc:
                raise ValueError(
                    f"MODEL_MIN_CONFIDENCE 必须是 0~1 的小数，收到 {raw_confidence!r}"
                ) from exc
            if not (0.0 <= model_min_confidence <= 1.0):
                raise ValueError(
                    f"MODEL_MIN_CONFIDENCE 必须在 0~1 之间，收到 {model_min_confidence}"
                )
        else:
            # 没配 = 自动（配了视觉模型就用默认门槛，没配就不复核）。
            model_min_confidence = None

        return cls(
            model_base_url=model_base_url,
            model_api_key=model_api_key,
            model_name=model_name,
            vision_base_url=vision_base_url,
            vision_api_key=vision_api_key,
            vision_name=vision_name,
            vision_retries=vision_retries,
            guess_on_failure=_env_flag("GUESS_ON_FAILURE", True),
            model_thinking=_env_flag("MODEL_THINKING", True),
            ocr_min_confidence=ocr_min_confidence,
            daily_ocr_min_confidence=daily_ocr_min_confidence,
            pk_next_question_delay=pk_next_question_delay,
            pk_ocr_fallback_seconds=pk_ocr_fallback_seconds,
            pk_text_timeout=pk_text_timeout,
            model_min_confidence=model_min_confidence,
        )

    def validate_model(self) -> None:
        """模型配置是答题必需项，缺失时立即报错而不是等到调用时。

        只配一个模型 —— 协议（OpenAI 兼容 / System One）由代码自己认，
        用户在配置里看不见它。
        """
        missing = [
            name
            for name, value in (
                ("MODEL_BASE_URL", self.model_base_url),
                ("MODEL_API_KEY", self.model_api_key),
                ("MODEL_NAME", self.model_name),
            )
            if not value
        ]
        if missing:
            raise ValueError(
                f"缺少必需的环境变量：{', '.join(missing)}（见 .env.example）"
            )
        if _is_placeholder(self.model_api_key):
            # 密钥还是模板里的占位符 —— 继续跑只会每题都失败然后蒙答案，
            # 看起来像"程序坏了"。在这里说清楚。
            raise ValueError(
                "MODEL_API_KEY 还是 .env.example 里的占位符 —— "
                "先填上你自己的密钥再跑（见 .env.example 顶部的三步）"
            )
        if _is_placeholder(self.model_base_url):
            # 地址里的占位符（比如 <workspace-id>）—— 请求会直接 DNS 失败，
            # 同样表现为"每题都蒙"。一样在这里拦住。
            raise ValueError(
                "MODEL_BASE_URL 还是 .env.example 里的占位符 —— "
                "先填上你供应商的实际地址再跑（见 .env.example 顶部的三步）"
            )

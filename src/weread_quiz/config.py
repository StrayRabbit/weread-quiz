"""配置加载。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

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


@dataclass
class AppConfig:
    """运行时配置，来自 .env / 环境变量。"""

    model_base_url: str
    model_api_key: str
    model_name: str
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

    @classmethod
    def from_env(cls) -> "AppConfig":
        load_dotenv(PROJECT_ROOT / ".env", override=False)

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

        return cls(
            model_base_url=os.getenv("MODEL_BASE_URL", ""),
            model_api_key=os.getenv("MODEL_API_KEY", ""),
            model_name=os.getenv("MODEL_NAME", ""),
            vision_retries=vision_retries,
            guess_on_failure=_env_flag("GUESS_ON_FAILURE", True),
            model_thinking=_env_flag("MODEL_THINKING", True),
            ocr_min_confidence=ocr_min_confidence,
            daily_ocr_min_confidence=daily_ocr_min_confidence,
            pk_next_question_delay=pk_next_question_delay,
            pk_ocr_fallback_seconds=pk_ocr_fallback_seconds,
        )

    def validate_model(self) -> None:
        """模型配置是答题必需项，缺失时立即报错而不是等到调用时。"""
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
            raise ValueError(f"缺少必需的环境变量：{', '.join(missing)}（见 .env.example）")

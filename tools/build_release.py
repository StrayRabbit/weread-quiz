"""构建免安装 Windows 包。

产物是一个 zip，解压即用 —— 不需要装 Python，也不需要 uv。

结构::

    weread-quiz-v<版本>-win64/
        python/          便携 CPython + 按 uv.lock 装好的依赖 + 本项目
        启动*.bat        三个启动器（自动识别免安装模式）
        首次配置.bat     复制 .env.example 到 .env 并用记事本打开
        .env.example
        使用说明.md
        LICENSE
        assets/

依赖版本来自 ``uv.lock``（``uv export --locked``），构建机装了什么发布出去就是什么。
便携 Python 的版本固定在 PYTHON_SPEC —— 换版本要重新实机验证一遍再改。

用法::

    uv run python tools/build_release.py            # 构建 + 自检
    uv run python tools/build_release.py --no-verify # 跳过自检
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build"
DIST = ROOT / "dist"

# 便携 Python 版本 —— 固定住，换版本必须重新实机验证。
PYTHON_SPEC = "3.11.17"

# 便携 Python 里用不到、可以剪掉的部分。剪完由 --verify 兜底检查。
STRIP_PATTERNS = [
    "include",
    "tcl",
    "Lib/idlelib",
    "Lib/ensurepip",
    "Lib/pydoc_data",
    "Lib/site-packages/pip",
    "Lib/site-packages/pip-*.dist-info",
    "Scripts/pip*.exe",
]

LAUNCHER_NAMES = [
    "启动PK.bat",
    "启动每日一答-瓜分天数.bat",
    "启动每日一答-瓜分书币.bat",
    "首次配置.bat",
]


def _force_utf8_output() -> None:
    """构建日志里有中文 —— 控制台默认 GBK 会编不出来。"""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001 - 换不了编码就按原样跑
            pass


def run(cmd: list[str]) -> None:
    print(">", " ".join(str(c) for c in cmd), flush=True)
    env = dict(os.environ)
    # 子进程要打印中文，Windows 默认 GBK 控制台会编不出来。
    env.setdefault("PYTHONIOENCODING", "utf-8")
    subprocess.run([str(c) for c in cmd], check=True, cwd=ROOT, env=env)


def project_version() -> str:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"([^"]+)"', text, re.M)
    if not match:
        raise SystemExit("pyproject.toml 里找不到 version")
    return match.group(1)


def portable_python_dir() -> Path:
    """build/python 下真正的版本目录（跳过 cpython-3.11-... 那个别名链接）。"""
    base = BUILD / "python"
    if not base.is_dir():
        raise SystemExit(f"{base} 不存在，先跑 uv python install")
    candidates: list[tuple[tuple[int, ...], Path]] = []
    for path in base.iterdir():
        match = re.fullmatch(r"cpython-(\d+)\.(\d+)\.(\d+)-windows-x86_64-none", path.name)
        if match and path.is_dir():
            candidates.append((tuple(int(g) for g in match.groups()), path))
    if not candidates:
        raise SystemExit(f"{base} 下找不到便携 Python")
    return max(candidates)[1]


def install_portable_python() -> Path:
    print("== 准备便携 Python ==", flush=True)
    run([
        "uv", "python", "install", PYTHON_SPEC,
        "--install-dir", BUILD / "python",
        "--no-bin", "--no-registry",
    ])
    return portable_python_dir()


def stage_python(src: Path, dest: Path) -> Path:
    print("== 拷贝便携 Python 到暂存目录 ==", flush=True)
    if dest.exists():
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dest)
    # uv 的托管 Python 带 PEP 668 标记，装包会被拒。
    # 这份拷贝是我们自己的产物，标记去掉后 uv / pip 都能正常装。
    marker = dest / "Lib" / "EXTERNALLY-MANAGED"
    if marker.exists():
        marker.unlink()
        print("   - 移除 Lib/EXTERNALLY-MANAGED（PEP 668 标记）")
    return dest / "python.exe"


def export_requirements(out: Path) -> Path:
    print("== 导出锁定依赖 ==", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    run([
        "uv", "export", "--format", "requirements-txt",
        "--no-hashes", "--no-emit-project", "--locked",
        "-o", out,
    ])
    return out


def install_packages(pyexe: Path, req: Path) -> None:
    print("== 安装依赖（按 uv.lock） ==", flush=True)
    run(["uv", "pip", "install", "--python", pyexe, "-r", req])

    print("== 安装本项目（非 editable，资源和代码随包走） ==", flush=True)
    run([
        "uv", "pip", "install", "--python", pyexe,
        "--no-editable", "--no-deps", "--reinstall-package", "weread-quiz",
        ROOT,
    ])


def strip_python(py_dir: Path) -> None:
    print("== 剪掉用不到的部分 ==", flush=True)
    for pattern in STRIP_PATTERNS:
        for path in py_dir.glob(pattern):
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            else:
                path.unlink(missing_ok=True)
            print(f"   - {path.relative_to(py_dir)}")


def copy_launchers(dest: Path) -> None:
    print("== 生成启动器 ==", flush=True)
    run([sys.executable, ROOT / "tools" / "write_launchers.py"])
    for name in LAUNCHER_NAMES:
        shutil.copy2(ROOT / name, dest / name)


def copy_support_files(dest: Path) -> None:
    print("== 拷贝附带文件 ==", flush=True)
    shutil.copy2(ROOT / ".env.example", dest / ".env.example")
    shutil.copy2(ROOT / "LICENSE", dest / "LICENSE")
    (dest / "assets").mkdir(exist_ok=True)
    for asset in sorted((ROOT / "assets").glob("*")):
        if asset.is_file():
            shutil.copy2(asset, dest / "assets" / asset.name)


def write_guide(dest: Path, version: str) -> None:
    (dest / "使用说明.md").write_text(
        GUIDE_TEMPLATE.format(version=version), encoding="utf-8"
    )


def write_release_notes(version: str) -> Path:
    """从 CHANGELOG.md 里取出本版本的段落，写成 Release 正文。

    不用 GitHub 的 "generate release notes" —— 仓库的 commit message 不规范，
    自动生成出来没法看。
    """
    changelog = ROOT / "CHANGELOG.md"
    text = changelog.read_text(encoding="utf-8")
    pattern = re.compile(
        rf"^##\s+v?{re.escape(version)}\b.*?(?=^##\s|\Z)", re.M | re.S
    )
    match = pattern.search(text)
    if not match:
        raise SystemExit(f"CHANGELOG.md 里没有 v{version} 的段落")
    body = match.group(0).strip()
    # 去掉段落标题行 —— Release 页面自带标题。
    body = body.split("\n", 1)[1].strip()
    out = DIST / "release-notes.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(body + "\n", encoding="utf-8")
    print(f"== Release 说明：{out} ==", flush=True)
    return out


def make_zip(stage: Path, out: Path) -> Path:
    print("== 打包 zip ==", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    # 顶层保留一个目录 —— 解压后不会把文件散落到当前目录。
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in sorted(stage.rglob("*")):
            if path.is_dir():
                continue
            zf.write(path, path.relative_to(stage.parent))
    return out


def verify(stage: Path, zip_path: Path) -> None:
    """实机自检：暂存目录能跑，zip 解压后也能跑。"""
    print("== 自检 ==", flush=True)
    pyexe = stage / "python" / "python.exe"

    # 项目根目录要落在解压目录里，.env 和 data/ 都靠它。
    probe = (
        "import weread_quiz.config as c;"
        "print(c.PROJECT_ROOT);"
        "print(c.RUNS_DIR)"
    )
    env = dict(os.environ)
    env["WEREQ_HOME"] = str(stage)
    print("-- 暂存目录 --", flush=True)
    subprocess.run(
        [str(pyexe), "-c", probe], check=True, cwd=stage, env=env,
    )
    subprocess.run(
        [str(pyexe), "-m", "weread_quiz", "--help"], check=True, cwd=stage, env=env,
    )

    print("-- 解压后 --", flush=True)
    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tmp)
        extracted = next(Path(tmp).iterdir())
        env2 = dict(os.environ)
        env2["WEREQ_HOME"] = str(extracted)
        subprocess.run(
            [str(extracted / "python" / "python.exe"), "-c", probe],
            check=True, cwd=extracted, env=env2,
        )
        subprocess.run(
            [str(extracted / "python" / "python.exe"), "-m", "weread_quiz", "--help"],
            check=True, cwd=extracted, env=env2,
        )


def main() -> int:
    _force_utf8_output()
    parser = argparse.ArgumentParser(description="构建免安装 Windows 包")
    parser.add_argument("--no-verify", action="store_true", help="跳过解压自检")
    args = parser.parse_args()

    version = project_version()
    pkg_name = f"weread-quiz-v{version}-win64"
    stage = BUILD / "stage" / pkg_name
    print(f"版本 {version} → {pkg_name}", flush=True)

    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)

    src_python = install_portable_python()
    pyexe = stage_python(src_python, stage / "python")
    req = export_requirements(BUILD / "requirements.txt")
    install_packages(pyexe, req)
    strip_python(stage / "python")
    copy_launchers(stage)
    copy_support_files(stage)
    write_guide(stage, version)

    zip_path = make_zip(stage, DIST / f"{pkg_name}.zip")
    size_mb = zip_path.stat().st_size / 1024 / 1024
    print(f"== 完成：{zip_path}（{size_mb:.1f} MB） ==", flush=True)

    write_release_notes(version)

    if not args.no_verify:
        verify(stage, zip_path)
    return 0


GUIDE_TEMPLATE = """# 微信读书自动答题工具 · 免安装版 v{version}

解压即用 —— **不需要装 Python，不需要装 uv**。

---

## 三步跑起来

### 1. 配置模型（一次性）

双击 **`首次配置.bat`**。它会把 `.env.example` 复制成 `.env` 并用记事本打开。

在 `.env` 顶部的「必填」一栏填三行：

```ini
MODEL_BASE_URL=https://api.minimaxi.com/v1
MODEL_API_KEY=sk-你的密钥
MODEL_NAME=MiniMax-Text-01
```

要求：端点是 **OpenAI 兼容** 的，且**需要支持图片输入**（本地 OCR 读不准时会走整图兜底）。
DeepSeek / 智谱 GLM / Kimi / 硅基流动（Qwen2.5-VL）都可以，换供应商只改这三行。

> 密钥那一行如果还写着 `sk-xxxxxxxx`，程序会拒绝启动并提醒你 ——
> 它不会默默跑起来然后每题都蒙。

> 填完记得保存记事本里的 `.env`（Ctrl+S），然后关掉记事本。

**想更快？** `.env` 里的 `MODEL_BASE_URL` / `MODEL_NAME` 换成决策模型的地址和名字就行
（阿里百炼的 `decision-model-preview`，或 Jev 的 `jev-latest`）—— 程序会自己认出
它走的协议。它快 14 倍（78ms 对 1094ms），代价是**读不了图**：本地 OCR 读不出题时
没有模型能兜底，那题只能蒙一个或停下。

**两个都想要？** 决策模型填在上面三行，再把 `.env` 里注释掉的
`VISION_BASE_URL` / `VISION_API_KEY` / `VISION_NAME` 三行取消注释，填上你的视觉模型
（比如 DeepSeek）。这样决策模型负责判下标，**拿不准（置信度 < 0.8）自动转给视觉
模型复核**，本地 OCR 读不出题时也由视觉模型读图 —— 实测答对率 85.5% → 98.6%。

### 2. 打开微信读书小程序

在**微信**里打开微信读书小程序，停在游戏中心入口页。
小程序窗口**不能被遮挡**，要留在最外层（程序会尝试自动置顶，跑完还原）。

直达链接（`#小程序://` 协议在浏览器里点不开，复制到微信「文件传输助手」再点开）：

| 入口 | 链接 |
|---|---|
| 问答 PK | `#小程序://微信读书/bJK5TaSZ3rUpu1D` |
| 每日一答 | `#小程序://微信读书/lI9387D9hnbmJnk` |

### 3. 双击启动器

| 启动器 | 作用 |
|---|---|
| `启动PK.bat` | 问答 PK：自动匹配 → 答完整局 |
| `启动每日一答-瓜分天数.bat` | 每日一答：瓜分天数路线 → 答完 12 题 |
| `启动每日一答-瓜分书币.bat` | 每日一答：瓜分书币路线 → 答完 12 题 |

**随时按 Esc 键中止。**

---

## 常见问题

**Q：双击启动器后窗口一闪就没了 / 什么都没发生**
A：看窗口里的提示。最常见的是 `.env` 没配 —— 重新双击 `首次配置.bat` 填好三行。

**Q：Windows 弹出「Windows 已保护你的电脑」**
A：点「更多信息」→「仍要运行」。这是因为 Python 运行时没有数字签名，
和你的系统无关，只在第一次运行时出现。

**Q：能不能先不点？**
A：可以。在 `python` 文件夹里按住 Shift 右键 →「在此处打开 PowerShell 窗口」，然后：

```powershell
.\\python\\python.exe -m weread_quiz pk --dry-run
```

`--dry-run` 只识别不点击，第一次建议先跑这个。

**Q：截图和日志在哪？**
A：解压目录下的 `data/` 目录，每题一张截图 + 一份 `run.log`。
命令行清理：`python\\python.exe -m weread_quiz clean`（默认保留最近 30 天）。

**Q：只支持 Windows？**
A：是。依赖 mss 抓屏 / RapidOCR 本地识别 / pyautogui 点击。

**Q：会不会用我的道具？**
A：不会。跳题卡 / 换题卡 / 续命卡一律不碰，也不会自动点「取消匹配」或「邀请好友」。

---

## 已知限制

- 仅 Windows 10 / 11。
- 强依赖小程序界面的几何特征（按钮边框颜色、题干区域），**小程序改版后可能要调参数**。
- 每日一答**答错即当日机会用完**，不可恢复。
- 视觉模型偶尔 6~8 秒慢响应，PK 倒计时 12 秒，个别题会超时蒙一个
  （蒙答有 25~33% 机会，想关掉就把 `.env` 里的 `GUESS_ON_FAILURE` 改成 `0`）。
- 不要同时跑两个启动器 —— 两个进程抢桌面会互相干扰。
- 显示缩放 100% / 125% / 150% / 200% 都支持，不用调。

---

## 一局大概烧多少 token

| 一局 | 请求次数 | 输入 | 输出 | 合计 |
|---|---|---|---|---|
| 问答 PK（10 题） | 12 | 3.08K | 596 | **3.68K** |
| 每日一答（12 题） | 13 | 1.72K | 1.85K | **3.57K** |

模型只判「哪个选项对」，主路径每题只发一次纯文本请求。

配了视觉模型复核的话，约三成的题会多发一次复核请求 —— token 和时间都会涨一点。

---

## 想调参数？

`.env` 里除了必填的三行，还可以加这些 —— 不写就是默认值，一般不用动：

| 变量 | 默认 | 说明 |
|------|------|------|
| `VISION_RETRIES` | `0` | 视觉模型失败后重试几次。PK 时间预算紧，重试容易超时 |
| `GUESS_ON_FAILURE` | `1` | 答不出来时是否点第一个按钮蒙一个（25~33% 机会） |
| `MODEL_THINKING` | `1` | 让视觉模型输出思考块。关掉快 312ms/题，但 10 道真实题里 2 道答案会变 —— 别关 |
| `MODEL_MIN_CONFIDENCE` | 留空 | 复核门槛：决策模型置信度低于它就转给视觉模型复核。留空 = 自动（配了视觉模型用 0.8，没配不复核）；`0` = 不复核 |
| `OCR_MIN_CONFIDENCE` | `0.7` | PK 本地 OCR 置信度阈值，低于阈值转整图路径 |
| `DAILY_OCR_MIN_CONFIDENCE` | `0.85` | daily 的阈值，独立于 PK |
| `PK_NEXT_QUESTION_DELAY` | `1.0` | PK 点击后等下一题的固定延迟（秒） |
| `PK_OCR_FALLBACK_SECONDS` | `3.0` | PK 本地 OCR 连续判否后，允许一次视觉兜底的等待时间（秒） |

改完重新启动生效。

---

## 联系方式

源码与更新：https://github.com/StrayRabbit/weread-quiz

本项目仅供学习研究使用，如有侵权，请联系删除。
第三方模型和微信相关素材的权利不因本项目 LICENSE 而改变。
"""


if __name__ == "__main__":
    sys.exit(main())

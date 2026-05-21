"""Python 启动钩子：自动激活 benchmark 扩展层。

Python 启动时会自动 import sys.path 上首个名为 sitecustomize 的模块（PEP 370 / site.py）。
项目根加入 sys.path 后（运行 ``python -m benchmark.cli ...`` 时自动加入），本文件触发
``benchmark.bootstrap`` 的所有 monkey patch 与 Skill 注册，从而在不修改 main 任何原文件、
不修改 CLI 命令的前提下让默认行为等价于 wsy_skill_dev。

异常被静默：缺依赖、未安装 benchmark 包等环境异常不应阻塞 Python 启动。
"""

try:
    import benchmark.bootstrap  # noqa: F401
except Exception:  # pragma: no cover - environment-dependent
    pass

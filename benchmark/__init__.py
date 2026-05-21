__version__ = "0.3.0"

try:  # 自动激活 wsy_skill_dev 扩展层；缺包/未安装时静默
    from . import bootstrap as _bootstrap  # noqa: F401
except Exception:  # pragma: no cover
    pass

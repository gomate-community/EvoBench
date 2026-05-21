"""pytest 启动钩子：确保 ``benchmark.bootstrap`` 在所有测试前被激活。

pytest 不一定经过项目根 sitecustomize.py（取决于运行目录），此处显式 import 兜底。
"""

import benchmark.bootstrap  # noqa: F401

"""
WeaverAgent Backend 启动入口
"""

import os
import sys

# 解决 Windows 控制台中文乱码问题：在所有导入之前设置 UTF-8 编码
if sys.platform == 'win32':
    # 设置环境变量确保 Python 使用 UTF-8
    os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
    # 重新配置标准输出流为 UTF-8
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── graphrag editable 安装兜底 ──────────────────────────────
# uv/pip 创建的 editable .pth 文件在某些环境下（此 venv base 指向 anaconda 时）
# 不会被 site.py 处理，导致 `import graphrag` 失败。这里显式把 monorepo
# 下的 8 个源目录加入 sys.path，与 .pth 写入的路径完全一致。
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_GRAPHRAG_PKGS = [
    "graphrag-common", "graphrag-cache", "graphrag-storage", "graphrag-input",
    "graphrag-chunking", "graphrag-vectors", "graphrag-llm", "graphrag",
]
for _pkg in _GRAPHRAG_PKGS:
    _p = os.path.join(_ROOT, "graphrag", "packages", _pkg)
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from app import create_app
from app.config import Config


def main():
    """主函数"""
    # 验证配置
    errors = Config.validate()
    if errors:
        print("配置错误:")
        for err in errors:
            print(f"  - {err}")
        print("\n请检查 .env 文件中的配置")
        sys.exit(1)
    
    # 创建应用
    app = create_app()
    
    # 获取运行配置
    host = os.environ.get('FLASK_HOST', '0.0.0.0')
    port = int(os.environ.get('FLASK_PORT', 5001))
    debug = Config.DEBUG
    
    # 启动服务
    app.run(host=host, port=port, debug=debug, threaded=True)


if __name__ == '__main__':
    main()


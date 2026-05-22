"""
Compatibility entrypoint for manually generating Vue chart data.

The canonical implementation lives in vue_data_service.py. Keeping this
wrapper lets older manual commands continue to work without maintaining two
copies of the generation logic.
"""
from pathlib import Path
import sys


FASTAPI_ROOT = Path(__file__).resolve().parents[3]
if str(FASTAPI_ROOT) not in sys.path:
    sys.path.insert(0, str(FASTAPI_ROOT))

from app.services.data.vue_data_service import *  # noqa: F401,F403
from app.services.data.vue_data_service import generate_frontend_data


if __name__ == "__main__":
    from datetime import datetime

    print("=" * 60)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 手动执行前端数据生成任务")
    print("=" * 60)

    result = generate_frontend_data()

    print("\n" + "=" * 60)
    if result:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 执行结果：成功")
    else:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 执行结果：失败")
    print("=" * 60)

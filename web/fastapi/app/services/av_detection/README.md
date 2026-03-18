# AV检测模块文件说明

## 文件位置

本模块已将AV检测相关文件集成到FastAPI项目中:

```
fastapi/
└── app/
    └── services/
        └── av_detection/
            ├── __init__.py              # 模块初始化文件
            ├── AV_Distributed_Client.py  # 分布式杀毒客户端
            └── vm_config.json            # 虚拟机配置文件
```

## 文件说明

### 1. AV_Distributed_Client.py
- **来源:** `AV_Detection_and_Probing-main/AV_Distributed_Client.py`
- **功能:** 分布式杀毒扫描客户端,支持多虚拟机、多引擎并行检测
- **主要类:** `AVDistributedClient`

### 2. vm_config.json
- **来源:** `AV_Detection_and_Probing-main/vm_config.json`
- **功能:** 虚拟机配置文件,包含9个虚拟机的IP、端口和引擎信息
- **配置内容:**
  - vm1 (192.168.8.19:27483): Avira, McAfee, WindowsDefender, IkarusT3, Emsisoft, FProtect, Vba32
  - vm2 (192.168.8.20:27483): ClamAV
  - vm3 (192.168.8.21:27483): Kaspersky
  - vm4 (192.168.8.22:27483): ESET
  - vm5 (192.168.8.23:27483): DrWeb
  - vm6 (192.168.8.24:27483): Avast
  - vm7 (192.168.8.25:27483): AVG
  - vm8 (192.168.8.29:27483): AdAware
  - vm9 (192.168.8.30:27483): FSecure

### 3. __init__.py
- **功能:** 使av_detection目录成为Python包,方便导入

## 使用方式

在API中导入使用:

```python
from app.services.av_detection import AVDistributedClient
from pathlib import Path

# 获取配置文件路径
config_path = Path(__file__).parent.parent / "services" / "av_detection" / "vm_config.json"

# 初始化客户端
av_client = AVDistributedClient(config_path=str(config_path))

# 使用客户端进行扫描
result = av_client.scan_single_file(file_path)
```

## 配置修改

如需修改虚拟机配置,请编辑 `vm_config.json` 文件:

```json
{
  "virtual_machines": [
    {
      "id": "vm1",
      "ip": "192.168.8.19",
      "port": 27483,
      "engines": ["Avira", "McAfee", "WindowsDefender", ...]
    },
    ...
  ]
}
```

## 注意事项

1. **文件同步:** 如果原始 `AV_Detection_and_Probing-main` 目录中的文件有更新,需要手动同步到本目录
2. **配置一致性:** 确保 `vm_config.json` 中的虚拟机IP和端口与实际环境一致
3. **网络连接:** 确保FastAPI服务器能够访问虚拟机网络(192.168.8.0/24)

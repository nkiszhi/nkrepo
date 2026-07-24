VMRUN = r"C:\Program Files (x86)\VMware\VMware Workstation\vmrun.exe"

# VM 内 AV_Scan_Server.py 路径（VMware 共享文件夹 UNC）
GUEST_AV_SERVER = r"\\vmware-host\Shared Folders\共享\AV_Detection_and_Probing-main\AV_Detection_and_Probing-main\AV_Scan_Server.py"
GUEST_REPO_DIR = r"\\vmware-host\Shared Folders\共享\AV_Detection_and_Probing-main\AV_Detection_and_Probing-main"
GUEST_WORK_DIR = r"\\vmware-host\Shared Folders\共享"

VMS = [
    {
        "id": "vm1", "name": "vm1",
        "vmx": r"C:\Users\34701\Documents\Virtual Machines\001\001.vmx",
        "ip": "192.168.8.100", "port": 27483,
        "user": "001", "pass": "123456", "interactive": False,
        "python": r"C:\Users\001\AppData\Local\Programs\Python\Python313\python.exe",
        "engines": ["Avira", "McAfee", "WindowsDefender", "IkarusT3", "Emsisoft", "FProtect", "Vba32"],
    },
    {
        "id": "vm2", "name": "vm2",
        "vmx": r"C:\Users\34701\Documents\Virtual Machines\003\003.vmx",
        "ip": "192.168.8.101", "port": 27483,
        "user": "003", "pass": "123456", "interactive": False,
        "python": r"C:\Users\003\AppData\Local\Programs\Python\Python313\python.exe",
        "engines": ["ClamAV"],
    },
    {
        "id": "vm3", "name": "vm3",
        "vmx": r"C:\Users\34701\Documents\Virtual Machines\004\004.vmx",
        "ip": "192.168.8.102", "port": 27483,
        "user": "004", "pass": "123456", "interactive": True,
        "python": r"C:\Users\004\AppData\Local\Programs\Python\Python313\python.exe",
        "engines": ["Kaspersky"],
    },
    {
        "id": "vm4", "name": "vm4",
        "vmx": r"C:\Users\34701\Documents\Virtual Machines\005\005.vmx",
        "ip": "192.168.8.113", "port": 27483,
        "user": "005", "pass": "123456", "interactive": True,
        "python": r"C:\Users\005\AppData\Local\Programs\Python\Python313\python.exe",
        "engines": ["ESET"],
    },
    {
        "id": "vm5", "name": "vm5",
        "vmx": r"C:\Users\34701\Documents\Virtual Machines\006\006.vmx",
        "ip": "192.168.8.104", "port": 27483,
        "user": "006", "pass": "123456", "interactive": True,
        "python": r"C:\Users\006\AppData\Local\Programs\Python\Python313\python.exe",
        "engines": ["DrWeb"],
    },
    {
        "id": "vm6", "name": "vm6",
        "vmx": r"C:\Users\34701\Documents\Virtual Machines\007\007.vmx",
        "ip": "192.168.8.105", "port": 27483,
        "user": "007", "pass": "123456", "interactive": True,
        "python": r"C:\Users\007\AppData\Local\Programs\Python\Python313\python.exe",
        "engines": ["Avast"],
    },
    {
        "id": "vm7", "name": "vm7",
        "vmx": r"C:\Users\34701\Documents\Virtual Machines\008\008.vmx",
        "ip": "192.168.8.106", "port": 27483,
        "user": "008", "pass": "123456", "interactive": True,
        "python": r"C:\Users\008\AppData\Local\Programs\Python\Python313\python.exe",
        "engines": ["AVG"],
    },
    {
        "id": "vm8", "name": "vm8",
        "vmx": r"C:\Users\34701\Documents\Virtual Machines\009\009.vmx",
        "ip": "192.168.8.107", "port": 27483,
        "user": "009", "pass": "123456", "interactive": False,
        "python": r"C:\Users\009\AppData\Local\Programs\Python\Python313\python.exe",
        "engines": ["AdAware"],
    },
    {
        "id": "vm9", "name": "vm9",
        "vmx": r"C:\Users\34701\Documents\Virtual Machines\010\010.vmx",
        "ip": "192.168.8.108", "port": 27483,
        "user": "010", "pass": "123456", "interactive": False,
        "python": r"C:\Users\010\AppData\Local\Programs\Python\Python313\python.exe",
        "engines": ["FSecure"],
    },
]

# 全 15 引擎列表
ALL_ENGINES = [
    "Avira", "McAfee", "WindowsDefender", "ClamAV", "Kaspersky",
    "ESET", "DrWeb", "IkarusT3", "Emsisoft", "Avast",
    "AVG", "FProtect", "AdAware", "Vba32", "FSecure",
]

def get_vm_by_id(vm_id: str) -> dict | None:
    for vm in VMS:
        if vm["id"] == vm_id:
            return vm
    return None

def get_vm_by_ip(ip: str) -> dict | None:
    for vm in VMS:
        if vm["ip"] == ip:
            return vm
    return None

def get_engine_vm(engine_name: str) -> dict | None:
    for vm in VMS:
        if engine_name in vm["engines"]:
            return vm
    return None

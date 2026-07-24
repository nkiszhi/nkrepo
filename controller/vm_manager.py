"""
VM 生命周期管理器 — 通过 vmrun.exe 管理 9 台 VMware 虚拟机
支持：启动/停止/快照回滚/部署代码/健康检查/确保 AV_Scan_Server 运行
用法：
  python vm_manager.py start-all        # 启动全部 VM 并启动扫描服务
  python vm_manager.py stop-all         # 停止全部 VM
  python vm_manager.py health           # 健康检查
  python vm_manager.py deploy           # 部署 repo.zip 到所有 VM
  python vm_manager.py ensure-server    # 确保所有 VM 的 AV_Scan_Server 在跑
  python vm_manager.py revert          # 回滚所有 VM 到干净快照
"""

import subprocess
import sys
import time
import re
from pathlib import Path
from datetime import datetime

import requests

from vm_config_full import VMS, VMRUN, GUEST_AV_SERVER, GUEST_REPO_DIR, GUEST_WORK_DIR

HERE = Path(__file__).resolve().parent
REPO_ZIP = str(HERE / "repo.zip")
POWERSHELL = r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe"


def _run(cmd: list[str], timeout: int = 120) -> subprocess.CompletedProcess:
    """执行本地命令，返回 CompletedProcess"""
    return subprocess.run(
        cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=timeout,
    )


def _vmrun_cmd(vm: dict, *args: str, interactive: bool = None) -> list[str]:
    """拼接 vmrun.exe 命令"""
    cmd = [VMRUN, "-T", "ws", "-gu", vm["user"], "-gp", vm["pass"]]
    args_list = list(args)
    # 决定是否加 -interactive
    use_interactive = interactive if interactive is not None else vm.get("interactive", False)
    if use_interactive and args_list and args_list[0] in ("runProgramInGuest", "runScriptInGuest"):
        args_list.insert(2, "-interactive")
    cmd.extend(args_list)
    return cmd


def _vmrun(vm: dict, *args: str, timeout: int = 120, interactive: bool = None) -> subprocess.CompletedProcess:
    """执行 vmrun 命令"""
    return _run(_vmrun_cmd(vm, *args, interactive=interactive), timeout=timeout)


class VMManager:
    """管理一组 VMware 虚拟机"""

    def __init__(self, vms: list[dict] = None):
        self.vms = vms or VMS

    def _find(self, vm_key: str) -> dict:
        """按 id 或 name 查找 VM"""
        for vm in self.vms:
            if vm["id"] == vm_key or vm["name"] == vm_key:
                return vm
        raise ValueError(f"VM not found: {vm_key}")

    # ---- 基础 vmrun 操作 ------------------------------------------------

    def start_vm(self, vm_key: str, gui: bool = False) -> bool:
        """无头启动 VM。vmrun start <vmx> [gui|nogui]"""
        vm = self._find(vm_key)
        mode = "gui" if gui else "nogui"
        r = _vmrun(vm, "start", vm["vmx"], mode, timeout=300)
        return r.returncode == 0

    def stop_vm(self, vm_key: str, mode: str = "soft") -> bool:
        """停止 VM。vmrun stop <vmx> [soft|hard]"""
        vm = self._find(vm_key)
        r = _vmrun(vm, "stop", vm["vmx"], mode, timeout=120)
        return r.returncode == 0

    def reset_vm(self, vm_key: str, mode: str = "soft") -> bool:
        """重置 VM。vmrun reset <vmx> [soft|hard]"""
        vm = self._find(vm_key)
        r = _vmrun(vm, "reset", vm["vmx"], mode, timeout=120)
        return r.returncode == 0

    def revert_to_snapshot(self, vm_key: str, snapshot_name: str = "clean") -> bool:
        """回滚 VM 到指定快照。vmrun revertToSnapshot <vmx> <name>"""
        vm = self._find(vm_key)
        r = _vmrun(vm, "revertToSnapshot", vm["vmx"], snapshot_name, timeout=60)
        return r.returncode == 0

    def list_snapshots(self, vm_key: str) -> list[str]:
        """列出 VM 所有快照。vmrun listSnapshots <vmx>"""
        vm = self._find(vm_key)
        r = _vmrun(vm, "listSnapshots", vm["vmx"], timeout=30)
        if r.returncode != 0:
            return []
        # 输出格式: "Total snapshots: N\nsnapshot1\nsnapshot2\n..."
        lines = r.stdout.strip().splitlines()
        return [l.strip() for l in lines if l.strip() and not l.startswith("Total")]

    def is_vm_running(self, vm_key: str) -> bool:
        """检查 VM 是否在运行。vmrun list"""
        vm = self._find(vm_key)
        r = _run([VMRUN, "list"], timeout=15)
        # 输出每行是一个正在运行的 VMX 路径
        return vm["vmx"] in r.stdout

    def get_vm_ip(self, vm_key: str) -> str | None:
        """获取 VM IP 地址。vmrun getGuestIPAddress <vmx> -wait"""
        vm = self._find(vm_key)
        r = _vmrun(vm, "getGuestIPAddress", vm["vmx"], "-wait", timeout=60)
        if r.returncode != 0:
            return None
        # 输出是一行 IP 地址
        match = re.search(r"\d+\.\d+\.\d+\.\d+", r.stdout)
        return match.group(0) if match else None

    def run_in_guest(self, vm_key: str, program: str, *args: str,
                     interactive: bool = None) -> subprocess.CompletedProcess:
        """在 VM 内执行程序。vmrun runProgramInGuest <vmx> <program> [args...]"""
        vm = self._find(vm_key)
        return _vmrun(vm, "runProgramInGuest", vm["vmx"], program, *args,
                      timeout=300, interactive=interactive)

    def copy_to_guest(self, vm_key: str, host_path: str, guest_path: str) -> bool:
        """复制文件到 VM。vmrun copyFileFromHostToGuest <vmx> <host> <guest>"""
        vm = self._find(vm_key)
        r = _vmrun(vm, "copyFileFromHostToGuest", vm["vmx"], host_path, guest_path, timeout=120)
        return r.returncode == 0

    def copy_from_guest(self, vm_key: str, guest_path: str, host_path: str) -> bool:
        """从 VM 复制文件。vmrun copyFileFromGuestToHost <vmx> <guest> <host>"""
        vm = self._find(vm_key)
        r = _vmrun(vm, "copyFileFromGuestToHost", vm["vmx"], guest_path, host_path, timeout=120)
        return r.returncode == 0

    def ensure_guest_dir(self, vm_key: str, path: str) -> bool:
        """确保 VM 内目录存在"""
        r = self.run_in_guest(vm_key, POWERSHELL,
                             "-Command", f"New-Item -ItemType Directory -Force '{path}' | Out-Null; exit 0")
        return True  # mkdir 可能返回 1 如果目录已存在，不算错误

    # ---- HTTP 健康检查 -------------------------------------------------

    def check_http_endpoint(self, vm_key: str, endpoint: str = "/engines") -> dict:
        """通过 HTTP 检查 VM 上 AV_Scan_Server 是否在线"""
        vm = self._find(vm_key)
        url = f"http://{vm['ip']}:{vm['port']}{endpoint}"
        try:
            r = requests.get(url, timeout=5)
            return {"ok": r.status_code == 200, "status_code": r.status_code, "body": r.json() if r.ok else None}
        except requests.ConnectionError:
            return {"ok": False, "error": "Connection refused"}
        except requests.Timeout:
            return {"ok": False, "error": "Timeout"}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def check_server_running(self, vm_key: str) -> bool:
        """检查 VM 上 AV_Scan_Server 是否在运行（HTTP :27483 可达）"""
        return self.check_http_endpoint(vm_key)["ok"]

    # ---- 部署 ----------------------------------------------------------

    def deploy_repo(self, vm_key: str, repo_zip_path: str = None) -> bool:
        """部署 repo.zip 到 VM → 解压到 C:\av_work\repo"""
        vm = self._find(vm_key)
        zip_path = repo_zip_path or REPO_ZIP

        if not Path(zip_path).exists():
            print(f"  [ERROR] repo.zip not found: {zip_path}")
            return False

        guest_zip = GUEST_WORK_DIR + r"\repo.zip"
        guest_repo = GUEST_WORK_DIR + r"\repo"

        # 1) 确保目录
        self.ensure_guest_dir(vm_key, GUEST_WORK_DIR)

        # 2) 复制 zip
        print(f"  copy zip → {vm['name']} ...")
        if not self.copy_to_guest(vm_key, zip_path, guest_zip):
            print(f"  [ERROR] copy zip failed")
            return False

        # 3) PowerShell 解压
        ps = (
            f"$ErrorActionPreference='Stop'; "
            f"New-Item -ItemType Directory -Force '{GUEST_WORK_DIR}' | Out-Null; "
            f"if (Test-Path '{guest_repo}') {{ Remove-Item -Recurse -Force '{guest_repo}' }}; "
            f"Expand-Archive -LiteralPath '{guest_zip}' -DestinationPath '{guest_repo}' -Force; "
            f"Write-Output 'DEPLOY_OK'"
        )
        r = self.run_in_guest(vm_key, POWERSHELL, "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps,
                             interactive=False)
        ok = r.returncode == 0 and "DEPLOY_OK" in r.stdout
        print(f"  deploy {'OK' if ok else 'FAILED'}  rc={r.returncode}")
        return ok

    def deploy_all(self, repo_zip_path: str = None):
        """部署到所有 VM"""
        print(f"\n{'='*60}")
        print(f"DEPLOY ALL  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")
        ok = 0
        for vm in self.vms:
            print(f"\n--- {vm['name']} ---")
            if self.deploy_repo(vm["id"], repo_zip_path):
                ok += 1
        print(f"\nDeploy done: {ok}/{len(self.vms)} OK")

    # ---- 启动服务 -------------------------------------------------------

    def ensure_server_running(self, vm_key: str) -> bool:
        """确保 VM 上 AV_Scan_Server.py 在运行。不在就通过 vmrun 启动。"""
        vm = self._find(vm_key)

        # 先快速检查
        if self.check_server_running(vm_key):
            print(f"  {vm['name']}: server already running")
            return True

        print(f"  {vm['name']}: starting AV_Scan_Server ...")
        python_path = vm['python']
        user_name = vm['user']
        # 1) 从共享文件夹拷到 VM 本地
        local_server = f"C:\\Users\\{user_name}\\Desktop\\AV_Scan_Server.py"
        local_cmd = f"C:\\Users\\{user_name}\\Desktop\\CMD_Detectors.py"
        r = self.run_in_guest(
            vm_key,
            POWERSHELL,
            "-Command",
            f"Copy-Item '{GUEST_AV_SERVER}' '{local_server}' -Force; "
            f"if (Test-Path '{GUEST_REPO_DIR}\\CMD_Detectors.py') {{ Copy-Item '{GUEST_REPO_DIR}\\CMD_Detectors.py' '{local_cmd}' -Force }}; "
            f"exit 0",
        )
        # 2) 后台启动
        r = self.run_in_guest(
            vm_key,
            POWERSHELL,
            "-Command",
            f"$p = Start-Process -FilePath '{python_path}' -ArgumentList '{local_server}' -PassThru; Write-Output ('PID=' + $p.Id); exit 0",
        )
        if r.returncode != 0:
            print(f"  [ERROR] start command failed, rc={r.returncode}")
            return False

        # 等待服务起来
        for i in range(10):
            time.sleep(2)
            if self.check_server_running(vm_key):
                print(f"  {vm['name']}: server started OK")
                return True
            print(f"  waiting ... ({i+1}/10)")

        print(f"  [WARN] {vm['name']}: server may not have started")
        return False

    def ensure_all_servers_running(self):
        """确保所有 VM 的 AV_Scan_Server 在运行"""
        print(f"\n{'='*60}")
        print(f"ENSURE SERVERS  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")
        ok = 0
        for vm in self.vms:
            print(f"\n--- {vm['name']} ---")
            # 先确认 VM 在运行
            if not self.is_vm_running(vm["id"]):
                print(f"  VM not running, starting ...")
                self.start_vm(vm["id"])
                time.sleep(15)  # 等 VM 启动
            if self.ensure_server_running(vm["id"]):
                ok += 1
        print(f"\nServers ready: {ok}/{len(self.vms)}")
        return ok

    # ---- 健康检查 -------------------------------------------------------

    def health_check(self, vm_key: str) -> dict:
        """综合健康检查：VM 是否运行 + AV_Scan_Server 是否在线"""
        vm = self._find(vm_key)
        running = self.is_vm_running(vm_key)
        http = self.check_http_endpoint(vm_key) if running else {"ok": False, "error": "VM not running"}

        return {
            "vm": vm["name"],
            "ip": vm["ip"],
            "vm_running": running,
            "server_ok": http["ok"],
            "server_detail": http,
            "engines": vm["engines"],
        }

    def health_check_all(self) -> list[dict]:
        """所有 VM 健康检查"""
        results = []
        for vm in self.vms:
            r = self.health_check(vm["id"])
            results.append(r)
            status = "🟢" if r["server_ok"] else ("🟡" if r["vm_running"] else "🔴")
            detail = r["server_detail"].get("error", "OK" if r["server_ok"] else "?")
            print(f"  {status} {r['vm']:5s}  running={r['vm_running']}  server={r['server_ok']}  ({detail})")
        return results

    # ---- 批量操作 -------------------------------------------------------

    def start_all(self, ensure_server: bool = True):
        """启动所有 VM，可选自动启动 AV_Scan_Server"""
        print(f"\n{'='*60}")
        print(f"START ALL  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")
        for vm in self.vms:
            print(f"\n--- {vm['name']} ---")
            if self.is_vm_running(vm["id"]):
                print(f"  already running")
            else:
                ok = self.start_vm(vm["id"])
                print(f"  start {'OK' if ok else 'FAILED'}")
                if ok:
                    time.sleep(10)  # 等 VM 启动

        if ensure_server:
            print("\n[等待 VM 完全启动后检查服务...]")
            time.sleep(15)
            self.ensure_all_servers_running()

    def stop_all(self, mode: str = "soft"):
        """停止所有 VM"""
        print(f"\n{'='*60}")
        print(f"STOP ALL  mode={mode}  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")
        for vm in self.vms:
            print(f"\n--- {vm['name']} ---")
            if not self.is_vm_running(vm["id"]):
                print(f"  not running")
                continue
            ok = self.stop_vm(vm["id"], mode)
            print(f"  stop {'OK' if ok else 'FAILED'}")

    def reset_all(self, mode: str = "soft"):
        """重置所有 VM"""
        for vm in self.vms:
            print(f"reset {vm['name']} ...")
            self.reset_vm(vm["id"], mode)

    def revert_all(self, snapshot: str = "clean"):
        """回滚所有 VM 到指定快照"""
        print(f"\n{'='*60}")
        print(f"REVERT ALL → snapshot='{snapshot}'  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")
        ok = 0
        for vm in self.vms:
            print(f"\n--- {vm['name']} ---")
            if self.is_vm_running(vm["id"]):
                print(f"  VM running, stopping first ...")
                self.stop_vm(vm["id"], "soft")
                time.sleep(5)
            r = self.revert_to_snapshot(vm["id"], snapshot)
            print(f"  revert {'OK' if r else 'FAILED'}")
            if r:
                ok += 1
        print(f"\nRevert done: {ok}/{len(self.vms)} OK")


# ============================================================
#  命令行入口
# ============================================================

def main():
    mgr = VMManager()

    if len(sys.argv) < 2:
        print(__doc__)
        return

    cmd = sys.argv[1]

    if cmd == "start-all":
        mgr.start_all()
    elif cmd == "stop-all":
        mgr.stop_all()
    elif cmd == "reset-all":
        mgr.reset_all()
    elif cmd == "revert":
        snapshot = sys.argv[2] if len(sys.argv) > 2 else "clean"
        mgr.revert_all(snapshot)
    elif cmd == "health":
        print(f"\nVM Health Check  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("-" * 60)
        results = mgr.health_check_all()
        ok = sum(1 for r in results if r["server_ok"])
        print(f"\nTotal: {ok}/{len(results)} servers online")
    elif cmd == "deploy":
        repo = sys.argv[2] if len(sys.argv) > 2 else None
        mgr.deploy_all(repo)
    elif cmd == "ensure-server":
        mgr.ensure_all_servers_running()
    elif cmd == "list-running":
        for vm in VMS:
            running = mgr.is_vm_running(vm["id"])
            print(f"  {vm['name']:5s}  {'running' if running else 'stopped'}")
    else:
        print(f"Unknown command: {cmd}")
        print(__doc__)


if __name__ == "__main__":
    main()

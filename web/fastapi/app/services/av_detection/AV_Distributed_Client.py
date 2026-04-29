"""
AV分布式扫描客户端 - 支持多虚拟机、多引擎并行检测
功能: 自动调用9个虚拟机上的15个杀毒引擎,返回综合检测结果
"""

import json
import requests
import concurrent.futures
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime


class AVDistributedClient:
    """分布式杀毒扫描客户端"""

    def __init__(self, config_path: str = "vm_config.json"):
        """
        初始化客户端

        Args:
            config_path: 虚拟机配置文件路径
        """
        self.config = self._load_config(config_path)
        self.engine_to_vm = self._build_engine_mapping()

    def _load_config(self, config_path: str) -> dict:
        """加载虚拟机配置"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"配置文件未找到: {config_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"配置文件格式错误: {e}")

    def _build_engine_mapping(self) -> dict:
        """构建引擎到虚拟机的映射关系"""
        mapping = {}
        for vm in self.config['virtual_machines']:
            for engine in vm['engines']:
                mapping[engine] = {
                    'vm_id': vm['id'],
                    'ip': vm['ip'],
                    'port': vm['port']
                }
        return mapping

    def _check_vm_available(self, vm: dict) -> bool:
        """检查虚拟机是否可用"""
        try:
            url = f"http://{vm['ip']}:{vm['port']}/engines"
            resp = requests.get(url, timeout=5)
            return resp.status_code == 200
        except Exception:
            return False

    def get_available_engines(self) -> List[str]:
        """获取所有可用的杀毒引擎"""
        return list(self.engine_to_vm.keys())

    def get_vm_status(self) -> Dict[str, bool]:
        """检查所有虚拟机状态"""
        status = {}
        for vm in self.config['virtual_machines']:
            vm_key = f"{vm['id']} ({vm['ip']}:{vm['port']})"
            status[vm_key] = self._check_vm_available(vm)
        return status

    def _scan_single_engine(
        self,
        engine: str,
        file_paths: List[str],
        timeout: int = 300
    ) -> Dict[str, Any]:
        """
        使用单个引擎扫描文件

        Args:
            engine: 杀毒引擎名称
            file_paths: 待扫描文件路径列表
            timeout: 超时时间(秒)

        Returns:
            扫描结果字典
        """
        if engine not in self.engine_to_vm:
            return {
                'engine': engine,
                'success': False,
                'error': f"引擎未配置: {engine}"
            }

        vm_info = self.engine_to_vm[engine]
        base_url = f"http://{vm_info['ip']}:{vm_info['port']}"

        try:
            # 准备文件
            files = []
            for file_path in file_paths:
                files.append(('files', (Path(file_path).name, open(file_path, 'rb'), 'application/octet-stream')))

            try:
                # 发送扫描请求
                resp = requests.post(
                    f"{base_url}/scan",
                    data={"engine": engine},
                    files=files,
                    timeout=timeout
                )
                resp.raise_for_status()
                result = resp.json()
                
                # 过滤结果：确保只返回指定引擎的结果
                # 有些虚拟机可能返回所有引擎结果，需要过滤
                if 'results' in result:
                    filtered_results = {}
                    for file_name, detection in result['results'].items():
                        # 如果返回的是字典（多引擎结果），只取当前引擎
                        if isinstance(detection, dict):
                            filtered_results[file_name] = detection.get(engine, -1)
                        else:
                            # 单引擎结果，直接使用
                            filtered_results[file_name] = detection
                    result['results'] = filtered_results
                
                result['success'] = True
                result['vm_id'] = vm_info['vm_id']
                return result

            finally:
                # 关闭文件句柄
                for file_tuple in files:
                    file_tuple[1][1].close()

        except requests.exceptions.Timeout:
            return {
                'engine': engine,
                'success': False,
                'error': f"扫描超时({timeout}秒)",
                'vm_id': vm_info['vm_id']
            }
        except Exception as e:
            return {
                'engine': engine,
                'success': False,
                'error': str(e),
                'vm_id': vm_info['vm_id']
            }

    def scan_files(
        self,
        file_paths: List[str],
        engines: List[str] = None,
        max_workers: int = 15,
        timeout: int = 60
    ) -> Dict[str, Any]:
        """
        使用多个引擎扫描文件

        Args:
            file_paths: 待扫描文件路径列表
            engines: 要使用的引擎列表,如果为None则使用所有可用引擎
            max_workers: 最大并发数
            timeout: 单个引擎超时时间(秒)

        Returns:
            综合扫描结果
        """
        if engines is None:
            engines = self.get_available_engines()
        
        # 添加日志：显示实际使用的引擎列表
        print(f"[AV_Distributed_Client] scan_files called with engines: {engines}")
        print(f"[AV_Distributed_Client] engines type: {type(engines)}, length: {len(engines) if engines else 0}")

        if not file_paths:
            return {'error': '未提供待扫描文件'}

        start_time = datetime.now()

        # 并行调用所有引擎
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_engine = {
                executor.submit(self._scan_single_engine, engine, file_paths, timeout): engine
                for engine in engines
            }

            for future in concurrent.futures.as_completed(future_to_engine):
                engine = future_to_engine[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    results.append({
                        'engine': engine,
                        'success': False,
                        'error': f"执行异常: {str(e)}"
                    })

        # 统计结果
        total_time = (datetime.now() - start_time).total_seconds()

        # 按文件汇总结果
        file_results = {}
        for file_path in file_paths:
            file_name = Path(file_path).name
            file_results[file_name] = {
                'engines': {},
                'detection_count': 0,
                'total_engines': 0,
                'malicious': False
            }

        for result in results:
            if result.get('success') and 'results' in result:
                engine = result['engine']
                for file_name, detection in result['results'].items():
                    if file_name in file_results:
                        file_results[file_name]['engines'][engine] = detection
                        file_results[file_name]['total_engines'] += 1
                        if detection == 1:
                            file_results[file_name]['detection_count'] += 1

        # 判断是否恶意
        for file_name in file_results:
            file_results[file_name]['malicious'] = file_results[file_name]['detection_count'] > 0

        # 成功统计
        successful_scans = sum(1 for r in results if r.get('success'))
        failed_scans = len(results) - successful_scans

        return {
            'scan_time': datetime.now().isoformat(),
            'elapsed_seconds': round(total_time, 3),
            'total_files': len(file_paths),
            'total_engines': len(engines),
            'successful_scans': successful_scans,
            'failed_scans': failed_scans,
            'file_results': file_results,
            'engine_details': results
        }

    def scan_single_file(
        self,
        file_path: str,
        engines: List[str] = None,
        max_workers: int = 15,
        timeout: int = 60
    ) -> Dict[str, Any]:
        """
        扫描单个文件

        Args:
            file_path: 待扫描文件路径
            engines: 要使用的引擎列表
            max_workers: 最大并发数
            timeout: 超时时间

        Returns:
            扫描结果
        """
        return self.scan_files([file_path], engines, max_workers, timeout)

    def scan_single_file_streaming(
        self,
        file_path: str,
        engines: List[str] = None,
        max_workers: int = 15,
        timeout: int = 60
    ):
        """
        流式扫描单个文件 - 每完成一个引擎就返回结果

        Args:
            file_path: 待扫描文件路径
            engines: 要使用的引擎列表
            max_workers: 最大并发数
            timeout: 超时时间

        Yields:
            每个引擎的扫描结果
        """
        if engines is None:
            engines = self.get_available_engines()

        if not file_path:
            yield {'error': '未提供待扫描文件'}
            return

        file_name = Path(file_path).name
        start_time = datetime.now()

        # 并行调用所有引擎,但使用生成器逐个返回结果
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_engine = {
                executor.submit(self._scan_single_engine, engine, [file_path], timeout): engine
                for engine in engines
            }

            # 每完成一个就返回一个
            for future in concurrent.futures.as_completed(future_to_engine):
                engine = future_to_engine[future]
                try:
                    result = future.result()

                    # 提取单个文件的结果
                    if result.get('success') and 'results' in result:
                        detection = result['results'].get(file_name, -1)

                        # 确定状态
                        if detection == 1:
                            status = 'malicious'
                        elif detection == 0:
                            status = 'safe'
                        else:
                            status = 'unsupported'

                        yield {
                            'engine': engine,
                            'status': status,
                            'detection': detection,
                            'success': True,
                            'vm_id': result.get('vm_id', 'unknown'),
                            'elapsed_seconds': result.get('elapsed_seconds', 0)
                        }
                    else:
                        yield {
                            'engine': engine,
                            'status': 'error',
                            'success': False,
                            'error': result.get('error', '未知错误'),
                            'vm_id': result.get('vm_id', 'unknown')
                        }

                except Exception as e:
                    yield {
                        'engine': engine,
                        'status': 'error',
                        'success': False,
                        'error': f"执行异常: {str(e)}"
                    }

        # 返回完成信号
        total_time = (datetime.now() - start_time).total_seconds()
        yield {
            'type': 'complete',
            'elapsed_seconds': round(total_time, 3),
            'total_engines': len(engines)
        }


def print_scan_results(result: Dict[str, Any]):
    """打印扫描结果"""
    print("\n" + "=" * 80)
    print("【分布式杀毒扫描结果】")
    print("=" * 80)

    print(f"\n扫描时间: {result['scan_time']}")
    print(f"总耗时: {result['elapsed_seconds']}s")
    print(f"文件数量: {result['total_files']}")
    print(f"引擎数量: {result['total_engines']}")
    print(f"成功扫描: {result['successful_scans']}")
    print(f"失败扫描: {result['failed_scans']}")

    print("\n" + "-" * 80)
    print("【文件检测结果】")
    print("-" * 80)

    for file_name, file_result in result['file_results'].items():
        print(f"\n📄 文件: {file_name}")
        print(f"   检测引擎数: {file_result['total_engines']}")
        print(f"   报告恶意: {file_result['detection_count']}")
        print(f"   判定结果: {'🔴 恶意' if file_result['malicious'] else '🟢 正常'}")

        print(f"\n   引擎检测结果:")
        for engine, detection in file_result['engines'].items():
            status = "🔴 恶意" if detection == 1 else "🟢 正常"
            print(f"     {engine}: {status}")

    print("\n" + "-" * 80)
    print("【引擎详细信息】")
    print("-" * 80)

    for engine_detail in result['engine_details']:
        engine = engine_detail['engine']
        if engine_detail.get('success'):
            print(f"\n✅ {engine} (VM: {engine_detail.get('vm_id', 'N/A')})")
            print(f"   耗时: {engine_detail.get('elapsed_seconds', 'N/A')}s")
            print(f"   扫描文件: {engine_detail.get('total', 0)}")
            print(f"   恶意文件: {engine_detail.get('malicious', 0)}")
        else:
            print(f"\n❌ {engine} (VM: {engine_detail.get('vm_id', 'N/A')})")
            print(f"   错误: {engine_detail.get('error', '未知错误')}")

    print("\n" + "=" * 80)


# ============================================================
# 主程序示例
# ============================================================
if __name__ == "__main__":
    # 创建分布式客户端
    client = AVDistributedClient()

    # 1. 检查虚拟机状态
    print("【检查虚拟机状态】")
    vm_status = client.get_vm_status()
    for vm_key, status in vm_status.items():
        status_str = "🟢 在线" if status else "🔴 离线"
        print(f"  {vm_key}: {status_str}")

    # 2. 获取可用引擎列表
    print(f"\n【可用引擎列表】")
    engines = client.get_available_engines()
    for i, engine in enumerate(engines, 1):
        vm_info = client.engine_to_vm[engine]
        print(f"  {i}. {engine} (VM: {vm_info['vm_id']})")

    # 3. 扫描单个文件示例
    print("\n【开始单文件扫描】")
    single_file = r"C:\Program Files\Git\usr\lib\sasl2\msys-anonymous-3.dll"
    result = client.scan_single_file(single_file)
    print_scan_results(result)

    # 4. 批量扫描示例
    print("\n\n【开始批量文件扫描】")
    batch_files = [
        r"C:\Program Files\Git\usr\lib\sasl2\msys-crammd5-3.dll",
        r"C:\Program Files\Git\usr\lib\sasl2\msys-gs2-3.dll"
    ]
    batch_result = client.scan_files(batch_files)
    print_scan_results(batch_result)

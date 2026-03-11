
import os
import sys
import requests
import configparser

# 读取配置文件
config = configparser.ConfigParser()
config_path = os.path.join(os.path.dirname(__file__), '../../../config.ini')
config.read(config_path, encoding='utf-8')

# Docker API 配置
DOCKER_API_BASE = config.get('docker_models', 'api_base', fallback='http://192.168.8.202')
DOCKER_TIMEOUT = config.getint('docker_models', 'timeout', fallback=30)

# Docker 模型配置：模型名称 -> (端口号, 显示名称)
DOCKER_MODELS = {
    'ember': (config.getint('docker_models', 'ember', fallback=8000), 'Ember'),
    'malconv': (config.getint('docker_models', 'malconv', fallback=8001), 'Malconv'),
    'imcfn': (config.getint('docker_models', 'imcfn', fallback=8002), 'Imcfn'),
    'malconv2': (config.getint('docker_models', 'malconv2', fallback=8003), 'Malconv2'),
    'transformer': (config.getint('docker_models', 'transformer', fallback=8004), 'Transformer'),
    'inceptionv3': (config.getint('docker_models', 'inceptionv3', fallback=8005), 'InceptionV3'),
    'rcnf': (config.getint('docker_models', 'rcnf', fallback=8006), 'Rcnf'),
    'onedense_cnn': (config.getint('docker_models', 'onedense_cnn', fallback=8007), '1D-CNN'),
    'malgraph': (config.getint('docker_models', 'malgraph', fallback=8008), 'Malgraph')
}


def call_docker_api(file_path: str, port: int) -> float:
    """
    调用Docker API进行预测

    Args:
        file_path: 文件路径
        port: Docker服务端口号

    Returns:
        恶意概率 (0-1之间的浮点数)
    """
    try:
        url = f"{DOCKER_API_BASE}:{port}/predict"
        with open(file_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(url, files=files, timeout=DOCKER_TIMEOUT)

        if response.status_code == 200:
            data = response.json()
            # 解析返回结果
            # {"filename": "xxx", "result": [{"label": "benign/malicious", "malware_probability": 0}]}
            if 'result' in data and len(data['result']) > 0:
                result = data['result'][0]
                malware_prob = result.get('malware_probability', 0)
                # 如果label是benign，概率应该是1-malware_probability
                label = result.get('label', 'benign')
                if label == 'benign':
                    return 1 - malware_prob
                else:
                    return malware_prob
        return None
    except Exception as e:
        print(f"Docker API调用失败 (端口{port}): {str(e)}")
        return None


# 模型列表：(显示名称, 端口号)
models = [
    (display_name, port) for model_key, (port, display_name) in DOCKER_MODELS.items()
]


def ensemble_prediction(file_path):
    results = {}
    malicious_probs = []
    safe_probs = []

    for model_name, port in models:
        try:
            # 调用Docker API
            score = call_docker_api(file_path, port)
            if score is not None:
                score = float(score)
                result = '恶意' if score > 0.5 else '安全'
                results[model_name] = {
                    "probability": round(score, 4),
                    "result": result
                }
                if result == '恶意':
                    malicious_probs.append(score)
                else:
                    safe_probs.append(score)
            else:
                results[model_name] = {
                    "probability": None,
                    "result": '预测失败'
                }
        except Exception as e:
            results[model_name] = {
                "probability": None,
                "result": f"错误: {str(e)[:50]}"
            }

    if malicious_probs and len(malicious_probs) > len(safe_probs):
        ensemble_prob = max(malicious_probs)
        ensemble_result = '恶意'
    elif safe_probs:
        ensemble_prob = min(safe_probs)
        ensemble_result = '安全'
    else:
        ensemble_prob = None
        ensemble_result = '无有效预测'

    results["集成结果"] = {
        "probability": ensemble_prob,
        "result": ensemble_result
    }
    return results


def run_ensemble_prediction(file_path: str) -> dict:
    """提供给外部调用的预测接口"""
    if not os.path.isfile(file_path):
        return {"error": "文件不存在"}
    return ensemble_prediction(file_path)


if __name__ == "__main__":
    file_path1 = "/home/user/MCDM/csdata/1000/03e64ec14874cfa91056833443e2426b635a57112c950642cde9a8a384821b5a"
    result = run_ensemble_prediction(file_path1)

    # 打印结果（保留与原逻辑一致的输出格式）
    for model_name, res in result.items():
        if model_name == "集成结果":
            print(f"\n{model_name}:")
            print(f"  概率: {res['probability']:.4f}" if res['probability'] is not None else "  概率: 无有效预测")
            print(f"  结果: {res['result']}")
        else:
            print(f"模型: {model_name}")
            print(f"  预测概率: {res['probability']:.4f}" if res['probability'] is not None else "  预测失败")
            print(f"  结果: {res['result']}")




    
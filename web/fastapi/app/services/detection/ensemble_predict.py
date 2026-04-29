import configparser
import os
from collections import OrderedDict
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests

# 读取配置文件
config = configparser.ConfigParser()
config_path = os.path.join(os.path.dirname(__file__), '../../../config.ini')
config.read(config_path, encoding='utf-8')

# Docker API 配置
DOCKER_API_BASE = config.get('docker_models', 'api_base', fallback='http://192.168.8.202')
DOCKER_TIMEOUT = config.getint('docker_models', 'timeout', fallback=30)
DOCKER_MODE = config.get('docker_models', 'mode', fallback='codefender').strip().lower()
CODEFENDER_API_BASE = config.get('codefender', 'api_base', fallback=config.get('codefender', 'baseUrl', fallback='http://127.0.0.1:8001')).rstrip('/')
CODEFENDER_SCAN_ENDPOINT = config.get('codefender', 'scan_endpoint', fallback='/scan/file')
CODEFENDER_PREDICT_ENDPOINT = config.get('codefender', 'predict_endpoint', fallback='/predict')
CODEFENDER_TIMEOUT = config.getint('codefender', 'timeout', fallback=DOCKER_TIMEOUT)
CODEFENDER_HOST_PATH_PREFIX = config.get('codefender', 'host_path_prefix', fallback='').strip()
CODEFENDER_CONTAINER_PATH_PREFIX = config.get('codefender', 'container_path_prefix', fallback='').strip()
MALICIOUS_THRESHOLD = config.getfloat('docker_models', 'threshold', fallback=config.getfloat('codefender', 'threshold', fallback=0.5))

# 旧版 Docker 模型配置：模型名称 -> (端口号, 显示名称)
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

# CoDefender v2 建议返回这些模型名；如果服务端返回名称，以服务端为准。
DEFAULT_CODEFENDER_MODEL_NAMES = [
    'ByteHistogram', 'ByteEntropyHistogram', 'StringExtractor', 'GeneralFileInfo',
    'HeaderFileInfo', 'SectionInfo', 'ImportsInfo', 'ExportsInfo', 'DataDirectories',
    'Ember-GBDT', 'Ember-LightGBM', 'Ember-XGBoost', 'MalConv', 'MalConv2',
    'AvastNet', 'Dike', 'Echelon', 'IMCFN', 'RCNF', 'InceptionV3', 'ResNet50',
    'DenseNet121', '1D-CNN', 'Transformer', 'MalGraph', 'Opcode-CNN', 'Opcode-LSTM',
    'PE-Header-MLP', 'Strings-MLP', 'Section-CNN', 'Import-Graph', 'Hybrid-MLP',
    'CoDefender-Ensemble'
]


def _normalize_endpoint(endpoint: str) -> str:
    if not endpoint.startswith('/'):
        return f'/{endpoint}'
    return endpoint


def _normalize_probability(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text.endswith('%'):
            try:
                return max(0.0, min(1.0, float(text[:-1]) / 100))
            except ValueError:
                return None
        try:
            value = float(text)
        except ValueError:
            return None
    try:
        probability = float(value)
    except (TypeError, ValueError):
        return None
    if probability > 1:
        probability = probability / 100
    return max(0.0, min(1.0, probability))


def _read_first(data: Dict[str, Any], keys: Iterable[str], default: Any = None) -> Any:
    for key in keys:
        if key in data and data[key] is not None:
            return data[key]
    return default


def _extract_probability(data: Dict[str, Any]) -> Optional[float]:
    probability = _read_first(data, [
        'probability', 'malware_probability', 'malicious_probability', 'malware_prob',
        'malicious_prob', 'score', 'Score', 'confidence', 'Confidence', 'prob'
    ])
    probability = _normalize_probability(probability)
    if probability is not None:
        return probability

    label = str(_read_first(data, ['label', 'Label', 'result', 'Result', 'verdict', 'Verdict'], '')).lower()
    benign_probability = _normalize_probability(_read_first(data, ['benign_probability', 'benign_prob']))
    if benign_probability is not None and label in ('benign', 'safe', 'clean', '正常', '安全'):
        return 1 - benign_probability
    return None


def _extract_result(data: Dict[str, Any], probability: Optional[float]) -> str:
    is_malware = _read_first(data, ['is_malware', 'isMalware', 'IsMalware', 'malicious', 'is_malicious'])
    if isinstance(is_malware, bool):
        return '恶意' if is_malware else '安全'

    label = str(_read_first(data, ['label', 'Label', 'result', 'Result', 'verdict', 'Verdict', 'Status'], '')).strip().lower()
    if label in ('malware', 'malicious', 'virus', 'detected', 'infected', 'bad', '恶意'):
        return '恶意'
    if label in ('benign', 'safe', 'clean', 'undetected', 'normal', 'good', '安全', '正常'):
        return '安全'

    if probability is None:
        return '预测失败'
    return '恶意' if probability > MALICIOUS_THRESHOLD else '安全'


def _extract_virus_name(data: Dict[str, Any]) -> str:
    value = _read_first(data, [
        'virus_name', 'VirusName', 'threat_name', 'ThreatName', 'malware_name',
        'MalwareName', 'family', 'Family', 'name', 'Name'
    ], '')
    return '' if value is None else str(value)


def _format_model_result(raw_result: Dict[str, Any], fallback_name: str) -> Dict[str, Any]:
    probability = _extract_probability(raw_result)
    result = _extract_result(raw_result, probability)
    virus_name = _extract_virus_name(raw_result)

    formatted = {
        'probability': round(probability, 4) if probability is not None else None,
        'result': result,
    }
    if virus_name:
        formatted['virus_name'] = virus_name
    raw_name = _read_first(raw_result, ['model', 'model_name', 'Model', 'ModelName', 'name'], fallback_name)
    if raw_name:
        formatted['model_name'] = str(raw_name)
    return formatted


def _model_name(raw_result: Dict[str, Any], fallback_name: str) -> str:
    name = _read_first(raw_result, ['model', 'model_name', 'Model', 'ModelName', 'name', 'engine_name'], fallback_name)
    return str(name or fallback_name)


def _iter_model_items(model_results: Any) -> Iterable[Tuple[str, Dict[str, Any]]]:
    if isinstance(model_results, dict):
        for name, value in model_results.items():
            if isinstance(value, dict):
                yield str(name), value
            else:
                yield str(name), {'probability': value}
    elif isinstance(model_results, list):
        for index, value in enumerate(model_results):
            fallback_name = DEFAULT_CODEFENDER_MODEL_NAMES[index] if index < len(DEFAULT_CODEFENDER_MODEL_NAMES) else f'Model-{index + 1}'
            if isinstance(value, dict):
                yield _model_name(value, fallback_name), value
            else:
                yield fallback_name, {'probability': value}


def _parse_codefender_response(data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    result_root = data.get('Result') or data.get('result') or data
    model_results = _read_first(result_root, [
        'ModelResults', 'model_results', 'models', 'Models', 'SubModels', 'sub_models',
        'model_scores', 'ModelScores', 'Scores', 'scores', 'details', 'Details'
    ])

    results: 'OrderedDict[str, Dict[str, Any]]' = OrderedDict()
    if model_results is not None:
        for fallback_name, raw_result in _iter_model_items(model_results):
            results[fallback_name] = _format_model_result(raw_result, fallback_name)

    if not results:
        final_raw = result_root if isinstance(result_root, dict) else data
        results['CoDefender集成模型'] = _format_model_result(final_raw, 'CoDefender集成模型')

    final_raw = result_root if isinstance(result_root, dict) else data
    final_probability = _extract_probability(final_raw)
    final_result = _extract_result(final_raw, final_probability)
    final_virus_name = _extract_virus_name(final_raw)
    if final_probability is None:
        valid_probs = [item['probability'] for item in results.values() if item.get('probability') is not None]
        if valid_probs:
            final_probability = sum(valid_probs) / len(valid_probs)
            final_result = '恶意' if final_probability > MALICIOUS_THRESHOLD else '安全'

    ensemble = {
        'probability': round(final_probability, 4) if final_probability is not None else None,
        'result': final_result,
    }
    if final_virus_name:
        ensemble['virus_name'] = final_virus_name
    results['集成结果'] = ensemble
    return results


def _translate_path_for_container(file_path: str) -> str:
    abs_path = os.path.abspath(file_path)
    if not CODEFENDER_HOST_PATH_PREFIX or not CODEFENDER_CONTAINER_PATH_PREFIX:
        return abs_path

    host_prefix = os.path.abspath(CODEFENDER_HOST_PATH_PREFIX)
    try:
        common = os.path.commonpath([host_prefix, abs_path])
    except ValueError:
        return abs_path
    if common != host_prefix:
        return abs_path

    relative_path = os.path.relpath(abs_path, host_prefix)
    return os.path.join(CODEFENDER_CONTAINER_PATH_PREFIX, relative_path).replace('\\', '/')


def call_codefender_api(file_path: str) -> Optional[Dict[str, Dict[str, Any]]]:
    url = f'{CODEFENDER_API_BASE}{_normalize_endpoint(CODEFENDER_SCAN_ENDPOINT)}'
    payload = {'file_path': _translate_path_for_container(file_path)}
    try:
        response = requests.post(url, json=payload, timeout=CODEFENDER_TIMEOUT)
        if response.status_code == 404:
            return call_codefender_predict_api(file_path)
        response.raise_for_status()
        return _parse_codefender_response(response.json())
    except Exception as path_error:
        upload_result = call_codefender_predict_api(file_path)
        if upload_result is not None:
            return upload_result
        print(f'CoDefender API调用失败: {str(path_error)}')
        return None


def call_codefender_predict_api(file_path: str) -> Optional[Dict[str, Dict[str, Any]]]:
    url = f'{CODEFENDER_API_BASE}{_normalize_endpoint(CODEFENDER_PREDICT_ENDPOINT)}'
    try:
        with open(file_path, 'rb') as f:
            response = requests.post(url, files={'file': f}, timeout=CODEFENDER_TIMEOUT)
        response.raise_for_status()
        return _parse_codefender_response(response.json())
    except Exception as error:
        print(f'CoDefender上传预测API调用失败: {str(error)}')
        return None


def call_legacy_docker_api(file_path: str, port: int) -> Optional[float]:
    try:
        url = f'{DOCKER_API_BASE}:{port}/predict'
        with open(file_path, 'rb') as f:
            response = requests.post(url, files={'file': f}, timeout=DOCKER_TIMEOUT)
        response.raise_for_status()
        data = response.json()
        if 'result' in data and len(data['result']) > 0:
            result = data['result'][0]
            return _extract_probability(result)
        return _extract_probability(data)
    except Exception as e:
        print(f'Docker API调用失败 (端口{port}): {str(e)}')
        return None


# 模型列表：(显示名称, 端口号)
models = [
    (display_name, port) for model_key, (port, display_name) in DOCKER_MODELS.items()
]


def legacy_ensemble_prediction(file_path: str) -> Dict[str, Dict[str, Any]]:
    results = OrderedDict()
    malicious_probs = []
    safe_probs = []

    for model_name, port in models:
        try:
            score = call_legacy_docker_api(file_path, port)
            if score is not None:
                result = '恶意' if score > MALICIOUS_THRESHOLD else '安全'
                results[model_name] = {
                    'probability': round(score, 4),
                    'result': result
                }
                if result == '恶意':
                    malicious_probs.append(score)
                else:
                    safe_probs.append(score)
            else:
                results[model_name] = {
                    'probability': None,
                    'result': '预测失败'
                }
        except Exception as e:
            results[model_name] = {
                'probability': None,
                'result': f'错误: {str(e)[:50]}'
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

    results['集成结果'] = {
        'probability': round(ensemble_prob, 4) if ensemble_prob is not None else None,
        'result': ensemble_result
    }
    return results


def ensemble_prediction(file_path: str) -> Dict[str, Dict[str, Any]]:
    if DOCKER_MODE in ('legacy', 'multi_port', 'multi-port', '9docker'):
        return legacy_ensemble_prediction(file_path)

    codefender_results = call_codefender_api(file_path)
    if codefender_results is not None:
        return codefender_results

    if DOCKER_MODE in ('auto', 'codefender_with_legacy_fallback'):
        return legacy_ensemble_prediction(file_path)

    return {
        'CoDefender集成模型': {
            'probability': None,
            'result': '预测失败'
        },
        '集成结果': {
            'probability': None,
            'result': '无有效预测'
        }
    }


def run_ensemble_prediction(file_path: str) -> dict:
    """提供给外部调用的预测接口。"""
    if not os.path.isfile(file_path):
        return {'error': '文件不存在'}
    return ensemble_prediction(file_path)


if __name__ == '__main__':
    file_path1 = '/home/user/MCDM/csdata/1000/03e64ec14874cfa91056833443e2426b635a57112c950642cde9a8a384821b5a'
    result = run_ensemble_prediction(file_path1)

    for model_name, res in result.items():
        print(f'模型: {model_name}')
        probability = res.get('probability')
        print(f"  预测概率: {probability:.4f}" if probability is not None else '  预测失败')
        print(f"  结果: {res.get('result')}")
        if res.get('virus_name'):
            print(f"  病毒名称: {res.get('virus_name')}")

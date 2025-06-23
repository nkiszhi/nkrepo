
import os
import sys
from models.m_2017_malconv.exec_malconv import run_prediction as malconv_run_prediction
from models.m_2017_transformer.exec_transformer import run_prediction as transformer_run_prediction
from models.m_2018_ember.exec_ember import run_prediction as ember_run_prediction
from models.m_2019_1d_cnn.exec_1d_cnn import run_prediction as cnn_1d_run_prediction
from models.m_2020_inceptionv3.exec_InceptionV3 import run_prediction as InceptionV3_run_prediction
from models.m_2021_malconv2.exec_malconv2 import run_prediction as malconv2_run_prediction
from models.m_2021_rcnf.exec_rcnf import run_prediction as rcnf_run_prediction
from models.m_attention_rcnn.attention_rcnn import run_prediction as attention_rcnn_run_prediction
from models.m_rcnn.rcnn import run_prediction as rcnn_run_prediction
from models.m_vgg16.vgg16 import run_prediction as vgg16_run_prediction

models = [
    ('Malconv', malconv_run_prediction),
    ('Transformer', transformer_run_prediction),
    ('Ember', ember_run_prediction),
    ('1D-CNN', cnn_1d_run_prediction),
    ('InceptionV3', InceptionV3_run_prediction),
    ('Malconv2', malconv2_run_prediction),
    ('Rcnf', rcnf_run_prediction),
    ('Attention_rcnn', attention_rcnn_run_prediction),
    ('Rcnn', rcnn_run_prediction),
    ('Vgg16', vgg16_run_prediction)
]


def ensemble_prediction(file_path):
    results = {}
    malicious_probs = []
    safe_probs = []

    for model_name, run_prediction_func in models:
        try:
            score = run_prediction_func(file_path)
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




    
import os
import requests
import time
from io import BytesIO
from requests_toolbelt.multipart import encoder
import json

class VTAPI:
    def __init__(self, api_key):
        self.api_key = api_key  

    def post_url(self, sample_file_path):  
        if not os.path.exists(sample_file_path):
            print(f"警告：上传文件不存在 {sample_file_path}，跳过上传")
            return None

        try:
            with open(sample_file_path, 'rb') as file:  
                file_data = file.read()  
        except Exception as e:
            print(f"读取上传文件失败: {str(e)}")
            return None

        form_fields = [('file', ('file', BytesIO(file_data), 'application/octet-stream'))]  
        multipart_data = encoder.MultipartEncoder(fields=form_fields)
        multipart_header = {  
            'Content-Type': multipart_data.content_type,  
            'x-apikey': self.api_key,  # 使用实例变量
            'Accept': 'application/json'  
        }  

        upload_url = 'https://www.virustotal.com/api/v3/files'  
        try:
            response = requests.post(
                upload_url, 
                data=multipart_data, 
                headers=multipart_header, 
                timeout=30
            )  
            response.raise_for_status()
            return response.json()['data']['id']
        except requests.exceptions.RequestException as e:
            print(f"VT上传失败: {str(e)}")
            return None

    def get_API_result_detection(self, sha256, sample_dir_path):  
        result_file_path = os.path.join(sample_dir_path, f'{sha256}.json')
        if os.path.exists(result_file_path):
            return result_file_path 

        url_detection = f'https://www.virustotal.com/api/v3/files/{sha256}'
        headers = {'x-apikey': self.api_key, 'Accept': 'application/json'} 
        max_attempts = 20
        attempts = 0

        while attempts < max_attempts:  
            try:  
                response = requests.get(url_detection, headers=headers, timeout=10)  
                
                if response.status_code == 200:  
                    report = response.json()
                    if 'data' in report and 'attributes' in report['data']:  
                        last_analysis = report['data']['attributes'].get('last_analysis_results', {})  
                        
                        if last_analysis:  
                            os.makedirs(sample_dir_path, exist_ok=True)
                            with open(result_file_path, 'w', encoding='utf-8') as f:  
                                json.dump(report, f, indent=4, ensure_ascii=False)  
                            return result_file_path  
                        else:  
                            print(f"等待VT扫描结果（{attempts}/{max_attempts}）...")  
                            time.sleep(8)
                            attempts += 1
                    else:  
                        print("VT响应结构无效")  
                        return None
                        
                elif response.status_code == 404:  
                    print(f"VT无该文件记录: {sha256}")  
                    return None
                    
                else:  
                    print(f"VT查询失败，状态码: {response.status_code}")  
                    return None
                    
            except Exception as e:  
                print(f"VT检测查询错误: {str(e)}")  
                attempts += 1
                time.sleep(8)
        
        print(f"超过最大尝试次数，VT扫描未完成")
        return None

    def get_API_result_behaviour(self, sha256, sample_dir_path):  
        result_file_path = os.path.join(sample_dir_path, f'{sha256}_behaviour_summary.json')
        if os.path.exists(result_file_path):
            return result_file_path

        url = f'https://www.virustotal.com/api/v3/files/{sha256}/behaviour_summary' 
        headers = {'x-apikey': self.api_key}  # 使用实例变量
        try:  
            response = requests.get(url, headers=headers, timeout=10)  
            
            if response.status_code == 200:  
                os.makedirs(sample_dir_path, exist_ok=True)
                with open(result_file_path, 'w', encoding='utf-8') as f:  
                    json.dump(response.json(), f, indent=4, ensure_ascii=False)  
                return result_file_path  
                
            elif response.status_code == 404:  
                print(f"VT无行为报告记录: {sha256}")  
                return None
                
            else:  
                print(f"行为报告请求失败: {response.status_code}")  
                return None
                
        except Exception as e:  
            print(f"行为报告查询错误: {str(e)}")  
            return None

    def get_summary(self, sha256):  
        try:
            url = f'https://www.virustotal.com/api/v3/files/{sha256}'
            headers = {'x-apikey': self.api_key}  # 使用实例变量
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                return {
                    'last_analysis_stats': data['data']['attributes'].get('last_analysis_stats', {}),
                    'reputation': data['data']['attributes'].get('reputation', 0)
                }
            return None
        except Exception as e:
            print(f"获取VT摘要失败: {str(e)}")
            return None
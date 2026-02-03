import hashlib  
import os
import requests
import time
from io import BytesIO
from requests_toolbelt.multipart import encoder
import json



class VTAPI:

    def post_url(self,sample_file_path,api_key):
    
        with open(sample_file_path, 'rb') as file:  
          file_data = file.read()  
        # 创建一个多部分表单编码器  
        boundary = '------WebKitFormBoundary7MA4YWxkTrZu0gW'  
        form_fields = [  
            ('file', ('file', BytesIO(file_data), 'application/octet-stream'))  
        ]  
        multipart_data = encoder.MultipartEncoder(fields=form_fields, boundary=boundary)  
        multipart_header = {  
            'Content-Type': multipart_data.content_type,  
            'x-apikey': api_key,  
            'Accept': 'application/json'  
        }  
   
        upload_url = 'https://www.virustotal.com/api/v3/files'  
        response = requests.post(upload_url, data=multipart_data.to_string(), headers=multipart_header)  
        response.raise_for_status()  
   
        upload_response = response.json()  
        scan_id = upload_response['data']['id']  
        print(f"Scan ID: {scan_id}")
        return scan_id
        
    

    def get_API_result_detection(self, sha256, api_key, sample_dir_path):
        vt = VTAPI()
        result_file_path = os.path.join(sample_dir_path, f'{sha256}.json')
        sample_file_path = os.path.join(sample_dir_path, f'{sha256}')
    
        if os.path.exists(result_file_path):
            return result_file_path 
        else:
            scan_id = vt.post_url(sample_file_path, api_key)
            url_detection = f'https://www.virustotal.com/api/v3/files/{sha256}'
            headers = {'x-apikey': api_key, 'Accept': 'application/json'}  
        
            while True:  
                try:  
                    response = requests.get(url_detection, headers=headers)  
                    if response.status_code == 200:  
                        report = response.json()
                    
                        if 'data' in report and 'attributes' in report['data']:  
                            last_analysis_results = report['data']['attributes']['last_analysis_results']  
                        
                            if last_analysis_results not in (None, {}):
                                # 格式化保存 JSON
                                with open(result_file_path, 'w', encoding='utf-8') as result_file:  
                                    json.dump(report, result_file, 
                                            indent=4, 
                                            ensure_ascii=False,
                                            sort_keys=False)    
                                print(f'已格式化保存 {sha256}.json 到 {result_file_path}')  
                                return result_file_path  
                            else:  
                                print("等待扫描完成...")  
                                time.sleep(10)
                        else:  
                            print("无效的 API 响应结构")  
                            return None
                    else:  
                        print(f"获取结果失败，状态码：{response.status_code}")  
                        return None  # 这里应该退出循环
                except requests.exceptions.RequestException as e:  
                    print(f'检测发生错误 {sha256}: {e}')  
                    return {'error': str(e)}


            
    def get_API_result_behaviour(self, sha256, api_key, sample_dir_path):
        vt = VTAPI()
        result_file_path = os.path.join(sample_dir_path, f'{sha256}_behaviour_summary.json')
        sample_file_path = os.path.join(sample_dir_path, f'{sha256}') 
        
        if os.path.exists(result_file_path):
            return result_file_path
        else:
            scan_id = vt.post_url(sample_file_path,api_key)
            url_behavuours = f'https://www.virustotal.com/api/v3/files/{sha256}/behaviour_summary' 
            headers = {'x-apikey': api_key}  
            
            try:  
                response = requests.get(url_behavuours, headers=headers)  
                if response.status_code == 200:  
                    # 格式化保存行为摘要
                    behaviour_data = response.json()
                    with open(result_file_path, 'w', encoding='utf-8') as result_file:
                        json.dump(behaviour_data, result_file, 
                                  indent=4, 
                                  ensure_ascii=False,
                                  sort_keys=False)
                    
                    print(f'已格式化保存 {sha256}_behaviour_summary.json 到 {result_file_path}')  
                    return result_file_path  
                else:  
                    print(f'请求失败，状态码：{response.status_code}')  
                    return None
            except requests.exceptions.RequestException as e:  
                print(f'检测发生错误 {sha256}: {e}')  
                return {'error': str(e)}

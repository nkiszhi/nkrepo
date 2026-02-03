# -*- coding: utf-8 -*-
"""
VirusTotal API Module

Provides integration with VirusTotal API for malware analysis including:
- File upload and scanning
- Detection results retrieval
- Behaviour analysis retrieval
"""

import os
import json
import time
from io import BytesIO

import requests
from requests_toolbelt.multipart import encoder


class VirusTotalAPI:
    """VirusTotal API client for malware analysis."""

    BASE_URL = "https://www.virustotal.com/api/v3"

    def _upload_file(self, sample_file_path, api_key):
        """
        Upload a file to VirusTotal for scanning.

        Args:
            sample_file_path: Path to the file to upload
            api_key: VirusTotal API key

        Returns:
            Scan ID for the uploaded file
        """
        with open(sample_file_path, 'rb') as f:
            file_data = f.read()

        boundary = '------WebKitFormBoundary7MA4YWxkTrZu0gW'
        form_fields = [
            ('file', ('file', BytesIO(file_data), 'application/octet-stream'))
        ]
        multipart_data = encoder.MultipartEncoder(fields=form_fields, boundary=boundary)
        headers = {
            'Content-Type': multipart_data.content_type,
            'x-apikey': api_key,
            'Accept': 'application/json'
        }

        upload_url = f"{self.BASE_URL}/files"
        response = requests.post(upload_url, data=multipart_data.to_string(), headers=headers)
        response.raise_for_status()

        upload_response = response.json()
        scan_id = upload_response['data']['id']
        print(f"Scan ID: {scan_id}")
        return scan_id

    def get_detection_result(self, sha256, api_key, sample_dir_path):
        """
        Get detection results for a file from VirusTotal.

        Args:
            sha256: SHA256 hash of the file
            api_key: VirusTotal API key
            sample_dir_path: Directory path where the sample and results are stored

        Returns:
            Path to the JSON result file, or None on error
        """
        result_file_path = os.path.join(sample_dir_path, f'{sha256}.json')
        sample_file_path = os.path.join(sample_dir_path, sha256)

        # Return cached result if exists
        if os.path.exists(result_file_path):
            return result_file_path

        # Upload file and get scan results
        self._upload_file(sample_file_path, api_key)
        url = f"{self.BASE_URL}/files/{sha256}"
        headers = {'x-apikey': api_key, 'Accept': 'application/json'}

        while True:
            try:
                response = requests.get(url, headers=headers)
                if response.status_code == 200:
                    report = response.json()

                    if 'data' in report and 'attributes' in report['data']:
                        last_analysis_results = report['data']['attributes'].get('last_analysis_results')

                        if last_analysis_results:
                            # Save formatted JSON result
                            with open(result_file_path, 'w', encoding='utf-8') as f:
                                json.dump(report, f, indent=4, ensure_ascii=False)
                            print(f"Saved {sha256}.json to {result_file_path}")
                            return result_file_path
                        else:
                            print("Waiting for scan to complete...")
                            time.sleep(10)
                    else:
                        print("Invalid API response structure")
                        return None
                else:
                    print(f"Failed to get result, status code: {response.status_code}")
                    return None

            except requests.exceptions.RequestException as e:
                print(f"Detection error for {sha256}: {e}")
                return {'error': str(e)}

    def get_behaviour_result(self, sha256, api_key, sample_dir_path):
        """
        Get behaviour analysis results for a file from VirusTotal.

        Args:
            sha256: SHA256 hash of the file
            api_key: VirusTotal API key
            sample_dir_path: Directory path where the sample and results are stored

        Returns:
            Path to the JSON result file, or None on error
        """
        result_file_path = os.path.join(sample_dir_path, f'{sha256}_behaviour_summary.json')
        sample_file_path = os.path.join(sample_dir_path, sha256)

        # Return cached result if exists
        if os.path.exists(result_file_path):
            return result_file_path

        # Upload file and get behaviour summary
        self._upload_file(sample_file_path, api_key)
        url = f"{self.BASE_URL}/files/{sha256}/behaviour_summary"
        headers = {'x-apikey': api_key}

        try:
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                behaviour_data = response.json()
                with open(result_file_path, 'w', encoding='utf-8') as f:
                    json.dump(behaviour_data, f, indent=4, ensure_ascii=False)
                print(f"Saved {sha256}_behaviour_summary.json to {result_file_path}")
                return result_file_path
            else:
                print(f"Request failed, status code: {response.status_code}")
                return None

        except requests.exceptions.RequestException as e:
            print(f"Detection error for {sha256}: {e}")
            return {'error': str(e)}

    # Backward compatibility aliases
    def get_API_result_detection(self, sha256, api_key, sample_dir_path):
        """Deprecated: Use get_detection_result instead."""
        return self.get_detection_result(sha256, api_key, sample_dir_path)

    def get_API_result_behaviour(self, sha256, api_key, sample_dir_path):
        """Deprecated: Use get_behaviour_result instead."""
        return self.get_behaviour_result(sha256, api_key, sample_dir_path)


# Backward compatibility alias
VTAPI = VirusTotalAPI

# nkrepo: A malware sample repository

Malware samples are stored in DATA folder. Based on the SHA256 of each
malware sample, samples are stored in the subfolder
"DATA/SHA256[0]/SHA256[1]/SHA256[2]/SHA256[3]/" with 4 levels architecture.

1. count_samples.py: count all malware samples in the repo
2. count_labels.py: count the number of json files which contain VirusTotal scan results.
3. search.py: the input is a SHA256 string, and the output is the specific file information including file size, modify time and so on. 
4. stat.py: statistics all samples file size, file type, modify time, and save the statistics info in the csv files for each 4-level folder.
5. check.py: check if all samples have VirusTotal scan results. If a
   sample without VirusTotal scan result, the sample SHA256 will be listed in the
   sha256.txt file.
6. update.py: 

## Search 

First, input sample SHA256 value into search.py, and output following information:
1. Basic information including file name, file size, file type, MD5, SHA-1, SHA256, access time, modify time, changetime, file compress or packer information;
2. If sample is a PE file, output PE sections information, malware commonly used APIs, anamoly information.


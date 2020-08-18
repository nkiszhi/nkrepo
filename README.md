# nkrepo: A malware sample repository

Malware samples are stored in DATA folder. Based on the SHA256 of each
malware sample, samples are stored in the subfolder
"DATA/SHA256[0]/SHA256[1]/SHA256[2]/SHA256[3]/" with 4 levels architecture.

1. count_samples.py: count all malware samples in the repo
2. count_labels.py: count the number of json files which contain VirusTotal scan results.
3. get_sha256.py: check if all samples have VirusTotal scan results. If a
   sample without VirusTotal scan result, the sample SHA256 will be listed in the
   sha256.txt file.

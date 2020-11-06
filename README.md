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
3. If sample has a vt json file, output Anti-Virus engines detection results.


## stat folder

The stat folder contains the statistical result of repo containing file size, file modify time, file type.
i

## Python文件及其作用
get_json_list将数据集中所有后缀为json的文件名写入json.txt的文本文档中

get_pack_samples先扫描各病毒文件夹下是否有f_pack_info.csv文件，如果没有，进行病毒文件的扫描，标记规则如下：是userdb.txt文件，则写入信息，若只是pe32则写文件名加none最后放入f_pack_info.csv文件中
get_all_pack_samples是用来查找数据集中的f_pack_info.csv文件，并将它信息提取出来，放入一个新的excel表格中，表格名称为pack_info.csv
get_pack将pack_info.csv重新排序后放入pack.csv，中间用none判断来，所以现在pack.csv里面只有userdb.txt

get_samples_info_about_userdb.py,是将所有病毒文件中userdb.txt文件信息收集，放入pack.csv文件中
get_samples_info_about_allpe32.py,是将所有病毒文件中所有pe文件信息收集，放入f_info_del.csv
get_info_count将f_info_del.csv中文件大小的信息统计分组，单位为kb,并将结果写入all_KB_size.csv
get_filetype将f_info_del.csv中的内容经过判断后放入f_pack_del.csv,f_pack_del.csv是pe32文件中除了userdb.txt之外的文件

get_filetype_filesize将数据集中所有f_pack_info.csv文件中的信息都放入f_info_del.csv
get_info_count将f_info_del.csv中文件大小信息统计分组，单位为kb,并将结果写入all_KB_size.csv
get_filetype将f_info_del.csv中的内容经过判断后放入f_pack_del.csv,f_pack_del.csv是pe32文件中除了userdb.txt之外的文件

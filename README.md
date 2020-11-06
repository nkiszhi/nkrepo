# nkrepo: A malware sample repository

Malware samples are stored in DATA folder. Based on the SHA256 of each
malware sample, samples are stored in the subfolder
"DATA/SHA256[0]/SHA256[1]/SHA256[2]/SHA256[3]/" with 4 levels architecture.

## nkrepo中Python文件作用
check.py:	统计DATA文件夹下所有sha256文件名称，并写入到sha256.txt中，在屏幕上输出文件总数
count_f_info.py:	统计DATA文件夹中f_pack_info.csv的个数并输出到屏幕
count_labels.py：	统计DATA文件夹中json文件个数
count_samples.py:	统计DATA文件夹中病毒样本个数

del_sample.py:	输入一个sha256.txt文件，删除文件中的样本，并将已删除/不存在的文件个数输出到屏幕
delvirus.py:    将本地服务器的sample文件和主服务器上的文件进行对比，删除本地重复的文件

get_json_list:	将数据集中所有后缀为json的文件名写入json.txt的文本文档中

get_pack_samples:	先扫描各病毒文件夹下是否有f_pack_info.csv文件，如果没有，进行病毒文件的扫描，标记规则如下：是userdb.txt文件，则写入信息，若只是pe32则写文件名加none最后放入f_pack_info.csv文件中
get_all_pack_samples:	是用来查找数据集中的f_pack_info.csv文件，并将它信息提取出来，放入一个新的excel表格中，表格名称为pack_info.csv
get_pack:	将pack_info.csv重新排序后放入pack.csv，中间用none判断来，所以现在pack.csv里面只有userdb.txt

get_samples_info_about_userdb.py:	是将所有病毒文件中userdb.txt文件信息收集，放入pack.csv文件中
get_samples_info_about_allpe32.py:	是将所有病毒文件中所有pe文件信息收集，放入f_info_del.csv
get_info_count:	将f_info_del.csv中文件大小的信息统计分组，单位为kb,并将结果写入all_KB_size.csv
get_filetype:	将f_info_del.csv中的内容经过判断后放入f_pack_del.csv,f_pack_del.csv是pe32文件中除了userdb.txt之外的文件

get_filetype_filesize:	将数据集中所有f_pack_info.csv文件中的信息都放入f_info_del.csv
get_info_count:	将f_info_del.csv中文件大小信息统计分组，单位为kb,并将结果写入all_KB_size.csv
get_filetype:	将f_info_del.csv中的内容经过判断后放入f_pack_del.csv,f_pack_del.csv是pe32文件中除了userdb.txt之外的文件

init_repo.py:   检测并补充DATA文件夹下的目录

search-ch.py:	传入一个sha256，判断是否为pe文件，若不是则退出；若是则输出信息：
		1. 文件路径
		2. MD5
		3. SHA-1
		4. SHA-256
		5. 文件访问时间
		6. 文件内容修改时间
		7. 文件属性修改时间
		8. 文件大小
		9. 导入表
		10.导出表
		11.文件类型
		12.文件头信息
		13.恶意api检测
		14.编译器和加壳信息检测
		15.异常信息检测

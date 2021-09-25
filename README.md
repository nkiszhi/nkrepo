# nkrepo: 恶意代码样本库管理项目
#### 南开大学 反病毒实验室（NKAMG，Nankai Anti-Malware Group）



#### ![nkrepo](images/nkrepo.png)



## 1. 恶意代码样本库内容
- 恶意代码样本库，样本总数量千万级别；
- 恶意代码文件类型包括PE（Windows操作系统）、ELF（Linux操作系统）、APK（Android系统）、可执行脚本（Web平台）、JAVA文件（嵌入式、物联网平台）等； 
- 样本的类型标签、家族标签；
- 杀毒软件检测结果；
- 静态特征，包括字节序列、API函数、字符串等；
- 动态特征，沙箱中记录的API调用序列。

## 2. 恶意代码样本库查询
- Web形式和命令行形式的基于哈希值的快速查询；
- 基于样本创建时间的查询；
- 基于恶意代码家族信息的查询；
- 基于恶意代码类型信息的查询；
- 基于恶意代码特征信息的查询；
- 基于杀毒软件检测结果的查询；
- 多种查询条件的综合样本查询；
- Yara规则检索；

## 3. 恶意代码样本库下载
- 支持多个恶意代码样本的同时下载，提供可编程的API下载接口，支持基于API接口的软件二次开发； 
- 支持多样本的自动压缩和加密下载，支持GB级的海量样本打包和压缩；
- 支持torrent文件分享和P2P模式下载。 

## 4. 恶意代码样本库统计分析
- 基于Web界面展示样本库中样本的统计信息；
- 展示恶意代码库中样本的时间分布；                                  
- 展示样本大小分布、样本的文件类型分布；                                  
- 展示样本的家族信息、类型信息的统计结果；                                  
- 展示样本多种特征信息的统计结果。  
                                
## 5. 恶意代码样本库自动更新
- 恶意代码样本的添加接口，对恶意代码样本库进行样本文件的扩充； 
- 恶意代码标签自动更新； 
- 恶意代码样本特征点的自动提取； 
- 特征点的扩充，可以添加新的特征点提取函数、更新样本特征文件； 
- 机器学习算法的接入，支持对样本标注机器学习模型检测结果； 
- 自动更新统计信息；
       

# Management of Malware Samples

## Statistics
count_samples.py: Counting the number of all malware samples. 
count_kav_label.py: Counting the Kaspersky labels in the repo. 
count_vt_label.py: Counting the VirusTotal labels in the repo. 

## Initialization
init_repo.py:  Create a 4-tier storage structure. The first layer has 16
folders, which are named 0 to 9 and a to f. In each folder at the first layer,
there are 16 subfolders, which are also named 0 to 9 and a to f. The subfolders
are at the second layer. In total there are 256 (16 x 16) folders at the second
layer. In the same way, there are 4096 (16 x 16 x 16) subfolders at the third
layer and 65536 (16 x 16 x 16 x 16) subfolders at the fourth layer. For each
malware sample, we firstly calculate the sample SHA256 value. Secondly, based on the
first 4 characters of this SHA256 value, we store the sample to the specific
subfolder at the fourth layer. For example, there is a malware sample and its
SHA 256 value is 1234...64. The first 4 characters of the SHA256 value is 1234,
so this malware sample will be stored in the subfolder 1/2/3/4. 

## Add and delete malware samples
del_sample.py : delete malware samples from repo.
add_sample.py : add new malware samples into repo.


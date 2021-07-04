# nkrepo: 恶意代码样本库管理项目
#### 南开大学 反病毒实验室（NKAMG，Nankai Anti-Malware Group）

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
        


# Get scan results from VirusTotal
vt\_scan.py将样本的sha256哈希值提交给VirusTotal，得到VirusTotal的样本扫描结果。
该程序需要提前注册VirusTotal的API Key，并写入config.ini文件中。
因为VirusTotal对Key有查询次数限制，该程序支持同时使用多个API Key。
VirusTotal对查询的频率也做了限制，每次查询都会间隔20秒的时间。
vt\_scan.py的输入参数有两个，一个是存储需要扫描样本的文件夹，一个是保存扫描结果的文件夹。
每个样本的VirusTotal扫描结果都存储在一个json格式的文件中。
所有样本的扫描结果都存储在指定的文件夹中。
每个扫描结果的json文件都以样本的sha256命名，后缀为json。


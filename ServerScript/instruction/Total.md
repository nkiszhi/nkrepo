# 自动维护脚本具体说明
## 总览图
![](Total.png)  
其中instructionsend.py对应着本地信息发送脚本，scriptmain.py对应着服务器指令接收脚本（也附加有启动整个自动维护脚本程序的功能），nkaz、nkvt、nkvs文件夹内的脚本程序对应着图中nkaz维护脚本，nkvs维护脚本等。
## 样本库存放路径及要求
nkaz : /VirusDatabase/NKAZDATA/  
nkvt : 对应的标签数据和病毒放在同一目录下。
nkvs : /VirusDatabase/NKVSDATA/  
1. 存放方式依然为按照sha256的四级索引。
2. 无文件后缀名的为病毒文件。
3. .json为后缀除file.json的文件为标签文件。
4. 每个底路径下有一个file.json文件，用于统计相应底路径下的所有**病毒**信息（不需要统计标签文件的信息），例如在 /VirusDatabase/NKAZDATA/0/0/0/0/ 路径下的file.json应该拥有 /VirusDatabase/NKAZDATA/0/0/0/0/ 目录下所有病毒的统计信息，注意 /VirusDatabase/NKAZDATA/0/0/0/  下是没有file.json文件的。

目前file.json内容初步定为：
```json
{
    // 该目录下总样本数量
    "TotalFileNumber" : 10000,

    // 该目录下文件时间分布
    "FileTimeDistribution" : {
        "2012" : 3000,
        "2013" : 4000,
        "2014" : 500,
        "2015" : 500,
        "2016" : 500,
        "2017" : 500,
        "2018" : 500,
        "2019" : 500,
        // 假设没有2020年的文件可以直接把这一个键值对抹去，其余同理
        "2020" : 500
    },

    // 该目录下文件大小分布，与上同理
    "FileSizeDistribution" : {
        "<10KB" : 200,
        "10KB-100KB" : 400,
        "100KB-1MB" : 400,
        "1MB-10MB" : 40,
        // 假设没有>10MB的文件可以直接把这一个键值对抹去，其余同理
        ">10MB" : 50
    },

    // 该目录下文件类型信息，与上同理
    "FileTypeDistribution" : {
        "PEFile" : 500,
        "ELF" : 50,
        // 假设没有HTML文件可以直接把这一个键值对抹去，其余同理
        "HTML" : 500,
        "Compressed File" : 400,
        "other" : 500
    },
    "FamilyDistribution" : {
        "abc.skfjskf" : 1000,
        "bcd.sdfhjsdjkfk" : 2000,
        "bcd.sdkfjsk" : 2000
    }
}
```
## 具体维护脚本的基本功能
1. 自动更新。在某个时间点唤醒脚本进行数据库更新，例如，在每周日程序将会被唤醒，到Androzoo网站上下载一下最新的数据。因此，每个具体维护脚本应该有一个自己的独立的主进程，此主进程用于实现脚本的自动维护功能。
2. 可以被人为操控。自行设计指令，并在指令管理器里注册指令，完成对具体维护脚本主进程的操控，指令由本地的指令发送脚本instructionsend.py来发送给服务器，服务器将进行解析后运行。
3. 可以随时查看脚本的运行状态。
4. 总结。每个具体维护脚本应该有两大块：主进程和主进程控制代码。主进程用于实现自动维护的功能，控制代码用于控制主进程并观察主进程的执行情况。
## 具体维护脚本必须提供的接口
```python
 # 启动主进程
 def start(self) : pass

 # 终止主进程
 def stop(self) : pass

 # 获得状态信息
 def getState(self) -> str : 
     return "all right"

 # 检查主进程是否正在运行
 def isRunning(self) -> bool : 
     return True

 # 检查主进程的运行是否正常
 def isNormal(self) -> bool : 
     return True

 # 获得样本的统计信息，以字典的方式返回所有file.json的内容汇总
 def getStatistics(self) -> dict : 
     # 对于nkvt这里可以直接返回空字典
     return {"TotalFileNumber" : 1000}

 # 获得样本的总体数量
 def getCount(self) -> int : 
     # 对于nkvt这里可以直接返回0
     return 1000
```
具体可以看 ./main/Models.py 中的 ScriptModel 类，此文件中也有一些你可能用得到的函数，欢迎查看并使用。
## 指令管理器使用说明
在每个具体维护脚本的类的__init__函数中将会接收到一个参数instmanager，此参数是一个指令管理器，使用此类中的如下成员函数进行指令注册。注册完成的指令将会在被指令之心时触发相应函数的运行。
```python
instmanager.addInstruction(
    inst : str, 
    targetfunc : callable([type(Client), tuple, any]), 
    targetscript : str, 
    description = "", 
    level = 0
)
```
### 参数解释
+ inst表示指令内容，指令内容不可以为nkaz nkvt nkvs main，指令内容也不能重复。
+ targetfunc表示被绑定的函数，函数要求见下。
+ targetscript表示目标脚本，只可以为nkaz nkvt nkvs 或 main。
+ description表示指令描述，在系统自动生成help的时候会被显示出来
+ level表示指令等级，目前有两个等级
  + 0表示普通等级，在执行前不会对用户进行身份验证。
  + 1表示特权等级，在执行前会对用户进行身份验证。
### 被绑定指令的函数的要求
1. 第一个参数必须为Client类，用于接收一些和本地指令传输脚本的互动函数。
2. 第二个参数必须为tuple类，用于接收用户传来的参数。
3. 函数有且只有以上的两个参数
```python
func(client : type(Client), args : tuple)
```
## Client类使用说明
目前Client有如下功能（以下均为Client类的成员函数）
```python
 # 向用户发送信息并显示在用户的终端上
client.sendMessageShow(self, message : str)
```
```python
# 身份验证，times表示可以最多输入多少次错误密码，验证通过返回True，失败返回False
client.authentication(self, times = 5) -> bool
```
```python
# 行为确认，message为提示信息，confirmway为用户需要输入的字符（可以为多个），确认成功返回True，否则为False
client.instructionConfirm(self, message : str, *confirmway : str) -> bool
```
## 具体维护脚本启动说明
在scriptmain.py被执行时，将会创建Androzoo，VirusShare，VirusTotal类的具体实例，所以具体维护脚本的**初始化程序请放在__init__函数中**。  
例如，你应该在__init__中创建自己的具体维护脚本的主进程，进程的名称为自己维护的部分，如nkaz的脚本的进程名称应该为nkaz，nkvt的脚本的进程名称应该为nkvt，同时应该设置好各种参数和变量。同样，如果start函数具有启动此进程的功能，可以直接在__init__中调用start函数。
## 错误与异常管理器使用说明
异常处理器用于存储相关异常，并对异常信息进行解析和管理，指令如下：
```
error num   查看新出现了多少个错误
error rm    删除所有的异常信息
error all   显示所有的异常信息，包括已经被查看过的历史信息。
error size  显示异常信息存储文件的大小
error help  打开error指令的参数帮助
error       查看最新出现的异常信息
```
**异常处理器可能无法获得除主控进程之外的其他进程的异常信息，如具体维护脚本的主进程异常信息就可能无法获得。**
## debug模式
debug模式只是将提供服务和接受服务的服务器端口更改为4001，同时仅开启一个具体维护脚本的主进程。可以在scriptmain.py中设置，方法如下。
```python
script = ScriptManager(debug = 'nkaz') # 多传入一个参数debug，内容为需要启动的维护脚本
```
同时需要将insturctionsend.py中的debug变量设置成True。
## file.json信息更新协议
nkaz和nkvs等样本下载脚本只负责进行病毒样本的下载，不会进行任何的file.json操作，但是需要将新下载的文件的sha256写入自己维护的脚本库顶层目录下的dirtyfile文件夹中。
nkvt和后续打标签脚本将负责file.json的信息统计，Models内有如下函数可供调用。
```python
# 更新所有file.json，可以在下载完所有文件后调用此函数，rootpath表示需要更新哪个目录下的库，请注意这里是顶层目录，dirtylist表示需要更新的脏文件
updateTree(dirtylist, rootpath)
```
## web信息查询传输协议
前端使用GET请求向后端发送请求，格式如下
```
/search?Sha256Search=abcd
/search?FamilySearch=abcd&Page=1
/search?TimeSearch=abcd&Page=1
```
**请注意大小写**
后端返回一个JSON字符串，格式如下
```json
// FamilySearch, TimeSearch
{
    "Pages" : 30, // 需要显示的总页数
    "Result" : ["sha256", "sha256"] // 查询结果
}
```
对于Sha256Search，其返回的结果按照原网站设计，将全部的字典内容一一输出，例如返回结果如下
```json
{
    "Size" : "40KB",
    "Family" : "abc.abc.abc",
    "Time" : "2020"
}
```
那么显示的结果为
```
Size : 40KB,
Family : abc.abc.abc,
Time : 2020
```
当然，显示结果界面可以按照需要进行美化
## 样本特征feat文件
```json
{
    "Family" : "abc.abc"
}
```
## 其他
1. 每个脚本文件夹内部的**data文件夹是用于存放脚本临时文件**的，如果脚本需要储存一些临时文件或者数据，请放在此文件夹下，如果不需要存放临时数据，可以选择删除。
3. 主控程序的log部分还没有实现，等待后期开发。
# nkvt使用说明

## 主功能函数

### vtStart

> nkvt脚本的启动函数

```python
def vtStart(self)
```



###  downloadTables

> 类内执行下载的接口

```python
def downloadTables(self,nameNumber)
```

#### devporc

> 创建多进程，实现多进程的从virustotal打标签、下载json文件到nkvt中的临时目录

```python
def devporc(vtkey,list_sha256,dir_results)
```

#### out_get_vt_result

>打标签的函数主体

```python
def out_get_vt_result(vtkey, sha256, dir_results,n_all)
```

### movFiles

> 移动已经下载好的json文件到对应目录下

```python
def movFiles(self)
```

### isTomorrow
> 判断日期是否更新，为唤醒nkvt的首要标准 
```python
def isTomorrow(self)
```


## 命令行指令

### 启动

```
nkvt start
```

### 停止

```
nkvt stop
```

### 查看nkvt当前运行状态

```
nkvt state
```

### 查看已打标签的数量

```
nkvt count
```


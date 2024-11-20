# 数据库

项目使用了MySQL数据库。数据库中存储了样本的信息和恶意域名信息，还有Web的用户信息。

## 恶意样本数据
样本信息存储在256个sample表格中。
因为样本数量在千万级别，全都写到一个表格中访问速度非常慢。
基于样本sha256哈希值的前两个字符，将样本数据分别存储在256个数据表中。
数据表的命名规则为"sample\_xy"。
x和y是sha256哈希值的前两位，取值范围为0到9，a到f。
例如样本的sha256哈希值是12xxxxx，该样本的信息就保存在sample\_12的表格中。

## 恶意域名数据
恶意域名的信息存储在domain表格中。
数据库每天创建一个domain表格，表格的命名规则为"domain\_yyyymmdd"。
例如2024年1月1日的domain表格的名字为domain\_20240101。

## Web用户信息

Web用户登录信息保存在user表格中。


## 数据库初始化
init\_db.py程序实现项目数据库的初始化工作，会创建数据库和sample表格、domain表格、user表格。


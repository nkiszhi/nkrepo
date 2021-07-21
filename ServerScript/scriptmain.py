#!/usr/bin/python
#-- coding:utf8 --
from main.ScriptManager import ScriptManager
from main.ExceptionManager import ExceptionManager
from main.Models import *

def main() :
    # 初始化
    script = ScriptManager()

    expm = script.getExceptionManager()

    while True :
        try :
            # 启动服务
            script.auto()
        except Exception as e:
            print(e)
            expm.handle()

if __name__ == '__main__' :
    main()
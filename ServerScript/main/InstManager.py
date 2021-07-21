from main.Server import Client

# 特权级指令
PRIVILEGE = 1
# 普通指令
NORMAL = 0

class InstManager :
    def __init__(self):
        # 指令字典
        self.__instList = {
            'nkaz' : {},
            'nkvt' : {},
            'nkvs' : {},
            'main' : {}
        }
        # 指令等级
        self.__instLevel = [NORMAL, PRIVILEGE]

    # 注册指令，inst表示指令内容，targetfunc表示指令绑定的函数，targetscript表示目标脚本，description表示指令描述，level为指令等级
    def addInstruction(self, inst : str, targetfunc : callable([type(Client), tuple, any]), targetscript : str, description = "", level = NORMAL) :
        # 检查输入是否合法
        if inst == "" or (targetscript not in self.__instList.keys()) or (level not in self.__instLevel) or inst in self.__instList.keys() : 
            raise Exception("Illegal Instruction.")
        # 检查指令是否存在
        if inst in self.__instList[targetscript].keys() :
            raise Exception("Duplicated instruction.")
        # 注册指令
        self.__instList[targetscript][inst] = (targetfunc, level, description)
    
    # 获得帮助
    def getHelp(self) :
        helpmessage = "Usage: [targetscript] [instruction] [args]"
        for targetscript, funcs in self.__instList.items() :
            helpmessage += '\n' + targetscript + " : "
            for funcname, func in funcs.items() :
                helpmessage += "\n\t" + funcname + "\n\t\t" + func[2]
        return helpmessage

    # 移除某个指令
    def removeInstruction(self, inst : str, targetscript : str) :
        if targetscript not in self.__instList.keys() : return
        if inst not in self.__instList[targetscript].keys() : return
        self.__instList[targetscript].pop(inst)
    
    # 解析指令
    def parseInstruction(self, inst : str) :
        # 去除首尾的空格等，防止干扰
        inst.strip()
        # 检查指令是否合法
        if inst == "" : return []
        # 分解
        insts = inst.split(" ")
        if len(insts) == 0 : return []
        # 搜索
        if insts[0] in self.__instList.keys() :
            if len(insts) >= 2 and insts[1] in self.__instList[insts[0]].keys() : 
                args = []
                if len(insts) >= 3 : args = insts[2:]
                return [self.__instList[insts[0]][insts[1]][0], self.__instList[insts[0]][insts[1]][1], args]
        if insts[0] in self.__instList['main'].keys() :
            args = []
            if len(insts) >= 2 : args = insts[1:]
            return [self.__instList['main'][insts[0]][0], self.__instList['main'][insts[0]][1], args]
        return []
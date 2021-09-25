import os
hexstring = "0123456789abcdef"
sha256_list = []
for i in hexstring:
    for j in hexstring:
        for k in hexstring:
            for l in hexstring:
                folder = "./DATA/"+i+"/"+j+"/"+k+"/"+l+"/"
                list_all = os.listdir(folder)
                for f in list_all:
                    if len(f)==64:
                        sha256_list.append(f)
                        print(f)
a=0
with open('total_sha256_name.json','w') as f:
    for i in sha256_list:
        f.write("%s\n" % i)
        a+=1
with open('number_of_sample','w') as f:
    f.write(a)


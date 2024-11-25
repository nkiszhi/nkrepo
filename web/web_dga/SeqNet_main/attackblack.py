import os
import sys

# import magic
import numpy as np
from secml.array import CArray

# from models.model import MalNet
from secml_malware.models.malconv import MalConv
# from data.transforms import get_trans 
# import torch.nn as nn

from secml_malware.attack.blackbox.c_wrapper_phi import CEnd2EndWrapperPhi
from secml_malware.attack.blackbox.ga.c_base_genetic_engine import CGeneticAlgorithm
from secml_malware.models.c_classifier_end2end_malware import CClassifierEnd2EndMalware, End2EndModel
# from secml_malware.models.malconv import MalConv

from secml_malware.attack.blackbox.c_gamma_sections_evasion import CGammaSectionsEvasionProblem

# need to install Ember
# https://github.com/elastic/ember
# 
# pip install git+https://github.com/elastic/ember.git

goodware_folder = './data/NewDataset/test/norm' #INSERT GOODWARE IN THAT FOLDER
# net = 'malconv'

number = sys.argv[1]

# model = MalNet.load_from_checkpoint(checkpoint_path='./log/malconv/checkpoint/malconv-epoch=60-Accuracy=0.98.ckpt', logdir='./log', net=net)

# model = CClassifierEnd2EndMalware(model)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

net = CClassifierEnd2EndMalware(MalConv())
net.load_pretrained_model()
net = CEnd2EndWrapperPhi(net)

section_population, what_from_who = CGammaSectionsEvasionProblem.create_section_population_from_folder(goodware_folder, how_many=10, sections_to_extract=['.rdata'])

attack = CGammaSectionsEvasionProblem(section_population, net, population_size=10, penalty_regularizer=1e-6, iterations=int(number), threshold=0)

from secml_malware.attack.whitebox.c_header_evasion import CHeaderEvasion

folder = './data/NewDataset/whiteattack500/mal'  #INSERT MALWARE IN THAT FOLDER
newfolder = './data/NewDataset/blackattack500_'+number+'/mal'
X = []
y = []
file_names = []

count = 0
for i, f in enumerate(os.listdir(folder)):
    path = os.path.join(folder, f)
    # if "PE32" not in magic.from_file(path):
    #     continue
    with open(path, "rb") as file_handle:
        code = file_handle.read()
    
    x = CArray(np.frombuffer(code, dtype=np.uint8)).atleast_2d()
    _, confidence = net.predict(x, True)

    # if confidence[0, 1].item() < 0.5:
    #     continue

    print(f">{i} Added {f} with confidence {confidence[0,1].item()}")
    X.append(x)
    conf = confidence[1][0].item()
    y.append([1 - conf, conf])
    file_names.append(path)

engine = CGeneticAlgorithm(attack)


for i, (sample, label) in enumerate(zip(X, y)):
    y_pred, adv_score, adv_ds, f_obj = engine.run(sample, CArray(label[1]))
    print(engine.confidences_)
    print(f_obj)

    adv_x = adv_ds.X[0,:]
    engine.write_adv_to_file(adv_x,'adv_exe')
    engine.write_adv_to_file(adv_x,os.path.join(newfolder, os.path.basename(file_names[i])))
    with open('adv_exe', 'rb') as h:
        code = h.read()
    real_adv_x = CArray(np.frombuffer(code, dtype=np.uint8))
    _, confidence = net.predict(CArray(real_adv_x), True)
    print(confidence[0,1].item())






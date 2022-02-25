import os
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence
import numpy as np

from htk import HTKFile

CLASS_NAMES = ['NCS', 'CNS', 'CRY', 'OTH']

class NetVCM(nn.Module):
    def __init__(self, nInput, nHidden, nOutput):
        super(NetVCM, self).__init__()

        self.fc1 = nn.Linear(nInput, nHidden)
        self.fc2 = nn.Linear(nHidden, nHidden)
        self.fc3 = nn.Linear(nHidden, nHidden)
        self.fc4 = nn.Linear(nHidden, nOutput)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        return F.softmax(x, dim=1)


class NetLing(nn.Module):
    def __init__(self, nInput, nHidden, nOutput):
        super(NetLing, self).__init__()

        self.fc1 = nn.Linear(nInput, nHidden)
        self.fc2 = nn.Linear(nHidden, nHidden)
        self.fc3 = nn.Linear(nHidden, nOutput)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return F.softmax(x, dim=1)


class NetSyll(nn.Module):
    def __init__(self, nInput, nHidden, nOutput):
        super(NetSyll, self).__init__()

        self.fc1 = nn.Linear(nInput, nHidden)
        self.fc2 = nn.Linear(nHidden, nHidden)
        self.fc3 = nn.Linear(nHidden, nOutput)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return F.softmax(x, dim=1)

def load_model(path):
    # Create model and load weights
    vcm_net = NetVCM(nInput=88, nHidden=1024, nOutput=4)
    vcm_net.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))
    # Place model in evaluation mode
    vcm_net.eval()
    return vcm_net


def predict_vcm(model, input, mean_var):
    # Read normalisation parameters
    assert os.path.exists(mean_var)

    with open(mean_var, 'rb') as f:
        mv = pickle.load(f)

    m, v = mv['mean'], mv['var']
    std = lambda feat: (feat - m) / v

    # Load input feature and predict
    htk_reader = HTKFile()
    htk_reader.load(input)

    feat = std(np.array(htk_reader.data))
    input = torch.from_numpy(feat.astype('float32'))

    with torch.no_grad():
        output_ling = model(input).data.data.cpu().numpy()
    prediction_confidence = output_ling.max()  # post propability

    cls_ling = np.argmax(output_ling)
    predition_vcm = CLASS_NAMES[cls_ling]  # prediction

    return predition_vcm, prediction_confidence
import os
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence
import numpy as np

from .htk import HTKFile

# NCS = non-canonical syllable, CNS = canonical syllable, CRY = cry, OTH = laugh+junk
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
    vcm_net.to('cpu')
    return vcm_net


def predict_vcm(model, input, mean_var, device):
    # Read normalisation parameters
    assert os.path.exists(mean_var)

    # Get scaler
    with open(mean_var, 'rb') as f:
        mv = pickle.load(f)
    m, v = mv['mean'], mv['var']
    scale = lambda feat: (feat - m) / v

    # Load input features
    try:
        htk_reader = HTKFile()
        htk_reader.load(input)
    except Exception as e:
        raise type(e)('HTK file could not be read properly! '
                      'Base Exception: {}'.format(e))

    # Scale input
    feat = scale(np.array(htk_reader.data))
    input = torch.from_numpy(feat.astype('float32')).to('cpu')

    # Do prediction
    with torch.no_grad():
        try:
            output_ling = model(input).data.data.cpu().numpy()
        except Exception as e:
            raise type(e)("Error: Cannot proceed with VCM prediction for file {}\n"
                          "Base Exception: {}".format(input, e))

    # Get class and confidence
    prediction_confidence = output_ling.max()  # post propability
    cls_ling = np.argmax(output_ling)
    predition_vcm = CLASS_NAMES[cls_ling]  # prediction

    return predition_vcm, prediction_confidence
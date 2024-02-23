from HiFiGan.models import Generator
import json
import torch

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

## Loads the Vocoder model from Hifi-Gan 
#(code taken from https://github.com/jik876/hifi-gan/blob/master/inference.py)
def load_model(config_path, device):
    ## Load the config from the json file 
    with open(config_path, 'r') as f:
        config = json.load(f)
    h = AttrDict(config)
    generator = Generator(h).to(device)
    generator.eval()
    generator.remove_weight_norm()

    return generator

def vocoder_infer(mel, vocoder):
    ## add 10 empty channels to dim 1
    # reverse the second dimension order
    with torch.no_grad():
        wav = vocoder(mel)
    return wav

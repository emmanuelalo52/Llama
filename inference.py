from typing import Optional
import torch
import time
from pathlib import Path
import json
from sentencepiece import SentencePieceProcessor
from tqdm import tqdm
from llama_architecture import *

class LLama:
    def __init__(self,model,tokenizer,config):
        self.model = model
        self.config = config
        self.tokenizer = tokenizer
    @staticmethod
    def build(checkpoint_dir,tokenizer_path,load_model,max_seq_len,max_batch_size,device):
        prev_time = time.time()
        if load_model:
            checkpoints = sorted(Path(checkpoint_dir).glob('*.pth'))
            assert len(checkpoints) > 0
            chk_path = checkpoints[0]
            print(f'Loading checkpoint {chk_path}')
            checkpoint = torch.load(chk_path,map_location="cpu")
            print(f'Loaded checkpoint in {(time.time()-prev_time):.2f}s')
            prev_time = time.time()
        with open(Path(checkpoint_dir)/"params.json",'r') as f:
            params = json.loads(f.read())
        config:LlamaConfig = LlamaConfig(
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            device=device,
            **params
        )
        tokenizer = SentencePieceProcessor()
        tokenizer.load(tokenizer_path)
        config.vocab_size = tokenizer.vocab_size()

        if device == 'cuda':
            torch.set_default_tensor_type(torch.cuda.HalfTensor)
        else:
            torch.set_default_tensor_type(torch.BFloat16Tensor)

        model = LLamaTransformer(config).to(device)

        if load_model:
            del checkpoint["rope.freqs"]
            model.load_state_dict(checkpoint,strict=True)
            print(f'Loaded state dict in {(time.time() - prev_time):.2f}s')
        return LLama(model,tokenizer,config)

if __name__=='__main__':
    allow_cuda = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = LLama.build(checkpoint_dir='C:/Users/hp/OneDrive/Documents/Llama/Llama-2-7b',
                        tokenizer_path='C:/Users/hp/OneDrive/Documents/Llama/Llama-2-7b/tokenizer.model',
                        load_model=True,
                        max_seq_len=1024,
                        max_batch_size=3,
                        device=device)
    print("It worked")
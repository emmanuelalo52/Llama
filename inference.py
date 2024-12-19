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
        self.tokenizer = tokenizer
        self.config = config
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
    
    def text_completion(self,prompt,temperature=0.6,top_p=0.9,max_gen_len=None):
        if max_gen_len is None:
            max_gen_len = self.config.max_seq_len - 1
        #convert prompts to tokens
        prompt_tokens = [self.tokenizer.encode(prompt,out_type=int,add_bos=True,add_eos=False) for prompt in prompts]
        batch_size = len(prompt_tokens)
        assert batch_size<=self.config.batch_size
        max_prompt_len = max(len(prompt)for prompt in prompts)
        assert max_prompt_len<=self.config.max_seq_len
        total_len = min(self.config.max_seq_len, max_gen_len + max_prompt_len)

        #paddig list of the tokens
        pad_id = self.tokenizer.pad_id()
        tokens = torch.full((batch_size,total_len),pad_id,dtype=torch.long,device=device)
        for k,t in enumerate(prompt_tokens):
            tokens[k,:len(t)] = torch.tensor(t,dtype=torch.long,device=device)
        
        #check if we've reached the end of sentence
        eos_reached = torch.Tensor([False] * batch_size,device=device)
        prompt_token_masks = tokens != pad_id
        for cur_pos in tqdm(range(1,total_len)):
            with torch.no_grad():
                logits = self.model.forward(tokens[:,cur_pos-1:cur_pos],cur_pos)
            if temperature > 0:
                probs = torch.softmax(logits[:,-1]/temperature,dim=-1)
                next_token = self._sample_top_p(probs,top_p)
            else:
                #use greedy search
                next_token = torch.argmax(logits[:, -1], dim=-1)
            next_token = next_token.reshape(-1)
            #replace padded tokens
            next_token = torch.where(prompt_tokens[:,cur_pos],tokens[:,cur_pos],next_token)
            tokens[:,cur_pos] = next_token
            #use EOS when we get to the end of the padded token
            eos_reached |= (~prompt_token_masks[:,cur_pos]) & (next_token == self.tokenizer.eos_id())
            if all(eos_reached):
                break

        out_tokens = []
        out_text = []
        for prompt_index, current_prompt_tokens in enumerate(tokens.tolist()):
            #cut output from EOS if present
            if self.tokenizer.eos_id() in current_prompt_tokens:
                eos_idx = current_prompt_tokens.index(self.tokenizer.eos_id())
                current_prompt_tokens = current_prompt_tokens[:eos_idx]
            out_tokens.append(current_prompt_tokens)
            out_text.append(self.tokenizer.decode(current_prompt_tokens))
        return (out_tokens,out_text)
    
    def _sample_top_p(self, probs, p):
        # (B, vocab_size)
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
        # (B, vocab_size)
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        # (B, vocab_size)
        # (Substracting "probs_sort" shifts the cumulative sum by 1 position to the right before masking)
        mask = probs_sum - probs_sort > p 
        # Zero out all the probabilities of tokens that are not selected by the Top P
        probs_sort[mask] = 0.0 
        # Redistribute the probabilities so that they sum up to 1.
        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
        # Sample a token (its index) from the top p distribution
        next_token = torch.multinomial(probs_sort, num_samples=1)
        # Get the token position in the vocabulary corresponding to the sampled index
        next_token = torch.gather(probs_idx, -1, next_token) 
        return next_token
        




if __name__ == '__main__':
    torch.manual_seed(0)

    allow_cuda = False
    device = 'cuda' if torch.cuda.is_available() and allow_cuda else 'cpu'

    prompts = [
        "Simply put, the theory of relativity states that ",
        "If Google was an Italian company founded in Milan, it would",
        # Few shot promt
        """Translate English to French:
        
        sea otter => loutre de mer
        peppermint => menthe poivrée
        plush girafe => girafe peluche
        cheese =>""",
        # Zero shot prompt
        """Tell me if the following person is actually Doraemon disguised as human:
        Name: Umar Jamil
        Decision: 
        """
    ]

    model = LLama.build(checkpoint_dir='C:/Users/hp/OneDrive/Documents/Llama/Llama-2-7b',
                        tokenizer_path='C:/Users/hp/OneDrive/Documents/Llama/Llama-2-7b/tokenizer.model',
                        load_model=True,
                        max_seq_len=1024,
                        max_batch_size=len(prompts),
                        device=device)

    out_tokens, out_texts = (model.text_completion(prompts, max_gen_len=64))
    assert len(out_texts) == len(prompts)
    for i in range(len(out_texts)):
        print(f'{out_texts[i]}')
        print('-' * 50)
# if __name__=='__main__':
#     torch.manual_seed(0)
#     allow_cuda = True
#     device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#     prompts = [""]
#     model = LLama.build(checkpoint_dir='C:/Users/hp/OneDrive/Documents/Llama/Llama-2-7b',
#                         tokenizer_path='C:/Users/hp/OneDrive/Documents/Llama/Llama-2-7b/tokenizer.model',
#                         load_model=True,
#                         max_seq_len=1024,
#                         max_batch_size=len(prompts),
#                         device=device)
#     print("It worked")
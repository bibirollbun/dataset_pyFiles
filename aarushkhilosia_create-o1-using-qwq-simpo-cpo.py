!pip install pip3-autoremove


!pip-autoremove torch torchvision torchaudio -y


!pip install -U torch torchvision torchaudio datasets trl accelerate triton bitsandbytes torchao transformers==4.47.0 --extra-index-url https://download.pytorch.org/whl/cu121


!pip install -U datasets trl accelerate triton bitsandbytes torchao transformers 


!HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download Qwen/Qwen2.5-Math-1.5B-Instruct


!HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download Aarushhh/QwQ-LongCoT-verified-DPO --repo-type dataset





# Load model directly
from transformers import AutoModelForCausalLM, AutoTokenizer,TorchAoConfig, BitsAndBytesConfig
import torch#, torchao

# quantization_config = TorchAoConfig("int8_weight_only")
# quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")
quantization_config = None
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-1.5B-Instruct")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Math-1.5B-Instruct", device_map="cuda:0", torch_dtype=torch.float16, max_position_embeddings=2048, quantization_config=quantization_config)
# torchao.quantization.utils.recommended_inductor_config_setter
# model=torch.compile(model, mode='reduce-overhead', fullgraph=True)


from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
    modules_to_save=['input_layernorm', 'post_attention_layernorm', 'norm'],
    r=64,
    lora_alpha=40,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
    # init_lora_weights='pissa_niter_16',
    use_rslora=True,
    
)


# model.enable_input_require_grads()


model = get_peft_model(model, lora_config)


model.print_trainable_parameters()



# from peft import replace_lora_weights_loftq
# replace_lora_weights_loftq(model)


from datasets import load_dataset
dataset = load_dataset("Aarushhh/QwQ-LongCoT-verified-DPO")


import os
import torch
import torch.distributed as dist

@torch.compile
def zeropower_via_newtonschulz5(G, steps):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    if G.size(0) > G.size(1):
        X = X.T

    # Ensure spectral norm is at most 1
    X = X / (X.norm() + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A # adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X
    
    if G.size(0) > G.size(1):
        X = X.T
    return X

class SWAN(torch.optim.Optimizer):
    """
    SWAN - SGD with normalization and Newton-schulz orthogonalization
    used Muon as base
    Muon - MomentUm Orthogonalized by Newton-schulz

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - We believe this optimizer is unlikely to work well for training with small batch size.
    - We believe it may not work well for finetuning pretrained models, but we haven't tested this.

    Arguments:
        muon_params: The parameters to be optimized by Muon.
        lr: The learning rate. The updates will have spectral norm of `lr`. (0.02 is a good default)
        momentum: The momentum used by the internal SGD. (0.95 is a good default)
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        ns_steps: The number of Newton-Schulz iterations to run. (6 is probably always enough)
        adamw_params: The parameters to be optimized by AdamW. Any parameters in `muon_params` which are
        {0, 1}-D or are detected as being the embed or lm_head will be optimized by AdamW as well.
        adamw_lr: The learning rate for the internal AdamW.
        adamw_betas: The betas for the internal AdamW.
        adamw_eps: The epsilon for the internal AdamW.
        adamw_wd: The weight decay for the internal AdamW.
    """
    def __init__(self, muon_params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=6,
                 adamw_params=None, adamw_lr=3e-4, adamw_betas=(0.8, 0.95), adamw_eps=1e-8, adamw_wd=0, model=None):
        self.ngpt_model = model
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps,
                        adamw_lr_ratio=adamw_lr/lr, adamw_betas=adamw_betas,
                        adamw_eps=adamw_eps, adamw_wd=adamw_wd)

        params = list(muon_params)
        adamw_params = list(adamw_params) if adamw_params is not None else []
        params.extend(adamw_params)
        self.gradnorms = []            
        super().__init__(params, defaults)
    
        # Sort parameters into those for which we will use Muon, and those for which we will not
        for group in self.param_groups:
            for p in group['params']:
                self.state[p]['use_muon'] = False
        for p in muon_params:
            # Use Muon for every parameter in muon_params which is >= 2D and doesn't look like an embedding or head layer
            if p.ndim >= 2:
                if p.size(0) < 10000 and p.size(1) < 10000:
                    self.state[p]['use_muon'] = True
                    self.gradnorms.append(torch.nn.LayerNorm(p.size(), elementwise_affine=False).to(p.device))
                else:
                    self.state[p]['use_muon'] = False
            else:
                self.state[p]['use_muon'] = False

        if 'WORLD_SIZE' in os.environ:
            self.world_size = int(os.environ['WORLD_SIZE'])
            self.rank = int(os.environ['RANK'])
        else:
            self.world_size = 1
            self.rank = 0

    def step(self, closure=None):
        
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:

            ############################
            #           Muon           #
            ############################

            params = [p for p in group['params'] if self.state[p]['use_muon']]
            gradnorms = self.gradnorms
            lr = group['lr']
            # momentum = group['momentum']

            # generate weight updates in distributed fashion
            total_params = sum(p.numel() for p in params)
            updates_flat = torch.zeros(total_params, device='cuda', dtype=torch.bfloat16)
            curr_idx = 0
            for i, p in enumerate(params):
                # luckily this will perfectly distribute a transformer with multiple of 4 layers to 8 GPUs
                if i % self.world_size == self.rank:
                    g = p.grad
                    if g is None:
                        continue
                    if g.ndim > 2:
                        g = g.view(g.size(0), -1)
                    assert g is not None
                    # state = self.state[p]
                    # if 'momentum_buffer' not in state:
                    #     state['momentum_buffer'] = torch.zeros_like(g)
                    # buf = state['momentum_buffer']
                    # buf.mul_(momentum).add_(g)
                    # if group['nesterov']:
                    #     g = g.add(buf, alpha=momentum)
                    # else:
                    #     g = buf
                    g = gradnorms[i](g)
                    g = zeropower_via_newtonschulz5(g, steps=group['ns_steps'])
                    g *= max(1, g.size(0)/g.size(1))**0.5
                    updates_flat[curr_idx:curr_idx+p.numel()] = g.flatten()
                curr_idx += p.numel()

            # sync updates across devices. we are not memory-constrained so can do this simple deserialization
            if self.world_size > 1:
                dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)

            # deserialize and apply updates
            curr_idx = 0
            for p in params:
                g = updates_flat[curr_idx:curr_idx+p.numel()].view_as(p.data).type_as(p.data)
                p.data.add_(g, alpha=-lr)
                curr_idx += p.numel()

            ############################
            #       AdamW backup       #
            ############################

            params = [p for p in group['params'] if not self.state[p]['use_muon']]
            lr = group['adamw_lr_ratio'] * group['lr'] # in order for lr schedule to work
            beta1, beta2 = group['adamw_betas']
            eps = group['adamw_eps']
            weight_decay = group['adamw_wd']

            for p in params:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                if 'step' not in state:
                    state['step'] = 0
                    state['moment1'] = torch.zeros_like(g)
                    state['moment2'] = torch.zeros_like(g)
                state['step'] += 1
                step = state['step']
                buf1 = state['moment1']
                buf2 = state['moment2']
                buf1.lerp_(g, 1-beta1)
                buf2.lerp_(g.square(), 1-beta2)

                g = buf1 / (eps + buf2.sqrt())

                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                scale = bias_correction1 / bias_correction2**0.5
                p.data.mul_(1 - lr * weight_decay)
                p.data.add_(g, alpha=-lr/scale)

        # self.ngpt_model.normalize_model()

        return loss


optimizer = SWAN(model.parameters(), lr=0.02,adamw_lr=0.002, model=model)


from torch.optim.lr_scheduler import _LRScheduler

class WarmupConstantScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, last_epoch=-1):
        self.warmup_steps = warmup_steps
        super(WarmupConstantScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            return [base_lr * (self.last_epoch / self.warmup_steps) 
                    for base_lr in self.base_lrs]
        # Constant LR after warmup
        return self.base_lrs


scheduler = WarmupConstantScheduler(optimizer, warmup_steps=100)



# train_cpo.py
from datasets import load_dataset
from trl import DPOConfig, DPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
max_seq_length = 2048

training_args = DPOConfig(
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 4,
    save_total_limit=5,
    num_train_epochs = 1,
    fp16 = True,
    bf16 = False,
    logging_steps = 1,
    # weight_decay = 0.0,
    lr_scheduler_type = "constant",
    seed = 69,
    output_dir = "outputs",
    dataset_num_proc = 48,
    gradient_checkpointing=True,
    # gradient_checkpointing_kwargs={"use_reentrant": False},
    neftune_noise_alpha=5,

    # loss_type="simpo",
)



trainer = DPOTrainer(model=model,
                     optimizers=(optimizer, scheduler),
                     args=training_args,
                     processing_class=tokenizer,
                     train_dataset=dataset['train'])



# # train_cpo.py
# from datasets import load_dataset
# from trl import CPOConfig, CPOTrainer
# from transformers import AutoModelForCausalLM, AutoTokenizer
# max_seq_length = 2048

# training_args = CPOConfig(
#     per_device_train_batch_size = 1,
#     gradient_accumulation_steps = 4,
#     save_total_limit=5,
#     num_train_epochs = 1,
#     fp16 = False,
#     bf16 = True,
#     logging_steps = 1,
#     # weight_decay = 0.0,
#     lr_scheduler_type = "constant",
#     seed = 69,
#     output_dir = "outputs",
#     dataset_num_proc = 48,
#     gradient_checkpointing=True,
#     # gradient_checkpointing_kwargs={"use_reentrant": False},
#     neftune_noise_alpha=5,

#     loss_type="simpo",
# )



# trainer = CPOTrainer(model=model,
#                      optimizers=(optimizer, scheduler),
#                      args=training_args,
#                      processing_class=tokenizer,
#                      train_dataset=dataset['train'])



import os
os.environ['WANDB_API_KEY'] = ""
os.environ['WANDB_PROJECT'] = 'try o1 qwen 2.5 math 1.5b DPO '


# model=torch.compile(model, mode='reduce-overhead', fullgraph=True)


from accelerate import notebook_launcher
notebook_launcher(trainer.train(), num_processes=2, mixed_precision='fp16',gradient_as_bucket_view=True)


trainer.train()








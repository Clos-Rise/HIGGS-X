import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.quantization import quantize_dynamic
from loader import HGSFormat
import gc

class ALMPQOptimizer:
    def __init__(self, model: nn.Module, device='cuda'):
        self.model = model
        self.device = device
        self.layer_precisions = {}
        self._analyze_layers()
        self.grad_threshold_high = 1e-3
        self.grad_threshold_low = 1e-5
        self.precision_levels = ['int8', 'bfloat16', 'float16']

    def _analyze_layers(self):
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                param_count = sum(p.numel() for p in module.parameters())
                if param_count > 1_000_000:
                    self.layer_precisions[name] = 'float16'
                elif param_count > 100_000:
                    self.layer_precisions[name] = 'bfloat16'
                else:
                    self.layer_precisions[name] = 'int8'
            else:
                self.layer_precisions[name] = 'float16'

    def apply_quantization(self):
        int8_modules = []
        for name, module in self.model.named_modules():
            if name in self.layer_precisions and self.layer_precisions[name] == 'int8':
                if isinstance(module, (nn.Linear, nn.LSTM, nn.GRU)):
                    int8_modules.append(name)

        if int8_modules:
            self.model = quantize_dynamic(self.model, {nn.Linear}, dtype=torch.qint8)
            print(f"ALMPQ: INT8 к слоям: {int8_modules}")

        print("ALMPQ: Квантование применено (FP16/BF16 через AMP, INT8)")

    def adapt_during_training(self, grads):
        changed = False
        for name, grad in grads.items():
            if grad is None or name not in self.layer_precisions:
                continue
            avg_grad = grad.abs().mean().item()
            current_precision = self.layer_precisions[name]
            if current_precision not in self.precision_levels:
                continue
            idx = self.precision_levels.index(current_precision)

            if avg_grad > self.grad_threshold_high and idx < len(self.precision_levels) - 1:
                new_precision = self.precision_levels[idx + 1]
                self.layer_precisions[name] = new_precision
                changed = True
                print(f"[ALMPQ] ^ '{name}' до {new_precision}")
            elif avg_grad < self.grad_threshold_low and idx > 0:
                new_precision = self.precision_levels[idx - 1]
                self.layer_precisions[name] = new_precision
                changed = True
                print(f"[ALMPQ]: v '{name}' до {new_precision}")

        if changed:
            self.apply_quantization()

class HIGGSXAccelerator:
    """HIGGS-X."""
    def __init__(self, model: nn.Module, optimizer_cls, device='cuda'):
        self.device = device
        self.model = model.to(device)
        self.optimizer = optimizer_cls(self.model.parameters())
        self.scaler = GradScaler(device=device, enabled=True)
        self.almpq = ALMPQOptimizer(self.model, device)
        self.is_ultra = False

    def Ultra(self, on="Y"):
        if on.upper() == "Y":
            self.is_ultra = True
            print("HIGGS-X: Ultra ON!")
            self._apply_ultra_optimizations()
        else:
            self.is_ultra = False
            print("HIGGS-X: Ultra OFF!")

    def _apply_ultra_optimizations(self):
        self._dynamic_weight_quantization()
        self._gradient_checkpointing()

    def _dynamic_weight_quantization(self):
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                module.to(torch.float16)

    def _gradient_checkpointing(self):
        try:
            from torch.utils.checkpoint import checkpoint
            self.model.gradient_checkpointing_enable()
        except Exception as e:
            print(f"Gradient Checkpoint Error: {e}")

    def load_hgs_model(self, hgs_filepath):
        state_dict = HGSFormat.load_hgs(hgs_filepath, device=self.device)
        self.model.load_state_dict(state_dict)
        print(f"HIGGS-X: Модель из {hgs_filepath}")
        self.almpq.apply_quantization()

    def train_step(self, data, target):
        self.model.train()
        self.optimizer.zero_grad()
        with autocast(device_type=self.device, enabled=True):
            output = self.model(data.to(self.device))
            loss = nn.functional.cross_entropy(output, target.to(self.device))

        self.scaler.scale(loss).backward()

        grads = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grads[name] = param.grad.detach().cpu()

        self.almpq.adapt_during_training(grads)

        self.scaler.step(self.optimizer)
        self.scaler.update()
        return loss.item()

    def infer(self, data):
        self.model.eval()
        with torch.no_grad(), autocast(device_type=self.device, enabled=True):
            output = self.model(data.to(self.device))
        return output

    @staticmethod
    def get_model_size_bytes(model: nn.Module) -> int:
        total_bytes = 0
        for param in model.parameters():
            total_bytes += param.numel() * param.element_size()
        return total_bytes

    @staticmethod
    def measure_load_time(load_func, *args, **kwargs) -> float:
        import time
        start = time.perf_counter()
        load_func(*args, **kwargs)
        end = time.perf_counter()
        return end - start

    def generate(self, model, inputs, attention_mask, max_length=200, do_sample=True, top_p=0.9, temperature=0.8, eos_token_id=None, pad_token_id=None):
        """Генерация текста."""
        model.eval()
        with torch.no_grad():
            with autocast(device_type=self.device, dtype=torch.bfloat16, enabled=self.is_ultra):
                output = model.generate(
                    inputs,
                    attention_mask=attention_mask,
                    max_length=max_length,
                    do_sample=do_sample,
                    top_p=top_p,
                    temperature=temperature,
                    eos_token_id=eos_token_id,
                    pad_token_id=pad_token_id
                )
        return output

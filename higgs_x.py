import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.quantization import quantize_dynamic, default_dynamic_qconfig
from loader import HGSFormat

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
        """
        Инициализация точности.
        """
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
        #Квантование INT8
        int8_modules = []
        for name, module in self.model.named_modules():
            if name in self.layer_precisions and self.layer_precisions[name] == 'int8':
                if isinstance(module, (nn.Linear, nn.LSTM, nn.GRU)):
                    int8_modules.append(name)

        if int8_modules:
            self.model = quantize_dynamic(self.model, {nn.Linear}, dtype=torch.qint8)
            print(f"ALMPQ: Применено динамическое INT8 квантование к слоям: {int8_modules}")

        print("ALMPQ: Квантование применено (FP16/BF16 через AMP, INT8)")

    def adapt_during_training(self, grads):
        """
        Адаптивно меняет точность.
        """
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
                print(f"[ALMPQ] Повышаем точность слоя '{name}' до {new_precision}")
            elif avg_grad < self.grad_threshold_low and idx > 0:
                new_precision = self.precision_levels[idx - 1]
                self.layer_precisions[name] = new_precision
                changed = True
                print(f"[ALMPQ]: Понижаем точность слоя '{name}' до {new_precision}")

        if changed:
            self.apply_quantization()


class HIGGSXAccelerator:
    """
    Виртуальный ускоритель HIGGS-X с поддержкой ALMPQ и загрузкой из .hgs.
    """

    def __init__(self, model: nn.Module, optimizer_cls, device='cuda'):
        self.device = device
        self.model = model.to(device)
        self.optimizer = optimizer_cls(self.model.parameters())
        self.scaler = GradScaler(device=device)
        self.almpq = ALMPQOptimizer(self.model, device)
        self._initialize_model()

    def _initialize_model(self):
        self.almpq.apply_quantization()
        self.model.to(self.device)

    def load_hgs_model(self, hgs_filepath):
        """
        Загружает веса из .hgs файла и применяет их к модели.
        """
        state_dict = HGSFormat.load_hgs(hgs_filepath, device=self.device)
        self.model.load_state_dict(state_dict)
        print(f"HIGGS-X: Модель загружена из {hgs_filepath}")
        self.almpq.apply_quantization()

    def train_step(self, data, target):
        self.model.train()
        self.optimizer.zero_grad()
        with autocast(self.device):
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
        with torch.no_grad(), autocast(self.device):
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

import struct
import torch
import numpy as np


class HGSFormat:
    MAGIC = b'HGS1'

    @staticmethod
    def quantize_tensor(tensor: torch.Tensor, bits=8):
        tensor_cpu = tensor.detach().cpu()
        min_val = tensor_cpu.min()
        max_val = tensor_cpu.max()
        scale = (max_val - min_val) / (2**bits - 1)
        if scale == 0:
            scale = 1e-8
        q_tensor = ((tensor_cpu - min_val) / scale).round().clamp(0, 2**bits - 1).to(torch.uint8)
        return q_tensor.numpy(), float(min_val), float(scale)

    @staticmethod
    def dequantize_tensor(q_data, min_val, scale):
        return torch.tensor(q_data, dtype=torch.float32) * scale + min_val

    @staticmethod
    def save_hgs(model_state_dict, filepath):
        layers = list(model_state_dict.items())
        with open(filepath, 'wb') as f:
            f.write(HGSFormat.MAGIC)
            f.write(struct.pack('<I', len(layers)))

            index_pos = f.tell()
            index_data = []
            for name, tensor in layers:
                name_bytes = name.encode('utf-8')
                q_data, min_val, scale = HGSFormat.quantize_tensor(tensor, bits=8)
                index_data.append((name_bytes, tensor.shape, tensor.numel(), min_val, scale, 0, q_data.nbytes))
                f.write(struct.pack('<I', len(name_bytes)))
                f.write(name_bytes)
                f.write(struct.pack('<I', len(tensor.shape)))
                for dim in tensor.shape:
                    f.write(struct.pack('<Q', dim))
                f.write(struct.pack('<f', min_val))
                f.write(struct.pack('<f', scale))
                f.write(struct.pack('<Q', tensor.numel()))
                f.write(struct.pack('<Q', q_data.nbytes))
                f.write(struct.pack('<Q', 0))

            offsets = []
            for _, _, _, _, _, _, size_bytes in index_data:
                offsets.append(f.tell())
                q_data = index_data.pop(0)[6]

    @staticmethod
    def save_hgs(model_state_dict, filepath):
        layers = list(model_state_dict.items())
        with open(filepath, 'wb') as f:
            f.write(HGSFormat.MAGIC)
            f.write(struct.pack('<I', len(layers)))

            index_pos = f.tell()
            index_data = []
            quantized_datas = []
            for name, tensor in layers:
                name_bytes = name.encode('utf-8')
                q_data, min_val, scale = HGSFormat.quantize_tensor(tensor, bits=8)
                index_data.append({
                    'name_bytes': name_bytes,
                    'shape': tensor.shape,
                    'numel': tensor.numel(),
                    'min_val': min_val,
                    'scale': scale,
                    'q_data_len': q_data.nbytes,
                    'offset': 0
                })
                quantized_datas.append(q_data)

            for entry in index_data:
                f.write(struct.pack('<I', len(entry['name_bytes'])))
                f.write(entry['name_bytes'])
                f.write(struct.pack('<I', len(entry['shape'])))
                for dim in entry['shape']:
                    f.write(struct.pack('<Q', dim))
                f.write(struct.pack('<f', entry['min_val']))
                f.write(struct.pack('<f', entry['scale']))
                f.write(struct.pack('<Q', entry['numel']))
                f.write(struct.pack('<Q', entry['q_data_len']))
                f.write(struct.pack('<Q', 0))

            for i, q_data in enumerate(quantized_datas):
                index_data[i]['offset'] = f.tell()
                f.write(q_data.tobytes())
            f.seek(index_pos)
            for entry in index_data:
                f.seek(f.tell() + 4 + len(entry['name_bytes']) + 4 + 8 * len(entry['shape']) + 4 + 4 + 8 + 8)
                f.write(struct.pack('<Q', entry['offset']))

    @staticmethod
    def load_hgs(filepath, device='cpu'):
        state_dict = {}
        with open(filepath, 'rb') as f:
            magic = f.read(4)
            if magic != HGSFormat.MAGIC:
                raise RuntimeError("Неверный формат файла")

            num_layers = struct.unpack('<I', f.read(4))[0]

            index = []
            for _ in range(num_layers):
                name_len = struct.unpack('<I', f.read(4))[0]
                name = f.read(name_len).decode('utf-8')
                shape_len = struct.unpack('<I', f.read(4))[0]
                shape = tuple(struct.unpack('<Q', f.read(8))[0] for _ in range(shape_len))
                min_val = struct.unpack('<f', f.read(4))[0]
                scale = struct.unpack('<f', f.read(4))[0]
                numel = struct.unpack('<Q', f.read(8))[0]
                q_data_len = struct.unpack('<Q', f.read(8))[0]
                offset = struct.unpack('<Q', f.read(8))[0]
                index.append({
                    'name': name,
                    'shape': shape,
                    'min_val': min_val,
                    'scale': scale,
                    'numel': numel,
                    'q_data_len': q_data_len,
                    'offset': offset
                })

            for entry in index:
                f.seek(entry['offset'])
                q_data_bytes = f.read(entry['q_data_len'])
                q_data = np.frombuffer(q_data_bytes, dtype=np.uint8)
                tensor = HGSFormat.dequantize_tensor(q_data, entry['min_val'], entry['scale'])
                tensor = tensor.reshape(entry['shape']).to(device)
                state_dict[entry['name']] = tensor

        return state_dict

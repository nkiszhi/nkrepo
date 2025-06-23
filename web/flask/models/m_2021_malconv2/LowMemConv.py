import torch
import torch.nn as nn
import torch.nn.functional as F

def drop_zeros_hook(module, grad_input, grad_out):
    """
    用于将全为零的梯度替换为 None，从而节省内存和计算资源。
    在 PyTorch 中，梯度为 None 将不会反向传播。
    这是稀疏反向传播的一种近似实现，避免了冗余和无用的计算。
    """
    grads = []
    with torch.no_grad():
        for g in grad_input:
            if torch.nonzero(g).shape[0] == 0:  # ITS ALL EMPTY!
                grads.append(g.to_sparse())
            else:
                grads.append(g)
    return tuple(grads)

class CatMod(torch.nn.Module):
    """
    一个简单的模块，用于沿最后一个维度（dim=2）拼接张量。
    """
    def __init__(self):
        super(CatMod, self).__init__()

    def forward(self, x):
        return torch.cat(x, dim=2)


class LowMemConvBase(nn.Module):
    """
    一个基础的低内存卷积模型，用于处理长序列输入。
    """
    def __init__(self, chunk_size=65536, overlap=512, min_chunk_size=1024):
        """
        :param chunk_size: 每次处理的最大字节数，增加该值可能提高计算效率，但会消耗更多内存。
        :param overlap: 每块之间的重叠字节数。
        :param min_chunk_size: 最小处理块的大小
        """
        super(LowMemConvBase, self).__init__()
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_chunk_size = min_chunk_size

        # 用于高效的时间维度池化
        self.pooling = nn.AdaptiveMaxPool1d(1)
        # 用于张量拼接的模块
        self.cat = CatMod()
        self.cat.register_backward_hook(drop_zeros_hook)
        # 感受野、步长和输出通道的占位符
        self.receptive_field = None
        # 用于强制 checkpoint 正常工作的辅助张量
        self.dummy_tensor = torch.ones(1, dtype=torch.float32, requires_grad=True)

    def processRange(self, x, **kwargs):
        """
        将输入的 LongTensor 形状 (B, L) 转换为 (B, C, L)，
        其中 B 是批量大小，L 是输入长度，C 是通道数。
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def _get_device(self):
        """
        获取模型所在的设备。
        """
        return next(self.parameters()).device if self.parameters() else torch.device("cpu")

    def determinRF(self):
        """
        自动确定子网络的感受野和步长。
        如果之前已经计算过感受野，则直接返回。
        """
        if self.receptive_field is not None:
            return self.receptive_field, self.stride, self.out_channels

        cur_device = self._get_device()
        min_rf = 1
        max_rf = self.chunk_size

        with torch.no_grad():
            tmp = torch.zeros((1, max_rf), dtype=torch.long, device=cur_device)
            while True:
                test_size = (min_rf + max_rf) // 2
                try:
                    self.processRange(tmp[:, :test_size])
                    max_rf = test_size
                except:
                    min_rf = test_size + 1
                if max_rf == min_rf:
                    self.receptive_field = min_rf
                    out_shape = self.processRange(tmp).shape
                    self.stride = self.chunk_size // out_shape[2]
                    self.out_channels = out_shape[1]
                    break
        return self.receptive_field, self.stride, self.out_channels

    def pool_group(self, *args):
        """
        通过池化操作将多个张量的特征进行组合。
        """
        x = self.cat(args)
        x = self.pooling(x)
        return x

    def seq2fix(self, x, pr_args={}):
        """
        将输入的长整型张量 (B, L) 转换为固定长度表示 (B, C)。

        参数:
        - x: 输入张量，形状为 (B, L)。
        - pr_args: 传递给 `processRange` 的额外参数。
        返回:
        - x_selected: 固定长度表示张量，形状为 (B, C)。
        """
        receptive_window, stride, out_channels = self.determinRF()
        if x.shape[1] < receptive_window:
            x = F.pad(x, (0, receptive_window - x.shape[1]), value=0)

        batch_size, length = x.shape
        cur_device = self._get_device()

        winner_values = torch.full((batch_size, out_channels), -1.0, device=cur_device)
        winner_indices = torch.zeros((batch_size, out_channels), dtype=torch.int64, device=cur_device)

        step = self.chunk_size
        start = 0
        end = start + step

        with torch.no_grad():
            while start < end and (end - start) >= max(self.min_chunk_size, receptive_window):
                x_sub = x[:, start:end].to(cur_device)
                activs = self.processRange(x_sub.long(), **pr_args)
                activ_win, activ_indx = F.max_pool1d(activs, kernel_size=activs.shape[2], return_indices=True)
                activ_win = activ_win.squeeze(-1)
                activ_indx = activ_indx.squeeze(-1)
                selected = winner_values < activ_win
                winner_indices[selected] = activ_indx[selected] * stride + start
                winner_values[selected] = activ_win[selected]
                start = end
                end = min(start + step, length)

        final_indices = [torch.unique(winner_indices[b]) for b in range(batch_size)]
        chunk_list = []
        for b in range(batch_size):
            chunks = [x[b:b + 1, max(i - receptive_window, 0):min(i + receptive_window, length)] for i in final_indices[b]]
            chunk_list.append(torch.cat(chunks, dim=1).squeeze(0))

        x_selected = torch.nn.utils.rnn.pad_sequence(chunk_list, batch_first=True)
        x_selected = x_selected.to(cur_device)
        x_selected = self.processRange(x_selected.long(), **pr_args)
        x_selected = self.pooling(x_selected)
        x_selected = x_selected.view(x_selected.size(0), -1)

        return x_selected
import torch
import warnings


class CheckpointFunction(torch.autograd.Function):
    """
    梯度检查点功能，通过在反向传播时重新计算前向传播结果来减少内存消耗。
    适用于长序列处理或深层网络的内存优化。
    """

    @staticmethod
    def forward(ctx, run_function, num_inputs, *args):
        """
        前向传播阶段：
        - 保存输入张量和参数
        - 在无梯度模式下运行前向传播
        """
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:num_inputs])
        ctx.input_params = list(args[num_inputs:])

        # 分离输入张量但保留requires_grad属性
        ctx.input_tensors = [
            inp.detach().requires_grad_(inp.requires_grad)
            for inp in ctx.input_tensors
        ]

        # 检查是否有需要梯度的输入
        if not any(inp.requires_grad for inp in ctx.input_tensors):
            warnings.warn("None of the inputs have requires_grad=True. Gradients will be None")

        with torch.no_grad():
            output = run_function(*ctx.input_tensors)

        # 保存输出形状用于反向传播
        ctx.output_shape = output.shape if isinstance(output, torch.Tensor) else None
        return output

    @staticmethod
    def backward(ctx, *output_grads):
        """
        反向传播阶段：
        - 重新构建输入张量
        - 在有梯度模式下重新计算前向传播
        - 计算梯度并返回
        """
        # 恢复输入张量的requires_grad属性
        ctx.input_tensors = [
            inp.detach().requires_grad_(True)
            for inp in ctx.input_tensors
        ]

        with torch.enable_grad():
            # 重新计算前向传播
            output = ctx.run_function(*ctx.input_tensors)

            # 处理多输出情况
            if not isinstance(output, tuple):
                output = (output,)

            # 调整梯度维度以匹配输出形状
            processed_grads = []
            for grad, out in zip(output_grads, output):
                if grad is None:
                    processed_grads.append(None)
                    continue
                if ctx.output_shape and grad.shape != ctx.output_shape:
                    grad = grad.expand(ctx.output_shape)
                processed_grads.append(grad)

            # 计算梯度
            input_grads = torch.autograd.grad(
                outputs=output,
                inputs=ctx.input_tensors + ctx.input_params,
                grad_outputs=processed_grads,
                allow_unused=True
            )

        # 分离梯度以避免内存泄漏
        return (None, None) + tuple(
            grad.detach() if grad is not None else None
            for grad in input_grads
        )
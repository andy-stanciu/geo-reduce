import torch
import torch.nn as nn
import torch.distributed as dist


class DDPBucketedParameters(nn.Module):
    def __init__(self, module: torch.nn.Module, bucket_size_mb: float):
        super().__init__()
        self.module = module
        self.handles = []
        self.world_size = dist.get_world_size()
        
        # broadcast initial weights from rank 0 to all other ranks
        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)
        
        # create buckets in reverse parameter order
        self.buckets = self._create_buckets(bucket_size_mb)
        self.bucket_ready_grads = [set() for _ in range(len(self.buckets))]
        
        # register backward hooks on each parameter
        for param in self.module.parameters():
            if param.requires_grad:
                bucket_idx = self._find_bucket_for_param(param)
                param.register_post_accumulate_grad_hook(
                    self._make_grad_hook(param, bucket_idx)
                )
    

    def _create_buckets(self, bucket_size_mb: float):
        bucket_size_bytes = bucket_size_mb * 1024 * 1024
        buckets = []
        current_bucket = []
        current_size = 0
        
        # iterate through parameters in reverse order
        params_list = list(self.module.parameters())
        for param in reversed(params_list):
            if not param.requires_grad:
                continue
            
            param_size = param.numel() * param.element_size()
            
            if current_size + param_size > bucket_size_bytes and len(current_bucket) > 0:
                buckets.append(current_bucket)
                current_bucket = []
                current_size = 0
            
            current_bucket.append(param)
            current_size += param_size
        
        if len(current_bucket) > 0:
            buckets.append(current_bucket)
        
        return buckets
    

    def _find_bucket_for_param(self, param):
        for idx, bucket in enumerate(self.buckets):
            for bucket_param in bucket:
                if bucket_param is param:
                    return idx
        return None
    

    def _make_grad_hook(self, param, bucket_idx):
        def hook(_):
            self.bucket_ready_grads[bucket_idx].add(param)
            
            # if all gradients in this bucket are ready, all reduce
            if len(self.bucket_ready_grads[bucket_idx]) == len(self.buckets[bucket_idx]):
                self._all_reduce_bucket(bucket_idx)
        
        return hook
    

    def _all_reduce_bucket(self, bucket_idx):
        bucket = self.buckets[bucket_idx]
        
        # flatten all gradients in this bucket
        grads = []
        for param in bucket:
            if param.grad is not None:
                grads.append(param.grad.view(-1))
        
        if len(grads) > 0:
            flat_grad = torch.cat(grads)
            flat_grad.div_(self.world_size)
            handle = dist.all_reduce(flat_grad, op=dist.ReduceOp.SUM, async_op=True)
            self.handles.append((handle, bucket_idx, flat_grad))
    

    def forward(self, *inputs, **kwargs):
        self.bucket_ready_grads = [set() for _ in range(len(self.buckets))]
        return self.module(*inputs, **kwargs)
    

    def finish_gradient_synchronization(self):
        for handle, bucket_idx, flat_grad in self.handles:
            handle.wait()
            
            offset = 0
            for param in self.buckets[bucket_idx]:
                if param.grad is not None:
                    numel = param.grad.numel()
                    param.grad.copy_(flat_grad[offset:offset + numel].view_as(param.grad))
                    offset += numel
        
        self.handles.clear()
        self.bucket_ready_grads = [set() for _ in range(len(self.buckets))]

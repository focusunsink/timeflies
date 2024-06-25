import time 

HardwareSpec = {
    "A10":
    {
        "compute_ablility": 125 * 2 ** 40,
        "bandwidth": 600 * 2 ** 30,
    }
}



class TimeFlies:
    
    def __init__(self, 
                 model_type="llama2",
                 seq_len = 100,
                 hidden_size = 1024,
                 heads = 16,
                 head_size=64,
                 layer_num = 30,
                 machine="A10",
                 dtype="float16"):

        if machine == "A10" or machine == "a10":
            self.compute_ablility = HardwareSpec["A10"]["compute_ablility"]
            self.bandwidth = HardwareSpec["A10"]["bandwidth"]
        self.dtype = dtype
        if self.dtype == "int8" or self.dtype == "fp8":
            self.dtype_size = 1
        if self.dtype == "float16":
            self.dtype_size = 2
        if self.dtype == "float32":
            self.dtype_size = 4

        self.seq_len = seq_len 
        self.hidden_size = hidden_size
        self.heads = heads 
        self.head_size = head_size
        self.layer_num = layer_num 
        self.exp_inst = 4
        self.rsqrt_inst = 4

    def get_bottleneck_time(self, layer_type, cptt, rwt):
        if cptt > rwt:
            print(f'{layer_type} is compute bottleneck, ', end = '')
            bnt = cptt
        else:
            print(f'{layer_type} is memory bottleneck ', end = '')
            bnt = rwt 
        cptt = cptt * 1000000
        rwt = rwt * 1000000
        print(f'compute time is {cptt:.6f} us, read write time is {rwt:.6f} us')
        return bnt

    def LlamaRMSNorm(self):
        # pow
        pow_ops = self.seq_len * self.hidden_size 
        # mean
        mean_ops = self.seq_len * self.hidden_size * 2
        # + eps
        eps_ops = self.seq_len 
        # sqrt
       
        rsqrt_ops = self.seq_len * self.rsqrt_inst
        # x / rsqrt 
        norm_ops = self.seq_len * self.hidden_size 
        # * weight
        mul_weight_ops = self.seq_len * self.hidden_size 
        
        total_compute_ops = pow_ops + mean_ops + eps_ops + rsqrt_ops + norm_ops + mul_weight_ops
        print(total_compute_ops)
        total_compute_time = total_compute_ops / self.compute_ablility
        print(total_compute_time)

        input_bytes = self.seq_len * self.hidden_size * self.dtype_size
        weight_bytes = self.hidden_size * self.dtype_size 
        output_bytes = self.seq_len * self.hidden_size * self.dtype_size 

        total_read_write_bytes = input_bytes + weight_bytes + output_bytes 
        total_read_write_time = total_read_write_bytes / self.bandwidth
         
        bnt = self.get_bottleneck_time("layernorm", total_compute_time, total_read_write_time)
        
        return bnt
    
    def QKV_MatMul(self):
        # Matmul input featues = self.hidden_size, output features = self.hidden_size * 3  
        # compute 
        total_ops = self.seq_len * self.hidden_size * self.hidden_size * 3 
        total_compute_time = total_ops / self.compute_ablility
        
        # read write 
        input_read = self.seq_len * self.hidden_size * self.dtype_size 
        weight_read = self.hidden_size * self.hidden_size * 3 * self.dtype_size 
        output_write = self.seq_len * 3 * self.hidden_size 
        total_read_write_bytes = input_read + weight_read + output_write 
        total_read_write_time = total_read_write_bytes / self.bandwidth
    
        bnt = self.get_bottleneck_time("qkv", total_compute_time, total_read_write_time)

        return bnt

    def ROPE(self):
        """
        ignore rotate_half
        """

        """ 1. compute time """ 
        # cos * x 
        cos_q_ops = self.seq_len * self.hidden_size 
        # sin * x 
        sin_q_ops = self.seq_len * self.hidden_zize 
        # add_q
        add_q_ops = self.seq_len + self.hidden_size 
        # total q
        total_rope_q_ops = cos_q_ops + sin_q_ops + add_q_ops 
        # total k
        total_rope_k_ops = total_q_ops 
        # total
        total_ops = total_rope_q_ops + total_rope_k_ops 

        total_compute_time = total_ops / self.compute_ablility
        

        """ 2. read write time """
        q_read_bytes = self.seq_len * self.hidden_size * self.dtype_size 
        cos_read_bypes = q_read_bytes 
        sin_read_bytes = cos_read_bytes 
        q_write_bytes = q_read_bytes 

        k_read_byte = q_read_byte
        k_write_byte = q_write_byte 
        
        total_read_write_bytes = q_read_bytes + q_write_bytes + k_read_bytes + k_write_bytes + cos_read_bytes + sin_read_bytes

        total_read_write_time = total_read_write_bytes / self.bandwidth

        bnt = self.get_bottleneck_time("qkv", total_compute_time, total_read_write_time)

        return bnt

    def Attention(self):
        # 1. attn weight 
        attn_weight_ops = self.heads * self.seq_len * self.head_size * self.seq_len 
        # 2. div sqrt(d)
        scale_ops = self.heads * self.seq_len * self.seq_len 
        # 2. softmax 
        max_ops = self.heads * self.seq_len * self.seq_len 
        # minus max 
        minus_ops = self.heads * self.seq_len * self.seq_len 
        # exp 
        exp_ops = self.heads * self.seq_len * self.seq_len * self.exp_inst 
        # sum 
        sum_ops = self.heads * self.seq_len * self.seq_len
        # div, equal to  * 1/sum(exp(x))
        div_ops = self.heads * self.seq_len * self.seq_len
        # 3 * value 
        value_ops = self.heads * self.seq_len * self.seq_len * self.head_size 
        
        total_ops = attn_weight_ops + scale_ops + max_ops + minus_ops + exp_ops + sum_ops + div_ops + value_ops 
        total_compute_time = total_ops / self.compute_ablility
        # assume use flush attention, only q, k, v, caluse HBM load and store.
        q_read_bytes = self.heads * self.seq_len * self.head_size 
        k_read_bytes = q_read_bytes 
        v_read_bytes = q_read_bytes 
        output_write_bytes = q_read_bytes 

        total_read_write_bytes = q_read_bytes + k_read_bytes + v_read_bytes + output_write_bytes
        total_read_write_time = total_read_write_bytes / self.bandwidth
        bnt = self.get_bottleneck_time("attn", total_compute_time, total_read_write_time)

        return bnt
    
    def OutputMatmul(self):
        total_ops = self.seq_len * self.hidden_size * self.hidden_size 
        total_compute_time = total_ops / self.compute_ablility
        
        # read write 
        input_read = self.seq_len * self.hidden_size * self.dtype_size 
        weight_read = self.hidden_size * self.hidden_size * self.dtype_size 
        output_write = self.seq_len * self.hidden_size * self.dtype_size
        total_read_write_bytes = input_read + weight_read + output_write 
        total_read_write_time = total_read_write_bytes / self.bandwidth
        
        bnt = self.get_bottleneck_time("output matmul", total_compute_time, total_read_write_time)
        return bnt

    
    def MLPUp(self):

        # intermediate size = 4 * self.hidden_size 
        total_ops = self.seq_len * self.hidden_size * self.hidden_size * 4
        total_compute_time = total_ops / self.compute_ablility
        
        # read write 
        input_read = self.seq_len * self.hidden_size * self.dtype_size 
        weight_read = self.hidden_size * self.hidden_size * 4* self.dtype_size 
        output_write = self.seq_len * self.hidden_size * 4

        total_read_write_bytes = input_read + weight_read + output_write 
        total_read_write_time = total_read_write_bytes / self.bandwidth
        
        bnt = self.get_bottleneck_time("MLP up", total_compute_time, total_read_write_time)

        return bnt

    
    def MLPGate(self):

        # intermediate size = 4 * self.hidden_size 
        gate_ops = self.seq_len * self.hidden_size * self.hidden_size * 4
        exp_ops = self.seq_len * self.hidden_size * 4 * self.exp_inst
        add_ops = self.seq_len * self.hidden_size * 4 
        div_ops = self.seq_len * self.hidden_size * 4 
        # mul gate 
        mul_ops = self.seq_len * self.hidden_size * 4

        total_ops = gate_ops + exp_ops + add_ops + div_ops + mul_ops  
        total_compute_time = total_ops / self.compute_ablility
        
        # read write 
        gate_input_read = self.seq_len * self.hidden_size * self.dtype_size 
        weight_read = self.hidden_size * self.hidden_size * 4 * self.dtype_size
        up_input_read = self.seq_len * self.hidden_size * 4 

        output_write = self.seq_len * self.hidden_size * 4
        total_read_write_bytes = gate_input_read + weight_read + up_input_read + output_write

        total_read_write_time = total_read_write_bytes / self.bandwidth
        
        bnt = self.get_bottleneck_time("MLPGate", total_compute_time, total_read_write_time)

        return bnt

    def MLPDown(self):

        # intermediate size = 4 * self.hidden_size 
        total_ops = self.seq_len * self.hidden_size * self.hidden_size * 4
        total_compute_time = total_ops / self.compute_ablility
        
        # read write 
        input_read = self.seq_len * self.hidden_size * 4 * self.dtype_size 
        weight_read = self.hidden_size * self.hidden_size * 4 * self.dtype_size 
        output_write = self.seq_len * self.hidden_size
        total_read_write_bytes = input_read + weight_read + output_write 
        total_read_write_time = total_read_write_bytes / self.bandwidth
    
        bnt = self.get_bottleneck_time("MLPDown", total_compute_time, total_read_write_time)

        return bnt
    
    def MLP(self):
        up_bnt = self.MLPUp()
        gate_bnt = self.MLPGate()
        down_bnt = self.MLPDown()

        return up_bnt + gate_bnt + down_bnt

    def TotalTime(self):

        in_ln_t = self.LlamaRMSNorm()
        qkv_t = self.QKV_MatMul()
        attn_t = self.Attention()
        output_t = self.OutputMatmul()
        post_ln_t = self.LlamaRMSNorm()
        mlp_t = self.MLP()
    
        total_time = in_ln_t + qkv_t + attn_t + output_t + post_ln_t + mlp_t
        print(f"The best total time is {total_time * 1000000:.6f} us , {self.layer_num} layers will use {total_time * self.layer_num * 1000:.6f} ms")
         



if __name__ == "__main__":

    time_model = TimeFlies(model_type="llama2",
                           seq_len = 100,
                           hidden_size = 1024,
                           heads = 16,
                           head_size=64,
                           machine="A10",
                           dtype="float16")
    
    time_model.TotalTime()

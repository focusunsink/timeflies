import time 
from hardware_spec import HardwareSpec
import json 




class TimeFlies:
    
    def __init__(self, 
                 model_type="llama2",
                 seq_len = 100,
                 config_file = "config/Qwen2-VL-2B-Instruct.json",
                 machine="A10",
                 dtype="float16",
                 use_cache=True,
                 attention_version="v1"):
        
        with open(config_file, "r") as file:
            config = json.load(file)
        print(config)
        

        machine = machine.upper()
      
        self.compute_ablility = HardwareSpec[machine]["compute_ablility"][dtype]
        self.bandwidth = HardwareSpec[machine]["bandwidth"]
       

        self.dtype = dtype
        if self.dtype == "int8" or self.dtype == "fp8":
            self.dtype_size = 1
        if self.dtype == "float16":
            self.dtype_size = 2
        if self.dtype == "float32":
            self.dtype_size = 4

        self.seq_len = seq_len 
        self.hidden_size = config["hidden_size"]
        self.heads = config["num_attention_heads"] 
        self.kv_heads = config["num_key_value_heads"]
        self.head_size = self.hidden_size // self.heads
        self.layer_num = config["num_hidden_layers"]
        self.intermediate_size = config["intermediate_size"]

        self.exp_inst = 4
        self.rsqrt_inst = 4

        self.use_cache = use_cache
        self.attn_version = attention_version

        self.print_params()
    
    def print_params(self):
        print("hidden_size is: ", self.hidden_size)
        print("heads is : ", self.heads)
        print("head size is: ", self.head_size)
        print("layer num is: ", self.layer_num)

    def get_real_seq_len(self):
        if self.use_cache:
            return 1 
        else:
            return self.seq_len

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
        seq_len = self.get_real_seq_len()
        if self.use_cache:
            seq_len = 1
        else:
            seq_len = self.seq_len
        # pow
        pow_ops = seq_len * self.hidden_size 
        # mean
        mean_ops = seq_len * self.hidden_size * 2
        # + eps
        eps_ops = seq_len 
        # sqrt
       
        rsqrt_ops = seq_len * self.rsqrt_inst
        # x / rsqrt 
        norm_ops = seq_len * self.hidden_size 
        # * weight
        mul_weight_ops = seq_len * self.hidden_size 
        
        total_compute_ops = pow_ops + mean_ops + eps_ops + rsqrt_ops + norm_ops + mul_weight_ops
        print(total_compute_ops)
        total_compute_time = total_compute_ops / self.compute_ablility
        print(total_compute_time)

        input_bytes = seq_len * self.hidden_size * self.dtype_size
        weight_bytes = self.hidden_size * self.dtype_size 
        output_bytes = seq_len * self.hidden_size * self.dtype_size 

        total_read_write_bytes = input_bytes + weight_bytes + output_bytes 
        total_read_write_time = total_read_write_bytes / self.bandwidth
         
        bnt = self.get_bottleneck_time("layernorm", total_compute_time, total_read_write_time)
        
        return bnt
    
    def QKV_MatMul(self):
        # Matmul input featues = self.hidden_size, output features = self.hidden_size * 3  
        # compute 
        seq_len = self.get_real_seq_len()
        total_ops = seq_len * self.hidden_size * self.hidden_size * 3 
        total_compute_time = total_ops / self.compute_ablility
        
        # read write 
        input_read = seq_len * self.hidden_size * self.dtype_size 
        weight_read = self.hidden_size * self.hidden_size * 3 * self.dtype_size 
        output_write = seq_len * 3 * self.hidden_size 
        total_read_write_bytes = input_read + weight_read + output_write 
        total_read_write_time = total_read_write_bytes / self.bandwidth
    
        bnt = self.get_bottleneck_time("qkv", total_compute_time, total_read_write_time)

        return bnt

    def ROPE(self):
        """
        ignore rotate_half
        """
        seq_len = self.get_real_seq_len()
        """ 1. compute time """ 
        
        # cos * x 
        cos_q_ops = seq_len * self.hidden_size 
        # sin * x 
        sin_q_ops = seq_len * self.hidden_size 
        # add_q
        add_q_ops = seq_len + self.hidden_size 
        # total q
        total_rope_q_ops = cos_q_ops + sin_q_ops + add_q_ops 
        # total k
        total_rope_k_ops = total_rope_q_ops 
        # total
        total_ops = total_rope_q_ops + total_rope_k_ops 

        total_compute_time = total_ops / self.compute_ablility
        

        """ 2. read write time """
        q_read_bytes = seq_len * self.hidden_size * self.dtype_size 
        cos_read_bypes = q_read_bytes 
        sin_read_bytes = cos_read_bypes 
        q_write_bytes = q_read_bytes 

        k_read_bytes = q_read_bytes
        k_write_bytes = q_write_bytes 
        
        total_read_write_bytes = q_read_bytes + q_write_bytes + k_read_bytes + k_write_bytes + cos_read_bypes + sin_read_bytes

        total_read_write_time = total_read_write_bytes / self.bandwidth

        bnt = self.get_bottleneck_time("rope", total_compute_time, total_read_write_time)

        return bnt


    def attention_v1(self):
        pass

    
    def attention_v3(self):
        pass
        

    def Attention(self):
        seq_len = self.get_real_seq_len()
        # 1. attn weight 
        attn_weight_ops = self.heads * seq_len * self.head_size * self.seq_len # q len * (k len + cache_len) 
        # 2. div sqrt(d)
        atten_weight_size = self.heads * seq_len * self.seq_len 
        scale_ops = atten_weight_size
        # 3. softmax 
        max_ops = atten_weight_size
            # minus max 
        minus_ops = atten_weight_size
        # exp 
        exp_ops = atten_weight_size * self.exp_inst 
        # sum 
        sum_ops = atten_weight_size
        # div, equal to  * 1/sum(exp(x))
        div_ops = atten_weight_size
        # 3 * value  attn_weight * v
        value_ops = self.heads * seq_len * self.seq_len * self.head_size 
        
        total_ops = attn_weight_ops + scale_ops + max_ops + minus_ops + exp_ops + sum_ops + div_ops + value_ops 
        total_compute_time = total_ops / self.compute_ablility
        # assume use flush attention, only q, k, v, caluse HBM load and store.
        q_read_bytes = seq_len * self.hidden_size  
        k_read_bytes = self.seq_len * self.hidden_size  
        v_read_bytes = self.seq_len * self.hidden_size   
        output_write_bytes = q_read_bytes 


        total_read_write_bytes = q_read_bytes + k_read_bytes + v_read_bytes + output_write_bytes
        total_read_write_bytes *= self.dtype_size
        total_read_write_time = total_read_write_bytes / self.bandwidth
        bnt = self.get_bottleneck_time("attn", total_compute_time, total_read_write_time)

        return bnt
    
    def OutputMatmul(self):
        seq_len = self.get_real_seq_len()

        total_ops = seq_len * self.hidden_size * self.hidden_size 
        total_compute_time = total_ops / self.compute_ablility
        
        # read write 
        input_read = seq_len * self.hidden_size * self.dtype_size 
        weight_read = self.hidden_size * self.hidden_size * self.dtype_size 
        output_write = seq_len * self.hidden_size * self.dtype_size
        total_read_write_bytes = input_read + weight_read + output_write 
        total_read_write_time = total_read_write_bytes / self.bandwidth
        
        bnt = self.get_bottleneck_time("output matmul", total_compute_time, total_read_write_time)
        return bnt

    
    def MLPUp(self):

        # intermediate size = 4 * self.hidden_size 
        seq_len = self.get_real_seq_len()
        total_ops = seq_len * self.hidden_size * self.intermediate_size
        total_compute_time = total_ops / self.compute_ablility
        
        # read write 
        input_read = seq_len * self.hidden_size
        weight_read = self.hidden_size * self.intermediate_size
        output_write = seq_len * self.intermediate_size

        total_read_write_bytes = input_read + weight_read + output_write 
        total_read_write_bytes *= self.dtype_size
        total_read_write_time = total_read_write_bytes / self.bandwidth
        
        bnt = self.get_bottleneck_time("MLP up", total_compute_time, total_read_write_time)

        return bnt

    
    def MLPGate(self):
        seq_len = self.get_real_seq_len()
        # intermediate size = 4 * self.hidden_size 
        gate_ops = seq_len * self.hidden_size * self.intermediate_size
        exp_ops = seq_len * self.intermediate_size * self.exp_inst
        add_ops = seq_len * self.intermediate_size 
        div_ops = seq_len * self.intermediate_size
        # mul gate 
        mul_ops = seq_len * self.intermediate_size

        total_ops = gate_ops + exp_ops + add_ops + div_ops + mul_ops  
        total_compute_time = total_ops / self.compute_ablility
        
        # read write 
        gate_input_read = seq_len * self.hidden_size * self.dtype_size 
        weight_read = self.hidden_size * self.intermediate_size * self.dtype_size
        up_input_read = seq_len * self.intermediate_size

        output_write = seq_len * self.intermediate_size
        total_read_write_bytes = gate_input_read + weight_read + up_input_read + output_write

        total_read_write_time = total_read_write_bytes / self.bandwidth
        
        bnt = self.get_bottleneck_time("MLPGate", total_compute_time, total_read_write_time)

        return bnt

    def MLPDown(self):
        seq_len = self.get_real_seq_len()
        total_ops = seq_len * self.hidden_size * self.intermediate_size
        total_compute_time = total_ops / self.compute_ablility
        
        # read write 
        input_read = seq_len * self.intermediate_size * self.dtype_size 
        weight_read = self.hidden_size * self.intermediate_size * self.dtype_size 
        output_write = seq_len * self.hidden_size
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
        rope_t = self.ROPE()
        self.attn_t = self.Attention()
        output_t = self.OutputMatmul()
        post_ln_t = self.LlamaRMSNorm()
        mlp_t = self.MLP()

        self.per_layer_total_time = in_ln_t + qkv_t + rope_t + self.attn_t + output_t + post_ln_t + mlp_t
        self.total_time = self.per_layer_total_time * self.layer_num
        print("attention occupied: ", self.attn_t / self.per_layer_total_time * 100, " %")
        print(f"The best total time is { self.per_layer_total_time * 1000000:.6f} us for each layer, {self.layer_num} layers will use {self.total_time * 1000:.6f} ms")
         



if __name__ == "__main__":
    import sys 
    config_path = sys.argv[1]

    time_model = TimeFlies(model_type="llama2",
                           seq_len = 6000,
                           config_file = config_path,
                           machine="H800",
                           dtype="float16",
                           use_cache=True,
                           attention_version="v1")
    
    time_model.TotalTime()

import torch

# 1. 加载模型
# pth_path = "../outputs/videoartgs/sapien/168/final/deform/iteration_best/deform.pth"
pth_path = "../outputs/videoartgs/sapien/1280/init/deform/iteration_1/deform.pth"
state_dict = torch.load(pth_path, map_location='cpu', weights_only=True)

# 2. 构建包含 Size, 乘法算式 和 I/O 维度的树结构
def build_tree_with_io(state_dict):
    tree = {'_size': 0, '_children': {}}
    
    # 填充树
    for key, tensor in state_dict.items():
        parts = key.split('.')
        curr = tree
        for part in parts[:-1]:
            curr = curr['_children'].setdefault(part, {'_size': 0, '_children': {}})
            
        # 提取叶子节点特征
        leaf_name = parts[-1]
        shape = list(tensor.shape)
        numel = tensor.numel()
        
        # 【核心要求 1】：格式化 dim[0] * dim[1] * ... = total_dim
        if len(shape) > 0:
            shape_calc = " * ".join(map(str, shape)) + f" = {numel:,}"
        else:
            shape_calc = f"1 = {numel:,}" # 标量 (Scalar)
            
        # 【核心要求 2】：解析每一层的 Input 和 Output 维度
        io_str = ""
        if len(shape) == 2 and "weight" in leaf_name:
            # PyTorch Linear 层权重维度: [Out, In]
            io_str = f"In: {shape[1]:<4} ➔  Out: {shape[0]:<4}"
        elif len(shape) == 1 and "bias" in leaf_name:
            # Bias 只有输出维度
            io_str = f"Bias       ➔  Out: {shape[0]:<4}"
        elif len(shape) == 1:
            io_str = f"1D Tensor  ➔  Out: {shape[0]:<4}"
        else:
            io_str = f"N-D Shape: {shape}"
            
        # 将解析好的数据存入叶子节点
        curr['_children'][leaf_name] = {
            '_size': numel, 
            '_shape_calc': shape_calc,
            '_io_str': io_str
        }
    
    # 递归计算所有层级的参数和
    def calc_size(node):
        s = 0
        if '_children' in node:
            for k in node['_children']:
                s += calc_size(node['_children'][k])
            node['_size'] = s
        else:
            s = node['_size']
        return s
    
    calc_size(tree)
    return tree

# 3. 递归打印树
def print_tree(node, name, prefix=""):
    connector = "└── " if name != "root" else ""
    
    if '_shape_calc' in node: 
        # 如果是叶子节点（具体的权重张量），打印 I/O 和 乘法维度
        # 为了美观，使用了定宽格式化 (<22 和 <25)
        print(f"{prefix}{connector}{name:<12} | [ {node['_io_str']:<22} ] | 📐 {node['_shape_calc']}")
    else: 
        # 如果是分支节点（模块），打印包含的总参数量
        size_str = f"{node['_size']:,}"
        print(f"{prefix}{connector}{name} [📦 模块总参数: {size_str}]")
        
        if '_children' in node:
            children = node['_children']
            for i, (k, v) in enumerate(children.items()):
                is_last = (i == len(children) - 1)
                print_tree(v, k, prefix + ("    " if is_last else "│   "))

print("\n" + "="*90)
print(" 🌳 1. 神经网络全架构流向树 (Architecture Data Flow Tree)")
print("="*90)
tree_data = build_tree_with_io(state_dict)
print_tree(tree_data, "root")

print("\n" + "="*110)
print(" 📊 2. 全局参数权重清单与数值探针")
print("="*110)
print(f"{'参数名称 (Parameter Name)':<55} | {'Shape (Out, In)':<20} | {'数值探针'}")
print("-" * 110)

total_params = 0
for key, tensor in state_dict.items():
    num_elements = tensor.numel()
    total_params += num_elements
    
    shape_str = str(list(tensor.shape))
    flat_tensor = tensor.flatten()
    snippet = flat_tensor[:min(3, num_elements)].tolist()
    snippet_str = "[" + ", ".join([f"{x:.4f}" for x in snippet]) + ("...]" if num_elements > 3 else "]")
    
    print(f"{key:<55} | {shape_str:<20} | {snippet_str}")

print("-" * 110)
print(f"🧮 物理显存占用评估: 约 {total_params * 4 / (1024**2):.2f} MB (假设 Float32)")
print("="*110)
import torch

# 1. 加载模型权重 (依然使用 CPU 和 weights_only=True 保障安全)
# pth_path = "../outputs/videoartgs/sapien/168/init/deform/iteration_10000/deform.pth"
pth_path = "../outputs/videoartgs/sapien/168/final/deform/iteration_best/deform.pth"
state_dict = torch.load(pth_path, map_location='cpu', weights_only=True)

print("\n" + "="*70)
print(" 🌳 1. 神经网络全架构层级树 (Network Architecture Tree)")
print("="*70)

# 通过解析字典的 Key (如 a.b.c) 来逆向构建网络层级字典树
def build_and_print_tree(keys):
    tree = {}
    for key in keys:
        parts = key.split('.')
        current_level = tree
        for part in parts:
            current_level = current_level.setdefault(part, {})
            
    # 递归打印 ASCII 树
    def render(d, prefix=""):
        for i, (k, v) in enumerate(d.items()):
            is_last = (i == len(d) - 1)
            connector = "└── " if is_last else "├── "
            print(prefix + connector + k)
            render(v, prefix + ("    " if is_last else "│   "))
            
    render(tree)

build_and_print_tree(state_dict.keys())


print("\n" + "="*100)
print(" 📊 2. 全局参数权重清单与数值探针 (All Weights & Value Snippets)")
print("="*100)
# 打印表头
print(f"{'参数名称 (Parameter Name)':<55} | {'维度 (Shape)':<18} | {'数值探针 (First 3 values)'}")
print("-" * 100)

total_params = 0

for key, tensor in state_dict.items():
    # 统计参数总量
    num_elements = tensor.numel()
    total_params += num_elements
    
    # 格式化维度
    shape_str = str(list(tensor.shape))
    
    # 获取数值探针：将张量铺平，截取前 3 个数值保留 4 位小数
    flat_tensor = tensor.flatten()
    snippet = flat_tensor[:min(3, num_elements)].tolist()
    snippet_str = "[" + ", ".join([f"{x:.4f}" for x in snippet])
    snippet_str += ", ...]" if num_elements > 3 else "]"
    
    # 打印每一行
    print(f"{key:<55} | {shape_str:<18} | {snippet_str}")

print("-" * 100)
print(f"✅ 扫描完成！总计包含 {len(state_dict)} 个 Parameter/Buffer。")
print(f"🧮 网络总参数量 (Total Parameters): {total_params:,} 个浮点数")
print("="*100)
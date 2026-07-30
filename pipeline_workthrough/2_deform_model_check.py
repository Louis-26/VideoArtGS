import torch

# 1. Load the model checkpoint
# pth_path = "../outputs/videoartgs/sapien/168/final/deform/iteration_best/deform.pth"
pth_path = "../outputs/videoartgs/sapien/1280/init/deform/iteration_1/deform.pth"

state_dict = torch.load(
    pth_path,
    map_location="cpu",
    weights_only=True,
)


# 2. Build a tree structure containing tensor sizes,
#    dimension multiplication expressions, and input/output dimensions
def build_tree_with_io(state_dict):
    tree = {
        "_size": 0,
        "_children": {},
    }

    # Populate the tree
    for key, tensor in state_dict.items():
        parts = key.split(".")
        current_node = tree

        for part in parts[:-1]:
            current_node = current_node["_children"].setdefault(
                part,
                {
                    "_size": 0,
                    "_children": {},
                },
            )

        # Extract information for the leaf tensor
        leaf_name = parts[-1]
        shape = list(tensor.shape)
        num_elements = tensor.numel()

        # Core requirement 1:
        # Format the dimensions as:
        # dim[0] * dim[1] * ... = total_number_of_elements
        if len(shape) > 0:
            shape_calculation = " * ".join(map(str, shape)) + f" = {num_elements:,}"
        else:
            # Scalar tensor
            shape_calculation = f"1 = {num_elements:,}"

        # Core requirement 2:
        # Infer the input and output dimensions of each tensor
        io_description = ""

        if len(shape) == 2 and "weight" in leaf_name:
            # PyTorch Linear-layer weight shape: [Out, In]
            io_description = f"In: {shape[1]:<4} ➔  Out: {shape[0]:<4}"

        elif len(shape) == 1 and "bias" in leaf_name:
            # A bias tensor only has an output dimension
            io_description = f"Bias       ➔  Out: {shape[0]:<4}"

        elif len(shape) == 1:
            io_description = f"1D Tensor  ➔  Out: {shape[0]:<4}"

        else:
            io_description = f"N-D Shape: {shape}"

        # Store the parsed information in the leaf node
        current_node["_children"][leaf_name] = {
            "_size": num_elements,
            "_shape_calc": shape_calculation,
            "_io_str": io_description,
        }

    # Recursively calculate the total number of parameters
    # contained in each module
    def calculate_size(node):
        total_size = 0

        if "_children" in node:
            for child in node["_children"].values():
                total_size += calculate_size(child)

            node["_size"] = total_size
        else:
            total_size = node["_size"]

        return total_size

    calculate_size(tree)
    return tree


# 3. Recursively print the architecture tree
def print_tree(node, name, prefix=""):
    connector = "└── " if name != "root" else ""

    if "_shape_calc" in node:
        # Leaf node representing a specific parameter tensor:
        # print its I/O dimensions and dimension multiplication expression
        print(
            f"{prefix}{connector}{name:<12} "
            f"| [ {node['_io_str']:<22} ] "
            f"| 📐 {node['_shape_calc']}"
        )

    else:
        # Branch node representing a module:
        # print the total number of parameters contained in the module
        size_string = f"{node['_size']:,}"

        print(
            f"{prefix}{connector}{name} "
            f"[📦 Total Module Parameters: {size_string}]"
        )

        if "_children" in node:
            children = node["_children"]

            for index, (child_name, child_node) in enumerate(children.items()):
                is_last_child = index == len(children) - 1

                print_tree(
                    child_node,
                    child_name,
                    prefix + ("    " if is_last_child else "│   "),
                )


print("\n" + "=" * 90)
print(" 🌳 1. Neural Network Architecture Data Flow Tree")
print("=" * 90)

tree_data = build_tree_with_io(state_dict)
print_tree(tree_data, "root")


print("\n" + "=" * 110)
print(" 📊 2. Complete Parameter List and Numerical Value Probe")
print("=" * 110)

print(
    f"{'Parameter Name':<55} "
    f"| {'Shape (Out, In)':<20} "
    f"| {'Value Probe'}"
)

print("-" * 110)

total_parameters = 0

for key, tensor in state_dict.items():
    num_elements = tensor.numel()
    total_parameters += num_elements

    shape_string = str(list(tensor.shape))

    flattened_tensor = tensor.flatten()
    sample_values = flattened_tensor[: min(3, num_elements)].tolist()

    sample_string = (
        "["
        + ", ".join(f"{value:.4f}" for value in sample_values)
        + ("...]" if num_elements > 3 else "]")
    )

    print(
        f"{key:<55} "
        f"| {shape_string:<20} "
        f"| {sample_string}"
    )

print("-" * 110)

print(
    "🧮 Estimated Parameter Memory Footprint: "
    f"approximately {total_parameters * 4 / (1024**2):.2f} MB "
    "(assuming Float32)"
)

print("=" * 110)
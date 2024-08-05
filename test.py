# 示例列表
example_list = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4]

# 创建空字典
frequency_dict = {}

# 遍历列表并计数
for item in example_list:
    if item in frequency_dict:
        frequency_dict[item] += 1
    else:
        frequency_dict[item] = 1

# 打印结果
print(frequency_dict)

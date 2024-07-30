def count_greater_than(lst, threshold):
    return sum(1 for x in lst if x > threshold)

def count_less_than(lst, threshold):
    return sum(1 for x in lst if x < threshold)

# 測試函數
lst = [1, 3, 5, 7, 9, 2, 4, 6, 8, 10]
threshold = 5

greater_count = count_greater_than(lst, threshold)
less_count = count_less_than(lst, threshold)

print(f"Number of elements greater than {threshold}: {greater_count}")
print(f"Number of elements less than {threshold}: {less_count}")

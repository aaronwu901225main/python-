def count_elements(lst, threshold, comparison):
    if comparison == 'greater':
        return sum(1 for x in lst if x > threshold)
    elif comparison == 'less':
        return sum(1 for x in lst if x < threshold)
    else:
        raise ValueError("Comparison must be 'greater' or 'less'")

# 測試函數
lst = [1, 3, 5, 7, 9, 2, 4, 6, 8, 10]
threshold = 5

comparison = input("Enter 'greater' to count elements greater than the threshold, or 'less' to count elements less than the threshold: ")

count = count_elements(lst, threshold, comparison)

if comparison == 'greater':
    print(f"Number of elements greater than {threshold}: {count}")
elif comparison == 'less':
    print(f"Number of elements less than {threshold}: {count}")

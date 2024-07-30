def find_all_occurrences(s, char):
    return [i for i, c in enumerate(s) if c == char]

# 測試函數
s = "hello world"
char = 'o'
positions = find_all_occurrences(s, char)
print(f"Character '{char}' found at positions: {positions}")

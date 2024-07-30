def minimumDeletions(self, s):
    """
    :type s: str
    :rtype: int
    """
    def find_all_occurrences(s, char):
        return [i for i, c in enumerate(s) if c == char]
        
    def count_elements(lst, threshold, comparison):
        if comparison == 'greater':
            return sum(1 for x in lst if x > threshold)
        elif comparison == 'less':
            return sum(1 for x in lst if x < threshold)
        else:
            raise ValueError("Comparison must be 'greater' or 'less'")

    a = find_all_occurrences(s, 'a')
    b = find_all_occurrences(s, 'b')
    current = len(s)
        
    for i in range(len(s) + 1):
        # a on left side
        a_left_move = count_elements(a, i, 'greater') + count_elements(b, i, 'less')
        # b on left side
        b_left_move = count_elements(b, i, 'greater') + count_elements(a, i, 'less')
            
        current = min(current, a_left_move, b_left_move)
            
    return current
# 测试函数
s = "aabbaababbababaabbbaabbbbaababababbabbbababbabbaabaaabbbbbbaaabbbbabaababbaaabbbbaaabababbbaaa"
print(minimumDeletions(s))  # 输出应该是 40

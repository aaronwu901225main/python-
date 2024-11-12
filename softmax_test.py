import numpy as np

def softmax(x):
    exps = np.exp(x - np.max(x))  # Subtracting max for numerical stability
    return exps / np.sum(exps)

# Modified recursive version
def recursive_softmax(x, depth=0):
    if depth == len(x) - 1:
        return np.array([1 if i == np.argmax(x) else 0 for i in range(len(x))])
    
    temp = softmax(x)
    return recursive_softmax(temp, depth + 1)
    
# Example usage
x = np.array([2.0, 1.0, 0.1])
print("Original softmax:", softmax(x))
print("Recursive max-like softmax:", recursive_softmax(x))

from typing import List, Tuple
import random

def generate_data(
        p:int = 97, 
        train_split:float = 0.8, 
        ops:List[str] = ['+', '-', '/'],
        seed:int = 42) -> Tuple[List[str]]:
    
    random.seed(seed)
    data = []

    for op in ops:
        for a in range(p):
            for b in range(p):
                res = None
                if op=='+':
                    res = (a + b) % p
                elif op == '-':
                    res = (a - b) % p
                elif op == '/' and b != 0:
                    res = a * pow(b, -1, p) % p
                else:
                    continue

                data.append(f'{a} {op} {b} = {res}')

    random.shuffle(data)
    split = int(train_split * len(data))
    train = data[:split]
    test = data[split:]
    return (train, test)




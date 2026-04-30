from typing import List, Tuple
from pathlib import Path
import random

def generate_data(
        p:int = 97, 
        train_split:float = 0.8, 
        val_split:float = 0.20,
        ops:List[str] = ['+', '-', '/'],
        seed:int = 42,
        output_dir:str = None) -> Tuple[List[str]]:
    
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
    train_end = int(train_split * len(data))
    val_end = train_end + int(val_split * len(data))
    train = data[:train_end]
    val = data[train_end:val_end]
    test = data[val_end:]

    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        with open(output_path / "train.txt", "w") as f:
            f.write("\n".join(train))
        with open(output_path / "val.txt", "w") as f:
            f.write("\n".join(val))
        with open(output_path / "test.txt", "w") as f:
            f.write("\n".join(test))

    return (train, val, test)


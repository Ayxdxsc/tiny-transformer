import random, os

random.seed(8888)
ops = ['+', '-', '*']

def ev(e):
    t = []
    i = 0
    while i < len(e):
        if e[i].isdigit():
            j = i
            while j < len(e) and e[j].isdigit():
                j += 1
            t.append(int(e[i:j]))
            i = j
        else:
            t.append(e[i])
            i += 1
    r = t[0]
    s = []
    for k in range(1, len(t), 2):
        op, v = t[k], t[k+1]
        n = r + v if op == '+' else r - v if op == '-' else r * v
        if n < 0 or n > 9999:
            return None, None
        s.append(f'{r}{op}{v}={n}')
        r = n
    return r, ';'.join(s)

def cot(a, b):
    if b < 10: 
        return f'{a}+{b}={a+b}'
    bh, bl = divmod(b, 10)
    s1 = a + bh * 10
    return f'{a}+{bh*10}={s1}' + (f';{s1}+{bl}={s1+bl}' if bl else '')

def scot(a, b):
    if b < 10: 
        return f'{a}-{b}={a-b}'
    bh, bl = divmod(b, 10)
    s1 = a - bh * 10
    return f'{a}-{bh*10}={s1}' + (f';{s1}-{bl}={s1-bl}' if bl else '')

# 建目录
os.makedirs('calc-data', exist_ok=True)

print("正在生成基础数据集...")
with open('calc-data/01_add.txt', 'w') as f: 
    f.write('\n'.join([f'{a}+{b}<a>{cot(a,b)}<eos>' for a in range(100) for b in range(100)] * 5) + '\n')

with open('calc-data/02_sub.txt', 'w') as f: 
    f.write('\n'.join([f'{a}-{b}<a>{scot(a,b)}<eos>' for a in range(100) for b in range(a+1)] * 10) + '\n')

with open('calc-data/03_mul.txt', 'w') as f: 
    f.write('\n'.join([f'{a}*{b}<a>{a}*{b}={a*b}<eos>' for a in range(1,13) for b in range(1,13)] * 100) + '\n')

with open('calc-data/04_pow.txt', 'w') as f: 
    f.write('\n'.join([f'{a}^2<a>{a}*{a}={a*a}<eos>' for a in range(1,21)] * 100) + '\n')

with open('calc-data/05_sqrt.txt', 'w') as f: 
    f.write('\n'.join([f'√{a*a}<a>{a}*{a}={a*a};√{a*a}={a}<eos>' for a in range(1,21)] * 100) + '\n')

with open('calc-data/06_eq.txt', 'w') as f: 
    f.write('\n'.join([f'x+{a}={x+a}<a>x={x+a}-{a}={x}<eos>' for a in range(51) for x in range(100)] * 2 + [f'{a}x={a*x}<a>x={a*x}/{a}={x}<eos>' for a in range(2,10) for x in range(1,13)] * 100) + '\n')

print("正在生成混合运算数据集...")
lines = []
cnt = 0
while cnt < 80000:
    a, b, c = random.randint(0, 30), random.randint(0, 15), random.randint(0, 15)
    o1, o2 = random.choice(ops), random.choice(ops)
    v, cot_str = ev(f'{a}{o1}{b}{o2}{c}')
    if v is None: 
        continue
    lines.append(f'{a}{o1}{b}{o2}{c}<a>{cot_str}<eos>')
    cnt += 1

with open('calc-data/09_mixed.txt', 'w') as f: 
    f.write('\n'.join(lines) + '\n')

print("数据生成完毕！")

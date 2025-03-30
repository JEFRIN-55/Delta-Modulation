# Huffman and Shannon-Fano Coding

## Aim  
To calculate the average codeword length, entropy, efficiency, redundancy, and variance for a given set of probabilities and code lengths using Huffman and Shannon-Fano coding.  

## Tools Required  
- Python  
- NumPy  
- Math Library  

## Program  

### Huffman and Shannon-Fano Coding Calculation  
```python
import numpy as np
import math 

L = 0
hs = 0
p = []
lk = []
n = int(input("Enter the number of Samples: "))

for i in range(n): 
    pr = float(input(f"Enter the probability of sample value {i + 1}: "))  
    p.append(pr)

for j in range(n): 
    l = float(input(f"Enter the length of the sample value {j + 1}: "))  
    lk.append(l)

# Avg length of the codeword
for k in range(n):
    Avg1 = p[k] * lk[k]
    L += Avg1

# Entropy
for k in range(n):
    e = p[k] * math.log(1 / p[k], 2)
    hs += e
hs = round(hs, 3)

# Efficiency
eff = hs / L
eff = round(eff, 3)

# Redundancy 
red = round(1 - eff, 3) 

# Variance
var = 0
for k in range(n):
    var1 = p[k] * (lk[k] - L) ** 2
    var += var1
var = round(var, 3)

print(f"Average Codeword Length is: {L}")
print(f"Entropy is: {hs}")
print(f"Efficiency is: {eff}")
print(f"Redundancy is: {red}")
print(f"Variance is: {var}")
```

## Output Example  
```
Enter the number of Samples : 7
Enter the probability of sample values 1: 0.25
Enter the probability of sample values 2: 0.25
Enter the probability of sample values 3: 0.125
Enter the probability of sample values 4: 0.125
Enter the probability of sample values 5: 0.125
Enter the probability of sample values 6: 0.0625
Enter the probability of sample values 7: 0.0625
Enter the length of the sample values 1: 2
Enter the length of the sample values 2: 2
Enter the length of the sample values 3: 3
Enter the length of the sample values 4: 3
Enter the length of the sample values 5: 3
Enter the length of the sample values 6: 4
Enter the length of the sample values 7: 4
Average Codeword Length is : 2.625
Entropy is : 2.625
Efficiency is : 1.0
Redudancy is : 0.0
Variance is : 0.484
```

## Results  
- The average codeword length was calculated based on the given probabilities and code lengths.  
- The entropy of the source was determined.  
- The efficiency and redundancy of the coding scheme were computed.  
- The variance of the code lengths was analyzed.  

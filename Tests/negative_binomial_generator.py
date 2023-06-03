import numpy as np
from scipy.stats import binom

# Define the parameters for the two binomial distributions
n1, p1 = 10, 0.5  # parameters for first binomial distribution
n2, p2 = 20, 0.7  # parameters for second binomial distribution

# Define the proportion of the mixture
prop = 0.6  # 60% from the first distribution, 40% from the second

# Generate the data
size = 50
data = np.zeros(size, dtype=int)
for i in range(size):
    if np.random.rand() < prop:
        data[i] = binom.rvs(n1, p1)  # draw from the first distribution
    else:
        data[i] = binom.rvs(n2, p2)  # draw from the second distribution

string = "["
for each in data:
    string += str(each) + ", "
string += "]"
print(string)

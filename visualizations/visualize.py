
import numpy as np
import matplotlib.pyplot as plt

losses = np.genfromtxt('losses.csv')


fig = plt.figure()
plt.plot(losses)
#plt.show()
plt.savefig('losses.png', dpi=300, format='png')
import numpy as np
import matplotlib.pyplot as plt

config_name = "H4_rand_1"
config = np.loadtxt("{}.dat".format(config_name))

plt.title(config_name)
plt.scatter(x=config[:, 1], y=config[:, 2])
plt.xlabel("x (Å)")
plt.ylabel("y (Å)")
plt.xlim([-11, 11])
plt.ylim([-11, 11])
plt.gca().set_aspect('equal', adjustable='box')
plt.savefig("config_{}.pdf".format(config_name))
plt.show()

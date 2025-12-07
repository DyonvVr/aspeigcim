import numpy as np

np.set_printoptions(precision=6, suppress=True)

def gen_atom_config_H4_rand():
	orig = np.loadtxt("H4-w0.751-h0.751.dat")

	for i in range(1):
		pert = np.random.rand(12).reshape(4, 3) * 0.6 - 0.3
		# scal = np.random.rand() * 2 + 1.5
		pert[:, 2] = 0  # no z perturbation
		
		new = orig + np.hstack([np.zeros((4, 1)), pert])
		# new[:, 1:] *= scal
		
		np.savetxt("H4-rand-{}.dat".format(i), new)
		
		with open("H4-rand-{}.dat".format(i), 'r') as file :
			filedata = file.read()
		filedata = filedata.replace('1.000000000000000000e+00', '1')
		with open("H4-rand-{}.dat".format(i), 'w') as file:
			file.write(filedata)

def gen_atom_config_H2():
	orig = np.loadtxt("H2-w0.751.dat")
	
	width_factors = [0.25, 0.5, 0.75, 1.25, 1.5, 1.75, 2, 2.5, 3, 3.5, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 30, 50]
	
	for width_factor in width_factors:
		new = np.copy(orig)
		new[:, 1:] *= width_factor
		
		print("factor = {}".format(width_factor))
		print(new)
		
		filename = "H2-w{}.dat".format(0.751 * width_factor)
		np.savetxt(filename, new)
		
		with open(filename, 'r') as file:
			filedata = file.read()
		filedata = filedata.replace('.000000000000000000e+00', '')
		with open(filename, 'w') as file:
			file.write(filedata)

gen_atom_config_H4_rand()

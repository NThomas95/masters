'''
This code will simulate a planar, fully actuated, 2 link robot holding an end effector. We will assume the arm is 
acting in the horizontal plane so that we can neglected gravity. The arm can be modelled by

    [H11 H12][qdd_1] + [C11 C12][qd_1] = [T_1]
    [H21 H22][qdd_2]   [C21 C22][qd_2]   [T_2]

Where T_i are the torques acting on the joints. Please see notes from "Applied Nonlinear Control" chapter 9 to see a 
picture of the arm.
'''


import numpy as np
import matplotlib.pyplot as plt


# Here we define some constants
m1 = 1  # mass of the first arm
l1 = 1  # length of the first arm
me = 2  # mass of the end effector/second arm combo (effective mass)
deltae = np.pi/6  # angle offset of me and second arm
I1 = 0.12  # moment of inertia for first arm
lc1 = 0.5  # length to center of mass of first arm
Ie = 0.25  # effective moment of inertia for second arm/end effector combo
lce = 0.6  # length to effective mass ALONG THE LENGTH OF THE SECOND ARM
Kd = np.eye(2)*100  # derivative gain
Kp = 20*Kd  # proportional gain

# and some more constants. These constants depend on the above constants
# There constants are defined purely to simplify some of the complex expressions
# that are found in the inertial matrix
a1 = I1 + m1*lc1**2 + Ie + me*lce**2 + me*l1**2
a2 = Ie + me*lce**2
a3 = me*l1*lce*np.cos(deltae)
a4 = me*l1*lce*np.sin(deltae)

# how fine do we want our time scales
tf = 3 # final time
iterations = 8000
dt = tf/iterations
t = np.linspace(0, tf, iterations)

# Some arrays all size 2 x iterations. They are zeros at first but each column will be updated at each time step. 
# Further, q and qd, both are 0.
q = np.zeros((2, iterations+1))  # joint positions
qd = np.zeros_like(q)  # joint velocities
qdd = np.zeros_like(q) # joint accelerations
q_error = np.zeros_like(q)  # erro between current position and desired position

# what is the desired postition?
q_desired = np.array([np.pi/3, np.pi/2])


# this is the function that updates the inertial and coriolis matrices
def parameter_updater(position, position_rate, desired_position):
    '''
    The function takes takes the  current state and computes the Inertia and Coriolis Matrices

    param: position: The current joint angles. 2x iterations array
    param: position_rate: The joint velocities. 2x iterations array
    param: desired_position: Where do you want the arm to go. 2x1 array
    
    return: H: The inertia matrix
    return: C: The coriolis matrix
    return: Tau: The joint torques
    return: error: The error between current joint position and the desired position
    '''

    # These are the individual elements in the H matrix
    H11 = a1 + 2*a3*np.cos(position[1]) + 2*a4*np.sin(position[1])
    H12 = a2 + a3*np.cos(position[1]) + a4*np.sin(position[1])
    H21 = H12
    H22 = a2

    # These are the individual elements in the C matrix
    h = a3*np.sin(position[1]) - a4*np.cos(position[1])
    C11 = -h*position_rate[1]
    C12 = -h*(position_rate[0] + position_rate[1])
    C21 = h*position_rate[0]
    C22 = 0
    
    # Construct the arrays
    H = np.array([[H11, H12],
                  [H21, H22]])
    C = np.array([[C11, C12],
                  [C21, C22]])
    error = position - desired_position
    Tau = -(Kp @ error) - (Kd @ position_rate)
    
    return H, C, Tau, error


# this function runs a simulation
def get_acceleration(derivative, inertiaMatrix, coriolisMatrix, inputs):
    '''
    The function takes the current coefficient matrices and torques and state to compute the acceleration.

    param: derivative: current joint velocities
    param: inertiaMatrix: The inertia matrix
    param: coriolisMatrix: The coriolis matrix
    param: inputs: the current joint torques

    return: accel: The current acceleration, qdd
    '''

    # B is just a place holder to make the numpy operation more pretty
    B = inputs - coriolisMatrix @ derivative
    accel = np.linalg.solve(inertiaMatrix, B)
    
    return accel
    

count = 1
for i in range(iterations):
     
     print("The current iteration is: " + str(count), end="\r")

     hessian, coriolis, torques, q_error[:,i] = parameter_updater(q[:,i], qd[:,i], q_desired)  # get the paramters for this instance in time
     qdd[:,i] = get_acceleration(qd[:,i], hessian, coriolis, torques)
     qd[:,i+1] = qdd[:,i]*dt + qd[:,i]
     q[:,i+1] = qd[:,i]*dt + q[:,i]
     
     #if count == 1:
     #    print(qdd)
     count += 1
 

print("Successfully completed " + str(count-1) + " iterations.") 
q = q[:, :-1]  # trim the last element of q. It is one element longer than our time
qd = qd[:,:-1]  # same idea here

# turn the desired angles into a graphable array
g = np.ones_like(q) 
g[0] = g[0]*q_desired[0]
g[1] = g[1]*q_desired[1]

# plot our results
fig, ax = plt.subplots()
ax.plot(t, g[0], linestyle="--", label="desired_1")
ax.plot(t, g[1], linestyle="-.", label="desired_2")
ax.plot(t, q[0], label="Joint 1")
ax.plot(t, q[1], label="Joint 2")

plt.plot
plt.grid()
plt.xlim((0, tf))
plt.legend()
plt.ylabel("joint positions")
plt.xlabel("time")
plt.show()



















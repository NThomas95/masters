'''

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
tf = 5 # final time
iterations = 100
dt = tf/iterations
t = np.linspace(0, tf, iterations)

# Some initial conditions
q = np.zeros((2, iterations+1))
qd = np.zeros_like(q)
qdd = np.zeros_like(q)
q_error = np.zeros_like(q)

# what is the desired postition?
q_desired = np.array([np.pi/3, np.pi/2])


# this is the function that updates the inertial and coriolis matrices
def parameter_updater(position, position_rate, desired_position):
    H11 = a1 + 2*a3*np.cos(position[1]) + 2*a4*np.sin(position[1])
    H12 = a2 + a3*np.cos(position[1]) + a4*np.sin(position[1])
    H21 = H12
    H22 = a2
    h = a3*np.sin(position[1]) - a4*np.cos(position[1])
    C11 = -h*position_rate[1]
    C12 = -h*(position_rate[0] + position_rate[1])
    C21 = h*position_rate[0]
    C22 = 0
    
    H = np.array([[H11, H12],
                  [H21, H22]])
    C = np.array([[C11, C12],
                  [C21, C22]])
    error = position - desired_position
    #print(error)
    Tau = -(Kp @ error) - (Kd @ position_rate)
    print(Tau)
    

    return H, C, Tau, error


# this function runs a simulation
def get_acceleration(derivative, inertiaMatrix, coriolisMatrix, inputs):

    B = inputs - coriolisMatrix @ derivative
    print(B)
    accel = np.linalg.solve(inertiaMatrix, B)
    
    return accel
    

count = 1
for i in range(iterations):
     
     #print("The current iteration is: " + str(count), end="\r")

     hessian, coriolis, torques, q_error[:,i] = parameter_updater(q[:,i], qd[:,i], q_desired)  # get the paramters for this instance in time
     qdd[:,i] = get_acceleration(qd[:,i], hessian, coriolis, torques)
     qd[:,i+1] = qdd[:,i]*dt + qd[:,i]
     q[:,i+1] = qd[:,i]*dt + q[:,i]
     
     #if count == 1:
     #    print(qdd)
     count += 1
     
q = q[:, :-1]
#print(np.shape(q))
plt.plot(t, q[0])
plt.show()



















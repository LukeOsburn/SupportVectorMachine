import numpy as np
import matplotlib.pyplot as plt
np.random.seed(6)
import math

from sklearn.datasets.samples_generator import make_blobs

#Creating a data set, its the same data each time for the given inputs
(X,y) =  make_blobs(n_samples=45,n_features=2,centers=2,cluster_std=1.05,random_state=9
)

#we are effectively trying to find the equation of a plane
#what we really want are the two lines that lie on the plane for when z = -1 and -1
#we want this plane and these lines such that the perpindicular distance between the lines is maximum
#and no data lies in this region
#and that the lines perfectly segregate the data

#we need to add 1 to X values for the constant value in the equation
X1 = np.c_[np.ones((X.shape[0])),X]

#lets create a dictionary with our positive and negative values
#the positive and negative is simply a "class"
#we want all the positive values to be on one side of our segregating lines
#and the negative values on the other
postiveX=[]
negativeX=[]
for i,v in enumerate(y): #this includes a counter in the for line!
    if v==0:
        negativeX.append(X[i])
    else:
        postiveX.append(X[i])

data_dict = {-1:np.array(negativeX), 1:np.array(postiveX)}

print("this is our data")
print(data_dict)


w=[] #weights 2 dimensional vector
b=[] #constant term, for our plane, this will be the z-axis intercept

max_feature_value=float('-inf')
min_feature_value=float('+inf')


#lets find our maximum and minimum feature values
#this will help limit our search for our optimal solution
for yi in data_dict:
    if np.amax(data_dict[yi])>max_feature_value:
        max_feature_value=np.amax(data_dict[yi])
    if np.amin(data_dict[yi])<min_feature_value:
        min_feature_value=np.amin(data_dict[yi])

#the range from which we will look for our coffecients
range=1
#the size of the learning steps we take
lrate=0.04

#for our equation we have w1x+w2y+z=b (x,y and z are variables)
#so we want w1, w2 and b, these are constants (for planes, one of the variables won't need a coefficient)
#we will iterate through these until we find the plane with the best solution

def SVM_Training(data_dict):
    i=1
    global w
    global b

    length_Wvector = {}
    #w_optimum = max_feature_value
    #this will limit our search for w1 and w2
    #we will search for w1 and w2 between  w_optimum*(-1) and w_optimum


    #LETS ITERATE!
    w = np.array([1,1]) #arbitary starting values
    for ww in np.arange(-range, range, lrate):
          for www in np.arange(-range, range, lrate):
            w = np.array([ww,www]) #starting values
            for b in np.arange(-range*5, range*5, lrate*2):
                    correctly_classified = True
                    # every data point needs to be correct
                    for yi in data_dict:
                        for xi in data_dict[yi]:
                            if yi*(np.dot(w,xi)+b) < 1:  # we want  yi*(np.dot(w_t,xi)+b) >= 1 for correct classification
                                correctly_classified = False
                                break
                                break

                    #if all classifications are correct, lets calculate the perpindicular distance
                    if correctly_classified:
                        length_Wvector[np.linalg.norm(w)] = [w,b] #store w1 and w2 and b for maximum perpindicular distance
                        #the maximum perpindicular distance will be when the unit vector (w1,w2) is smallest
                        #its not necessarly to calculate the maximum width, just pick smallest unit vector (w1,w2)




    norms = sorted([n for n in length_Wvector])
    #for key,val in length_Wvector.items():
        #print key, "=>", val
    #this is our list of acceptable planes that divide our data correctly
    #the best one is with the largest perp distance which will have the smallest (w1,w2) unit vector
    #LETS PICK IT!


    if len(norms)==0:
        print("Potentially no solution")
        print("If you can see a solution try decreasing the learning rate")
        colors = {1:'r',-1:'b'}
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        plt.scatter(X1[:,1],X1[:,2],marker='o',c=y)
        plt.show()

        quit()

    minimum_wlength = length_Wvector[norms[0]]
    #This will print our possible planes with their different w and b values
    #for key,val in length_Wvector.items():
        #print key, "=>", val
    w = minimum_wlength[0]
    b = minimum_wlength[1]

SVM_Training(data_dict)

#we now have our equation, lets print it
print(w)
print(b)

w0=round(w[0], 3)
w1=(round(w[1], 3))
b=(round(b, 3))


print("The equation of our plane is: %sx+%sy+z=%s" % (w0,w1,b))
print("lets get our support vectors")
print("our support vectors we get from our plane equation for z=1 and z=-1, lets also plot the middle, z=0")

x=np.linspace(-20,20,100)

#b=-1
#positive support vector
z=1
y1=(-b+z-w0*x)/(w1)

#negative support vector
z=-1
negy1=(-b+z-w0*x)/(w1)

#this is the line from the plane for z=0
z=0
y0=(-b+z-w0*x)/(w1)

colors = {1:'r',-1:'b'}
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
plt.scatter(X1[:,1],X1[:,2],marker='o',c=y)
ax.plot(x,negy1,'-g',label='z=-1')
ax.plot(x,y0,'b--',label='z=0')
ax.plot(x,y1,'-k',label='z=1')
legend = ax.legend(fontsize='x-large')
#legend.get_frame().set_facecolor('C0')
# Put a nicer background color on the legend.

plt.axis([-20,20,-20,20])
plt.xlabel('x values')
plt.ylabel('y values')
plt.title("Equation of our plane: %sx+%sy+%s=z" % (w0,w1,b))
plt.grid(True)
plt.show()

import pandas as pd
import numpy as np
import csv as cs
import matplotlib.pyplot as plt
import math
import os
from sklearn import linear_model

# importing the data 
file = open("Boston_Housing_Data.csv", newline='')
reader = cs.reader(file)
model_order = 13 # Polynomial order + Intersect
# treat the first line as a header
header = next(reader)

#Initialize the lists containing the data
data_crim = []; data_medv = [];ln_data_medv = []
data_zn = []; data_indus =[]; data_nox =[]; data_rm =[]
data_age =[]; data_dis =[]; data_rad =[]; data_tax =[]
data_pt_ratio =[]; data_b =[]; data_lstat =[]; data_list = []


# convert the strings in the csv to floats
for row in reader:
    crim, zn, indus = float(row[0]), float(row[1]), float(row[2])
    nox, rm, age = float(row[3]), float(row[4]), float(row[5])
    dis, rad, tax = float(row[6]), float(row[7]), float(row[8])
    pt_ratio, b, lstat = float(row[9]), float(row[10]), float(row[11])
    medv = float(row[12])
    ln_medv = [math.log(medv)]
    step = [crim, zn, indus, nox, rm, age, dis, rad, tax, pt_ratio, b, lstat]

# append the values into the lists
    data_crim.append(crim); data_medv.append(medv); ln_data_medv.append(ln_medv);
    data_zn.append(zn); data_indus.append(indus); data_nox.append(nox);
    data_rm.append(rm); data_age.append(age); data_dis.append(dis);
    data_rad.append(rad); data_tax.append(tax); data_pt_ratio.append(pt_ratio)
    data_b.append(b); data_lstat.append(lstat)
    data_list.append(step);
    
#the data matrix
data_matrix = np.array(data_list)
data_matrix = np.hstack((np.ones((data_matrix.shape[0], 1)), data_matrix))
   
# split the data into trainng and testing sets    
splt_f = 0.90; splt = int(splt_f * data_matrix.shape[0]) 
x_train = data_matrix[:splt]; x_test = data_matrix[splt:, :-1] 
x_test = data_matrix[splt:] 

#The target variable
Ln_Y = np.array(ln_data_medv).reshape((-1,1))
y_train = Ln_Y[:splt, -1].reshape((-1, 1)) 
y_test = Ln_Y[splt:, -1].reshape((-1, 1))

def get_gradient(w, x, y):
    y_estimate = x.dot(w).flatten()
    error = (y.flatten() - y_estimate)
    mse = (1.0/len(x))*np.sum(np.power(error, 2))
    gradient = -(1.0/len(x)) * error.dot(x)
    return gradient, mse

error_list = []
iteration = []
w = np.random.randn(model_order)
alpha = 0.000001
tolerance = 1e-7

# Perform Stochastic Gradient Descent
epochs = 1
decay = 0.95
batch_size = 36
iterations = 0
while True:
    order = np.random.permutation(len(x_train))
    x_train = x_train[order]
    y_train = y_train[order]
    b=0
    while b < len(x_train):
        tx = x_train[b : b+batch_size]
        ty = y_train[b : b+batch_size]
        gradient = get_gradient(w, tx, ty)[0]
        error = get_gradient(w, x_train, y_train)[1]
        w -= alpha * gradient
        iterations += 1
        b += batch_size
        error_list.append(error)
    
    # Keep track of our performance
    if epochs%100==0:
        new_error = get_gradient(w, x_train, y_train)[1]
        iteration.append(epochs)
        print("Epoch: %d - Error: %.4f" %(epochs, new_error))
        
    
        # Stopping Condition
        if abs(new_error - error) < tolerance:
            print("Converged.")
            break
        
    epochs += 1
    alpha = alpha * (decay ** int(epochs/10000))


#TAX
tax_plot = plt.scatter(data_tax, data_medv)
rad_plot = plt.scatter(data_rad, data_medv)
pt_plot = plt.scatter(data_pt_ratio, data_medv)
b_plot = plt.scatter(data_b, data_medv)
lstat_plot = plt.scatter(data_lstat, data_medv)
dis_plot = plt.scatter(data_dis, data_medv)
age_plot = plt.scatter(data_age, data_medv)
rm_plot = plt.scatter(data_rm, data_medv)
indus_plot = plt.scatter(data_indus, data_medv)
nox_plot = plt.scatter(data_nox, data_medv)
crim_plot = plt.scatter(data_crim, data_medv)

plt.legend(handles=(tax_plot,rad_plot,pt_plot,b_plot, lstat_plot, dis_plot, age_plot, rm_plot, indus_plot, nox_plot, crim_plot),
           labels=('TAX', 'RAD', 'PT Ratio','B', 'L. Stat', "DIS",'AGE', 'RM','INDUS', 'NOX', 'CRIM' ),
           title="Features", title_fontsize=12,
           scatterpoints=1,
           bbox_to_anchor=(1, 1), loc=2, borderaxespad=1.,
           ncol=1,
           fontsize=10)



plt.xlabel("Features")
plt.ylabel("Median Cost of Housing")
plt.xlim(0, 500)
plt.ylim(0, 60)

dis_plot = plt.scatter([i for i in range(len(error_list))], error_list)
plt.xlabel("Number of iterations")
plt.ylabel("Error")
plt.xlim(0, 11000)
plt.ylim(0, 11000)

#Indus
indus_plot = plt.scatter(data_indus, ln_data_medv)
plt.xlabel("Polution-NOX")
plt.ylabel("Log_e(MEDV)")
plt.xlim(0, 30)
plt.ylim(0, 60)

#rm
rm_plot = plt.scatter(data_rm, ln_data_medv)
plt.xlabel("RM")
plt.ylabel("Log_e(MEDV)")
plt.xlim(0, 10)
plt.ylim(0, 60)

#Age
age_plot = plt.scatter(data_age, ln_data_medv)
plt.xlabel("AGE")
plt.ylabel("Log_e(MEDV)")
plt.xlim(0, 110)
plt.ylim(0, 60)

#dis
dis_plot = plt.scatter(data_dis, ln_data_medv)
plt.xlabel("DIS")
plt.ylabel("Log_e(MEDV)")
plt.xlim(0, 12)
plt.ylim(0, 60)

#rad
dis_plot = plt.scatter(data_rad, ln_data_medv)
plt.xlabel("RAD")
plt.ylabel("Log_e(MEDV)")
plt.xlim(0, 25)
plt.ylim(0, 60)

#TAX
dis_plot = plt.scatter(data_tax, ln_data_medv)
plt.xlabel("DIS")
plt.ylabel("Log_e(MEDV)")
plt.xlim(0, 750)
plt.ylim(0, 60)

#pt_ratio
dis_plot = plt.scatter(data_pt_ratio, ln_data_medv)
plt.xlabel("PT RATIO")
plt.ylabel("Log_e(MEDV)")
plt.xlim(0, 25)
plt.ylim(0, 60)

#B
dis_plot = plt.scatter(data_b, ln_data_medv)
plt.xlabel("B")
plt.ylabel("Log_e(MEDV)")
plt.xlim(0, 400)
plt.ylim(0, 60)

#LSTAT
dis_plot = plt.scatter(data_lstat, ln_data_medv)
plt.xlabel("LSTAT")
plt.ylabel("Log_e(MEDV)")
plt.xlim(0, 50)
plt.ylim(0, 60)




























































































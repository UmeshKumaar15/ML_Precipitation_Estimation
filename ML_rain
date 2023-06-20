#Umesh Kumaar
#Importing all libraries.
#Tested accuracy = 89.86% with respect to data

import sys
sys.path.append('c:\\users\\umesh\\appdata\\local\\programs\\python\\python310\\lib\\site-packages')

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tkinter import *
from tkinter import Tk, Button
from tkinter import filedialog
from tkinter.ttk import *
from sklearn.metrics import mean_squared_error
import csv
master = Tk()

#Declaring the factors of rain in variables array.
variables = ['Maximum Temperature', 'Minimum Temperature', 'Total Snow',
             'Sun Hours', 'UV Index Max', 'UV Index Min','Moon Illumination',
             'Dew Point', 'Feels Like', 'Heat Index', 'Wind Chill',
             'Wind Gust', 'Cloud Cover', 'Humidity', 'Pressure', 'Temperature', 'Visibility', 'Wind Direction', 'Wind Speed']

#Declaring empty array with these 19 factors to get store later.
data_array = np.empty((1, 19))

#Importing the dataset
dataset = pd.read_csv('C:\\Users\\umesh\\Documents\\Bangalore codes\\machine_learning\\Banglore_Rain_Data.csv')
X = dataset.iloc[:,1:-1].values  #X is training set.
y = dataset['precipMM']          #y is test set.

#Formatting the imported dataset by handling missing data.
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:, 5:7])
X[:, 5:7] = imputer.transform(X[:, 5:7])

#Spliting dataset into training set and test set in ratio of (9:1).
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 0)

#Performing feature scaling on the dataset.
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Training the model.
from sklearn.linear_model import LinearRegression
regressor= LinearRegression()
regressor.fit(X_train, y_train)

#Predicitng the values.
y_pred=regressor.predict(X_test)
xpred = regressor.predict(X_train)

#Plotting graph for training set.
def trainingset():
  plt.figure(1)
  X_index=np.arange(0,len(X_train),1)
  plt.scatter(X_index, y_train, color = 'red')
  plt.title("Rain precipitation Level (Training Set)")
  plt.xlabel("Days")
  plt.ylabel("Rain Precipitation (in ml)")
  plt.show()

#Plotting graph for test set.
def testset():
  X_index=np.arange(0,len(X_test),1)
  plt.scatter(X_index, y_test, color = 'red')
  plt.title("Rain Precipitation Levels (Test set)")
  plt.xlabel("Days")
  plt.ylabel("Rain Precipitation (in ml)")
  plt.show()

#Analysis of the factors of rain and plotting graph w.r.t their data.
def analysis(chfactor, label):
        X_index = np.arange(0, len(X_test) + len(X_train),1)
        factor_values = dataset[chfactor].values[:len(X_index)]
        plt.scatter(X_index[:len(factor_values)], factor_values, color = 'r')
        plt.title(label)
        plt.xlabel("Days")
        plt.ylabel(label)
        plt.show()

#Compute function collects input and with the collected data it fits the linear regression and prints right prediction with predicted precipitation values.
def compute():
    values = []
    for entry in entry_widgets:
        value = float(entry.get())
        values.append(value)
    #zinp is test array.
    # zinp = np.array([[33, 22,0, 11.6 ,7, 2, 79 , 18, 31,  33, 22 ,37, 94, 70, 1010, 32, 10, 269, 20]])
    # zinp = zinp.reshape(1,-1)
    new_row = np.array([values])
    global data_array
    data_array = np.vstack((data_array, new_row))
    data_array = data_array[1:]
    # print(data_array)
    clf = LinearRegression()
    clf.fit(X,y)
    data_array = data_array.reshape(1,-1)
    zpred = clf.predict(data_array)
    if(zpred>1):
      output = ("There is high chance of rainfall with precipitation of " + str(zpred) + " (in mm).")
    elif(zpred > 0.2):
      output = ("Rainfall is expected with precipitation of " + str(zpred) + " (in mm).")
    elif(zpred >0):
      output = ("There is very slight chance of rainfall with precipitation of " + str(zpred) + " (in mm).")
    else:
      output = ("Rainfall is unlikely " + str(zpred))
    myLabel = Label(Displayframe, text = output)
    myLabel.pack()

def clear():
      for entry in entry_widgets:
            entry.delete(0, END)

#Declaring center frame.
centerframe = Frame(master)
centerframe.pack(pady=50)
  
#Displaying Input fields.
global entry_widgets
entry_widgets = []
for i, variable in enumerate(variables):
    label = Label(centerframe, text=f"Enter the {variable}:")
    label.grid(row=i, column=0, sticky="E")
    entry = Entry(centerframe)
    entry.grid(row=i, column=1, padx=10, pady=5)
    entry_widgets.append(entry)

# Compute and clear button to gather data and clear
compute_button = Button(centerframe, text="Compute", command=compute)
compute_button.grid(row=len(variables), column=0, columnspan=2, pady =10)

clear_button = Button(centerframe, text="Clear", command=clear)
clear_button.grid(row=len(variables), column=1, columnspan=2, pady=10)

#Help Window
def help_window():
      helpwin = Toplevel(master)
      helpwin.title("Help")
      helpwin.geometry("500x300")
      Label(helpwin, text= info).pack()

def accuracy():
    mse = mean_squared_error(y_test, y_pred)
    accuracy = 1 - mse
    accuracy = "{:.4%}".format(accuracy)
    acc_label = Label(Displayframe, text= "Accuracy is " + accuracy)
    acc_label.pack()

#Text for help window.
global info
info = ('''Rain Prediction:
                          1.Enter the parameters in the given input boxes.
                          2.Press "Compute" button to get the prediction.
                          3.To check with another values, press "Clear" to clear all values.
           
        Analysis:
                    1.Press "Analysis" on the menu bar.
                    2.Select the factor to which u want to see analysis
                    3.Displayed Analysis is the plotted graph of choosen factors 
                      recorded from past weather station at Bengaluru.
                    ''')

#Buttons for printing training and test set graphs
botframe = Frame(master)
botframe.pack(side=BOTTOM)
tr_button = Button(botframe, text="See Training Set data representation", command= trainingset )
tr_button.pack( side = LEFT,padx=10, pady=10) 
ts_button = Button(botframe, text="See Test Set data representation", command= testset )
ts_button.pack(side = RIGHT, padx=10, pady=10) 

#Creating Menu
my_menu = Menu(master)
master.config(menu = my_menu)

#File Menu
file_menu = Menu(my_menu,tearoff=False)
my_menu.add_cascade(label="Home",menu=file_menu)
file_menu.add_command(label = "Check Accuracy",command=accuracy)
file_menu.add_command(label = "Exit",command=exit)

#Analysis Menu
analysis_menu = Menu(my_menu, tearoff=False)
my_menu.add_cascade(label="Analysis", menu= analysis_menu)
analysis_menu.add_command(label="Rain", command= lambda:analysis('precipMM', 'Precipitation'))
analysis_menu.add_command(label="Humidity", command = lambda:analysis('humidity', 'Humidity'))
analysis_menu.add_command(label="Temperature", command= lambda:analysis('tempC', 'Temperature'))
analysis_menu.add_command(label="Pressure", command= lambda:analysis('pressure', 'Pressure'))
analysis_menu.add_command(label="Dew Point", command= lambda:analysis('DewPointC', 'Dew Point'))
analysis_menu.add_command(label="Visibility", command=lambda: analysis('visibility', 'Visibility'))
analysis_menu.add_command(label="Heat Index", command= lambda: analysis('HeatIndexC', 'Heat Index'))

#Help Menu.
help_menu = Menu(my_menu, tearoff=False)
my_menu.add_cascade(label="Help", menu=help_menu)
help_menu.add_command(label="Info", command=help_window)

Displayframe = LabelFrame(master, text="Output Page :")
Displayframe.place(anchor= E,relx=0.6,rely=0.9)

master.title("Analysis and Prediction of Rain")
master.geometry("800x800")
master.iconbitmap("C:\\Users\\umesh\\Downloads\\MLrainLogo.ico")
master.mainloop()
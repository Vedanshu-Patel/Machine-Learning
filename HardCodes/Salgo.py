# Vedanshu Patel
# 20BCE0865
import numpy as np
#Enter the number of rows
n=int(input("Enter the number of rows "))
#Enter the number of columns
m=int(input("Enter the number of columns "))
arr=[]
y=[]
#Enter the features
print("Enter the features")
for i in range(n):
    x=[]
    for j in range(m):
        w=input("enter the input in column "+ str(j)+ " ")
        x.append(w)
    arr.append(x)
#printing the features
for i in range(n):
    for j in range(m):
        print(arr[i][j], end=" ")
    print(" ")
#Making the hypothesis arraay
for z in range(m):
        s=arr[0][z]
        y.append(s)

#Implementing the algorithm for find s
for h in range(n):
    for l in range(m):
        if((y[l]==arr[h][l] or y[l]=='?') and arr[h][m-1]=="Yes"):
            continue
        elif((y[l]!=arr[h][l] and y[l]!='?') and arr[h][m-1]=="Yes"):
            y[l]='?'
        elif(arr[h][m-1]=="No"):
            continue
#Printing the final answer
print("The hypothesis is ")
for g in range(m-1):
    print(y[g], end=" ")
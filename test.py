# import time					
# tic = time.perf_counter() # Start Time
# x = 0
# while x < 60000:
#     x += 1
# print(x)		  # Your code here
# toc = time.perf_counter() # End Time
# # Print the Difference Minutes and Seconds
# print(f"Build finished in {(toc - tic)/60:0.0f} minutes {(toc - tic)%60:0.0f} seconds")
# # For additional Precision
# print(f"Build finished in {toc - tic:0.4f} seconds")
# print ("time.time(): %f " %  time.time())
# print (time.localtime( time.time() ))
# print (time.asctime( time.localtime(time.time()) ))

# class Cat:
#     def __init__(self,color) -> None:
#         self.color = color
#     def eat(self):
#         print("chi---")
#     def printInfo(self):
#         print(self.color)
# xiaoming = Cat("白")
# xiaoming.eat()
# xiaoming.printInfo()
# xiaoming.color = "黑"
# xiaoming.printInfo()

#super可以让子类访问父类属性
# class Parents:
#     def __init__(self,age) -> None:
#         self.age = age
#         print("这是属性")
#     def fun(self):
#         print("这是方法")
#     def printInfo(self):
#         print(self.age)
# class Child(Parents):
#     def __init__(self,age) -> None:
#         super(Child, self).__init__(age)
#         print("实例化执行")
# test = Child(3)
# test.fun()
# test.age
# test.age = 6
# test.printInfo()
import os
import shutil
import numpy as np
# new_path ="/_videoSlope/"
# path = "bird/"
# num = 0
# for files in os.listdir(path):
#     if files.endswith("jpg"):
#         shutil.copy(os.path.join(path, files), new_path)
#         num += 1
#     else:
#         continue
# print("共复制%s张图"%num)
a = np.array([[1,2,1],[2,1,1],[1,1,3]],[[1,2,1],[2,1,1],[1,1,3]])
print(a)
b = np.zeros_like(a)
tmp = list(set(a.flatten().tolist()))
for num in tmp:
    if num != 1 :
        b[a==num]=num
print(tmp)
print(len(a[0,:]))
print(b)

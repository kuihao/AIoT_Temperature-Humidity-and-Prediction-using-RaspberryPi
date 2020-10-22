# 2020/10/22
1. **分析差異** 多元線性回歸公式解 與 SGD (Stochastic gradient descent, 隨機梯度下降法)
2. python 矩陣乘法
# 多元線性回歸
* https://github.com/TommyHuang821/Deep-Learning-API-example/blob/master/TensorFlow/TF%20%E7%AF%84%E4%BE%8B%20II%20(Regression).md
# Matrix Multiplication in Python
1. *numpy.dot()* ~= *numpy.matmul()* = *@*
    * *numpy.dot()* 是向量內積 (inner product)
    * 內積知識補充: https://ccjou.wordpress.com/2010/01/27/%E5%85%A7%E7%A9%8D%E7%9A%84%E5%AE%9A%E7%BE%A9/
    * *numpy.matmul()* 則是正常的矩陣乘法，運算子為 *@*
    * 這兩個函式的運算結果看來無異，且 python 對此兩種函式都支援矩陣自動轉置
      就算矩陣相乘的行與列沒有對應到，函式內部也會自動轉置
2. *numpy.multply()* 
    * 單純是矩陣內相對應的元素 (Entry) 互相乘積，小心別誤用
    * 可使用運算子 \*
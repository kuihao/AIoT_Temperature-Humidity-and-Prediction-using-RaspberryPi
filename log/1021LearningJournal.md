# 2020/10/21
* **學習繪製 plot 視覺化判斷目前參數及函數模型的好壞**
    * 練習的程式碼：https://github.com/kuihao/AIoT_Temperature-Humidity-and-Prediction-using-RaspberryPi/blob/master/testing/test_plot.py
    * **等值線 (Contour line)**
        * **plt.contourf(*args, data=None, **kwargs)[source]**
        * contour([X, Y,] Z, [levels], **kwargs)
        * X, Yarray-like, optional
            * The coordinates of the values in Z.
        * **Z array-like(N, M)**
            * **The height values over which the contour is drawn.**
        * **levelsint (欲等分切割成幾個區域)** or array-like, optional
            * **Determines the number and positions of the contour lines / regions.**
* 顯然 Linear Regression的方式是針對所有輸入的訓練資料，假設一個Linear Function之後，直接找尋一個最貼合資料預測結果的參數組合，這個參數的調整方法其實像是試錯，但是針對窮舉所有參數是不可能算完的，因此有Gradient descent及輔助Gradient descent的Optimizer可以提升找尋預測Error最小參數的效率

* 矩陣與矩陣相乘
    * 使用 **ndarray** 的 *.dot()* 方法或者 *np.dot()*
* 單位矩陣(Identity Matrix)
    * **NumPy** 建立單位矩陣的方法是 *np.eye( 維度, dtype = '型別' )*
* *np.astype* 內容轉型別參數
    * Data type  Description
    * bool_  Boolean (True or False) stored as a byte
    * int_  Default integer type (same as C long; normally either int64 or int32)
    * intc  Identical to C int (normally int32 or int64)
    * intp  Integer used for indexing (same as C ssize_t; normally either int32 or int64)
    * int8  Byte (-128 to 127)
    * int16  Integer (-32768 to 32767)
    * int32  Integer (-2147483648 to 2147483647)
    * int64  Integer (-9223372036854775808 to 9223372036854775807)
    * uint8  Unsigned integer (0 to 255)
    * uint16 Unsigned integer (0 to 65535)
    * uint32 Unsigned integer (0 to 4294967295)
    * uint64 Unsigned integer (0 to 18446744073709551615)
    * float_ Shorthand for float64.
    * float16 Half precision float: sign bit, 5 bits exponent, 10 bits mantissa
    * float32 Single precision float: sign bit, 8 bits exponent, 23 bits mantissa
    * float64 Double precision float: sign bit, 11 bits exponent, 52 bits mantissa
    * complex_  Shorthand for complex128.
    * complex64  Complex number, represented by two 32-bit floats (real and imaginary components)
    * complex128 Complex number, represented by two 64-bit floats (real and imaginary components)
# 2020/10/18 處理缺失資料
* 使用Python Modules: **sklearn.impute**
* sklearn v0.20更新之後移除了*sklearn.preprocessing.Imputer* 並以sklearn.impute.SimpleImputer取代，並額外提供另外兩種缺值提補方法，一共三種:
    * *sklearn.impute.SimpleImputer*
       * https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html
    * *sklearn.impute.MissingIndicato*
        * https://scikit-learn.org/stable/modules/generated/sklearn.impute.MissingIndicator.html
        * https://scikit-learn.org/stable/modules/impute.html#impute
    * *sklearn.impute.KNNImputer*
        * https://scikit-learn.org/stable/modules/generated/sklearn.impute.KNNImputer.html
* 中文解說：https://tsinghua-gongjing.github.io/posts/sklearn_imputation.html
* 補值的意義: 當蒐集的資料有缺失時有兩個做法，方法一是直接把缺失資料全部捨棄，但當數據缺稀或比例資料不均時，直接捨棄資料是相當浪費且失準的；因此方法二是有邏輯地填補缺值並填補值不會過度影響全部數據的分布。
# 2020/10/18 資料分類
* 原始資料是有序離散值的話 => *sklearn.preprocessing* 的 *Label Encoding*
* 原始資料是無序離散值的話 => *sklearn.preprocessing* 的 *One Hot Encoding* 或 *Pandas* 的 *(Dummies)*
* 用意：將名義尺度、類別尺度資料進行編碼動作，方便進行數據分析
* 複習統計學四大尺度
    * **名義尺度**(nominal scale)，又稱為**類別尺度**(categorical scale)
        * **編碼純粹為識別分類之用**，次序或數值大小均無意義
        * 使用敘述統計分析
        * 例如：性別分男女、婚姻狀態、信仰、職業、電話區碼 
    * **順序尺度**(ordinal scale)
        * 分類具有「程度」、「級別」、「優劣」等
        * **編碼順序具有「光譜、量級的差異」意義**
        * 常見為非數值型態，亦可為數值型態但四則運算的解釋性較低，常見使用敘述統計分析
        * 例如：滿意度量表、行為意圖量表
    * **等距尺度**(interval scale)
        * 資料只為**數值**型態，具有**次序意義**且**數值差異的四則運算有意義**
        * 零只是數值
        * 例如：溫度
    * **等比尺度**(ratio scale)
        * 具有等距尺度的性質，且**數值的比值具有意義**，就是有倍率關係
        * 零表示「無」的意思
        * 例如：金額、長度、重量、
    * 前二者常為屬性資料(qualitative data)、後二者常為計量資料(quantitative data)
* Reference
    * Linear Regression Homework: https://colab.research.google.com/drive/131sSqmrmWXfjFZ3jWSELl8cm0Ox5ah3C
    * Example: https://www.cnblogs.com/HL-space/p/10676637.html
    * Data Preprocession: https://medium.com/@doremi31618/%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92%E7%AD%86%E8%A8%98-%E6%95%B8%E6%93%9A%E9%A0%90%E8%99%95%E7%90%8601-ae90853978da
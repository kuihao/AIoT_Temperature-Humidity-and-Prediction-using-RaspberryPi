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

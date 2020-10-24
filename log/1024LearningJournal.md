# 2020/10/23
## DataFrame 亂序排序方式(打散資料)
1. **sklearn 的 shuffle 物件**
    引入：from sklearn.utils import shuffle
    函式：df = shuffle(df)
2. **pandas 的 sample**
    函式：df.sample(frac=1)
    說明：frac = 1 表示全部返還，fact = 小數則返還百分比數量；亦可使用 n=行數
## Debug: python unicode 路徑解碼
* python 需要在路徑'\'前加上 r 使直譯器可以識別為路徑符號而非跳脫字元

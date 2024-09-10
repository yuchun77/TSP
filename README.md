# TSP

這是我在中興大學演算法課程中的一項作業，這項作業使用了兩種方法解決了 Traveling Salesman Problem (TSP) 的問題，分別是 Dynamic Programming (DP) 方法和 Genetic Algorithm (GA) 的方法。

這份作業中，我比較了兩個演算法的效能：

![efficacy](https://github.com/user-attachments/assets/3d7b90f0-6f18-4f68-9ed6-32b83c52c127)
* 左圖中比較了兩種演算法隨著頂點數增加，所花時間的變化
  * 在頂點數多的情況下(>=15)，DP 的方法明顯比 GA 所花的時間更久
* 由於 GA 的方法並非 100% 可以找到最短路徑，在右圖中，我計算了 GA 的 error rate
  * 公式： $\frac{|\Sigma W_{GA} - \Sigma W_{DP}|}{\Sigma W_{DP}}$
  * 頂點數越多 GA 的 error 越大

詳細的 code 都在 `TSP.py` 中！

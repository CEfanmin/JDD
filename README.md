# JDD2017数据开发竞赛

## 主要思路为

	1、按照传统的特征工程->调参->模型融合
	2、利用basic seq2seq时序模型
    3、利用CNN进行8（R）、9（G）、10（B）多层特征提取，形成高维特征,然后利用XGBoost或者RandomForest进行回归

## 1、传统思路
### 特征工程

	（1）Data Exploration

在这一步要做的基本就是 EDA (Exploratory Data Analysis)，也就是对数据进行探索性的分析，从而为之后的处理和建模提供必要的结论。
用 pandas 来载入数据，并做一些简单的可视化来理解数据.

	（2）Visualization
通常来说 matplotlib 和 seaborn 提供的绘图功能就可以满足需求了。

比较常用的图表有：

查看目标变量的分布。当分布不平衡时，根据评分标准和具体模型的使用不同，可能会严重影响性能。

对 Numerical Variable，可以用 Box Plot 来直观地查看它的分布。

对于坐标类数据，可以用 Scatter Plot 来查看它们的分布趋势和是否有离群点的存在。

对于分类问题，将数据根据 Label 的不同着不同的颜色绘制出来，这对 Feature 的构造很有帮助。

绘制变量之间两两的分布和相关度图表。

    目前对于时序数据的处理采用了package的方式，将8-11月的数据进行sum()，然后利用"loan_price_sum/plannum=repayment_ability", "loan_price_sum/loan_count_sum=loan_ability","order_price_sum/order_count_sum=order_ability","click_count_sum/4=activity_ability"作为新的属性值。特征工程有待进一步丰富。

### 调参

	偷懒利用了TPOT进行调参，得出单模型还是使用XGBoost又快又好。

### 模型融合
	未完待续


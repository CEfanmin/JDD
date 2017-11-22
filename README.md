# JDD2017数据开发竞赛

**主要思路为：**

    1、按照传统的特征工程->调参->模型融合
	2、利用basic seq2seq时序模型

# 特征工程

	目前对于时序数据的处理采用了package的方式，将8-11月的数据进行sum()，然后利用"loan_price_sum/plannum=repayment_ability", "loan_price_sum/loan_count_sum=loan_ability","order_price_sum/order_count_sum=order_ability","click_count_sum/4=activity_ability"作为新的属性值。特征工程有待进一步丰富。

# 调参

	偷懒利用了TPOT进行调参，得出单模型还是使用XGBoost又快又好。

# 模型融合
	未完待续


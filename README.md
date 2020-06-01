# 贝叶斯估计
- **贝叶斯公式**：  
    $$P\left(B_i|A\right)=\frac{P\left(B_i\right)P\left(A|B_i\right)}{\sum_{j=1}^{n}P\left(B_j\right)P\left(A|B_j\right)}$$
+ **动态停等策略**  
  &emsp;&emsp;在传统的方法中，所有trial使用固定的数据长度。分类的结果取决于CSP滤波器提取到的特征在Fisher线性模型上的分布。根据以上，基于Bayes的DS策略，试图在给定数据长度和CSP特征的情况下估计正确预测的概率，并作为分类结果置信度的估计值。   
  &emsp;&emsp;具体方法如下：   
  &emsp;&emsp;根据正确预测和错误预测将所有训练试次的CSP特征在Fisher线性模型上的投影分别分为正确预测组和错误预测组，并构建似然概率密度函数。
  ``` matlab
    % matlab
    % 构建概率密度函数采用高斯核密度估计
    function [x,y] = get_distribution(points)
        % points 为 csp特征 在线性模型上的分布
        % fi 为 x 点的概率密度
        [fi,x] = ksdensity(points);
    end
  ```
    &emsp;&emsp;在模拟在线的过程中，对每一个新的数据片段，首先计算CSP特征并在Fisher线性模型上投影并得到预测结果，然后根据贝叶斯推论估计其被正确预测的后验概率: 
    $$P\left(H_1|csp,t\right)=\frac{P\left(csp|H_1,t\right)P\left(H_1|t\right)}{P\left(csp|H_1,t\right)P\left(H_1|t\right)+P\left(csp|H_0,t\right)P\left(H_0|t\right)}$$
    &emsp;&emsp;其中：$H_1$表示成功预测，$H_0$表示错误预测  
    &emsp;&emsp;&emsp;&emsp;&emsp;$P\left(H_i|t\right)$表示使用 $t$ 时间数据，正确/错误预测的概率  
    &emsp;&emsp;&emsp;&emsp;&emsp;$P\left(csp|H_i,t\right)$表示 $t$ 时间数据提取出的CSP特征正确/错误预测的概率
- **整体流程**
    1. 训练概率模型
  ```flow
  st=>start: 调试数据
  op=>operation: CSP+LDA得到投影点
  op2=>operation: 分别对h1\h2核密度估计
  op3=>operation: 得到h1\h2对应的核密度估计
  st->op->op2->op3
  ```
  预测
  ``` flow
  st2=>start: 初始化概率
  op2_1=>operation: 输入新的数据片段
  op2_2=>operation: 得到CSP在LDA上的投影点
  op2_3=>operation: 计算后验概率 
  cond1=>condition: 后验概率是否够大
  e=>end: 输出结果
  st2->op2_1->op2_2->op2_3->cond1
  cond1(no)->op2_1
  cond1(yes)->e
  ```
- **Code Review**
  

- **附录：**  
  1. matlab：：ksdensity 函数示例如下：
  ```matlab
  % matlab
  points = [1,2,3,4,5,2,3,4,5,6,2,3,4,5,1];
  [fi,x] = ksdensity(points);
  plot(x,fi);
  ```
<div align=center>
    <img src=".\ksdensity函数示例.jpg" width="400"/><br>
    ksdensity 函数示例
</div>
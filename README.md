# light_field_low_rank_optimization
Some tricks on light field decomposition with low rank and SVD method.  Please click Code_Usage_Instructions.pdf to read more details and ideas.`下载pycharm, 在windows下运行，所有代码中均已经配好路径，并附带数据集，直接打开对应的.py文件运行。`

如果X是一个m行n列的数值矩阵，rank(X)是X的秩，假如rank (X)远小于m和n，则我们称X是低秩矩阵。低秩矩阵每行或每列都可以用其他的行或列线性表出，可见它包含大量的冗余信息。利用这种冗余信息，可以对缺失数据进行恢复，也可以对数据进行特征提取.
PCA方法鲁棒性不佳是由于矩阵的噪声并不完全是高斯噪声。对应到视频序列中就是，长时间的静止视频中每帧的图片相关性极高，而在有物体运动时，往往是部分像素有极大的变化，但是变化的像素较少。这也就是说，视频中图像可以分成相关性极高的背景以及少量像素的前景图像。即低秩部分以及稀疏部分。这是RASL优于PCA的部分，即前景图像对矩阵的分解影响较小，如下图对镜头中的动目标检测。虽然论文中提到了，矩阵不是特别稀疏时也有很好的性能。


<img src="https://github.com/liangjiubujiu/light_field_low_rank_optimization/blob/master/images/1.jpg" width="600"  />

除此之外低秩矩阵还应用于鲁棒联合图像对齐，Peng等人利用低秩矩阵分解的原理设计出了一种基于低秩稀疏分解的鲁棒联合图像对齐方法(Robust alignment by sparse and low rank decomposition，RASL)。用于对齐一组本质上线性相关的图像。因为只有一组线性相关的图像在构成矩阵后，构造矩阵才具有低秩的特性。此方法的主要原理就是在低秩矩阵分解的过程中加入图像变换参数，使得经过变换后的图像构成的矩阵具有低秩结构，并通过低秩分解将其分解为低秩矩阵部分和稀疏矩阵部分，其中低秩矩阵部分代表对齐且去除噪声后的图像。 这种鲁棒联合图像对齐的方法，不仅仅能实现图像的自动对齐，而且在对齐图像 的同时能去除噪声等因素对图像的影响，如下图。
<img src="https://github.com/liangjiubujiu/light_field_low_rank_optimization/blob/master/images/2.jpg" width="600"  />

## 文章算法基本原理与流程
对图像进行对齐的基本思路是先寻找一变换因子。对于光场子图像，考虑子图像之间的差异较小，暂时考虑使用相似变换，即仅有旋转和平移变换。仿射变换包含了对图像进行的旋转、平移和缩放三种变换，每张自图像提取三个特征点即可使用仿射变换对子图像进行对齐。
寻找到变换因子后对图像实行此变换，经过变换后的图像构成的矩阵具有低秩结构，将变换后的矩阵通过分解，得到低秩矩阵部分和稀疏误差部分，这样不仅能对原图像实现姿态的纠偏， 而且能去除图像的噪声污染。显然，稀疏误差部分不仅包含噪声，还包含原图像上的高频信息，对于光场子图像或者视频中相邻帧图像，误差矩阵包含了移动物体的轮廓信息。用简单的流程图表示：

<img src="https://github.com/liangjiubujiu/light_field_low_rank_optimization/blob/master/images/3.jpg" height="600" width="300" />

## 实现效果
光场图像压缩和编解码主要包括五个步骤：光场子图像提取、子图像批量对齐、低秩矩阵分解、编码传输、解码和恢复光场图像。

### 1.	光场子图像提取
使用matlab 的工具包LFToolbox0.4对光场图像进行处理,得到一组光场图的子孔径图像，设一共有N张。（为了简化运算，取N=18）
每张子孔径图上取两个特征点，求解初始化的N个相似矩阵（仿射矩阵则需要使用三个点）。
求解相似矩阵是为了对后续ADMM（交叉方向乘子）进行初始化，由于仅在初始化的操作中使用，因此对特征点的精度要求不是过分严格，这里使用手动标定的方法。
具体步骤

* 1.	将N张光场子孔径图像导入opencv +python 3.6 中进行显示，使用matplotlib 包读取图像上的特征点，手动确定两个特征在每张图中坐标位置。
* 2.	将得到数据导入matlab，批量生成.mat矩阵，存放每张图上的两个特征点的坐标位置。
* 3.	将N个.mat特征点坐标位置数据和对应的N张子孔径图像打包。放在cvv\rasl_LF\data\data_test\Image

改进为`此处特征点的匹配和标定也可以使用opencv 包中的函数实现角点检测，提取两个特征点。但是，经过人工检查，效果并不好，两幅图中的特征点对应错误，使用角点检测后，有一对特征点没有正确匹配。`

<img src="https://github.com/liangjiubujiu/light_field_low_rank_optimization/blob/master/images/45.jpg" width="500" />

此处可以换用sift方法自动提取多个匹配的特征点。Sift方法使用`cvv\sift路径中的feature_points_main.py`即可观察到下图的效果,因此，使用sift方法正确匹配，可以简化数据集的创建步骤，但论文暂未使用该方法。

<img src="https://github.com/liangjiubujiu/light_field_low_rank_optimization/blob/master/images/6.jpg" width="500" />

### 2.	RASL 用于子图像批量对齐
<img src="https://github.com/liangjiubujiu/light_field_low_rank_optimization/blob/master/images/7.jpg" width="800" />

算法伪码：

 <img src="https://github.com/liangjiubujiu/light_field_low_rank_optimization/blob/master/images/8.jpg" width="500" />
 
使用python3 运行cvv\rasl_LF\main中的budahan.py即可观察到下图。经过2次迭代，得到对齐及矩阵分解的结果

<img src="https://github.com/liangjiubujiu/light_field_low_rank_optimization/blob/master/images/9.jpg" width="500" />

使用matlab经过9外层迭代和418次内层循环，得到15张子图像的秩收敛到r=8

<img src="https://github.com/liangjiubujiu/light_field_low_rank_optimization/blob/master/images/9.jpg" width="500" />
SVD低秩矩阵分解改进`即使可以把一个很大的矩阵分解成两个长条，但是SVD不够好，因为得到的分解结果矩阵不具有稀疏的特性。为进一步得到更为稀疏的分解形式，这里将对齐后的图片矩阵使用CUR分解。CUR分解，适合大数据；有编码、计算优势，不方便呈现。` 

<img src="https://github.com/liangjiubujiu/light_field_low_rank_optimization/blob/master/images/12.jpg" width="500" />

但无法用理论或者实验证实CUR的性能优于SVD。

<img src="https://github.com/liangjiubujiu/light_field_low_rank_optimization/blob/master/images/11.jpg" width="500" />

运行`cvv\CUR路径中的CUR.py`，可观察到分解效果，其中cvv\CUR文件1.xlsx是对齐后的图片拉成列向量后的矩阵。

# Personal-Instance-Segmentation-Paper-Record
# Under construction!
# Table of Contents
- [Deep Learning Methods](#deep-learning-methods)

## Deep Learning Methods

### **MNC ★★**
**[Paper]**  Instance-aware Semantic Segmentation via Multi-task Network Cascades<Br>
**[Year]** CVPR 2016 Oral<Br>
**[Authors]** 	[Jifeng Dai](http://www.jifengdai.org/), [Kaiming He](http://kaiminghe.com/),	[Jian Sun](http://www.jiansun.org/)  <Br>
**[Pages]**  https://github.com/daijifeng001/MNC <Br>
**[Description]** <Br>
1) 模型包含三个网络: differentiating instances, estimating masks, categorizing objects. 代码之后可以详细研究一下. <Br>
2) 区分实例, 即得到每个instance的不分类别的bounding box, 类似RPN, <Br>
3) 预测mask, 即得出每个bbox中的二值mask. bbox经过ROI Warp到固定尺寸, 用2个全连接层完成每个像素的二值分类. <Br>
4) 分类, 即根据bbox和mask得到当前instance的类别. 此处对于输入特征考虑了两种选择, 一是直接将bbox的特征作为输入, 二是将bbox的特征与mask做点乘, 只输出mask部分的特征其余位置置零. <Br>
5) 设计了5层的级联网络, 上述的三个步骤即为stage 1, 2, 3, 接下来以前面得到的class和bbox组成proposal, 再次进行mask预测和分类, 即重复stage 2和3. <Br>

### **InstanceFCN ★★**
**[Paper]**  Instace-sensitive Fully Convolutional Networks <Br>
**[Year]** ECCV 2016 <Br>
**[Authors]** 	[Jifeng Dai](http://www.jifengdai.org/), [Kaiming He](http://kaiminghe.com/),	[Yi Li](https://liyi14.github.io/), [Shaoqing Ren](http://shaoqingren.com/), [Jian Sun](http://www.jiansun.org/) <Br> 
**[Pages]**  <Br>
**[Description]** <Br>
1) DL用于Instance-segmentation较早的一篇paper. 在FCN的基础上提出instance-sensitive的InstanceFCN, 通过将每个pixel相对于某instance的relative position进行assemble, 得到output instance candidate.
2) **Instance-sensitive score maps**: 将FCN的"each output pixel is a classifier of an object category"改造成"each output pixel is a classifier of relative positions of instances". **Instance assembling module**: 将每个sliding window划分成k x k的网格, 对应于k^2个relative position. 相同网格中的像素取对应于相同位置的score map进行assemble.
3) **优点:** InstanceFCN具有local coherence的优点, 且没有任何high-dimensional layer. **缺点:**  inference时将输入进行多尺度缩放来处理multi-scale问题, 感觉有点简单粗暴; 模型的输出只能分辨每个instance, 但不能得出每个instance的类别.
4) 没有找到开源代码, 对training和inference的具体实现没仔细研究.

### **FCIS**
**[Paper]**   Fully Convolutional Instance-aware Semantic Segmentation <Br>
**[Year]** CVPR 2017 Spotlight <Br>
**[Authors]** 	[Yi Li](https://liyi14.github.io/), 	[Haozhi Qi](http://haozhi.io/), Xiangyang Ji, [Yichen Wei](https://www.microsoft.com/en-us/research/people/yichenw/)  <Br>
**[Pages]**  https://github.com/daijifeng001/R-FCN  <Br>
**[Description]** <Br>

### **FastMask ★**
**[Paper]** FastMask: Segment Multi-scale Object Candidates in One Shot <Br>
**[Year]** CVPR 2017 Spotlight  <Br>
**[Authors]** [Hexiang Hu](http://hexianghu.com/), [Shiyi Lan](https://voidrank.github.io/), Yuning Jiang, Zhimin Cao, [Fei Sha](http://www-bcf.usc.edu/~feisha/) <Br>
**[Pages]**  https://github.com/voidrank/FastMask <Br>
**[Description]**<Br>
1) 粗读. 提出了一个body, neck, head的one-shot模型. 
2) body net部分进行特征提取. 提取到的特征组成多尺度的特征金字塔, 分别送入共享参数的neck module提取multi-scale特征, neck module为residual neck. 得到的特征图进行降维后提取dense sliding window, sliding windows经batch normalization后送入head module, head module为attention head 
3) neck module部分以2为步长对feature map进行下采样, 可能导致尺度过于稀疏. 因此提出two-stream FastMask architecture, 使scale更密集.

### **DWT★**
**[Paper]**  Deep Watershed Transformation for Instance Segmentation <Br>
**[Year]** CVPR 2017  <Br>
**[Authors]** [Min Bai](http://www.cs.toronto.edu/~mbai/), [Raquel Urtasun](http://www.cs.toronto.edu/~urtasun/) <Br>
**[Pages]**  https://github.com/min2209/dwt <Br>
**[Description]** <Br>
1) 用分水岭的思想做instance segmentation，分Direction Network和Watershed Transform Network两个阶段
2) Direction Network：计算pixel到最近边界的距离变换的单位梯度
3) Watershed Transform Network：以前一阶段得到的梯度为输入，计算16个Bin能量的概率，Bin 0表示边界部分的能量
4) 边界部分有相同的能量，避免over segmentation问题
5) 实验部分，计算confidence score没看懂，可以再研究一下
6) 能量Bin的划分，每个能量在算loss权重部分没看懂


### **BAIS★**
**[Paper]**  Boundary-aware Instance Segmentation<Br>
**[Year]** CVPR 2017 <Br>
**[Authors]** [Zeeshan Hayder](https://scholar.google.com.au/citations?user=K2INPyYAAAAJ&hl<Br>=en), [Xuming He](https://xmhe.bitbucket.io/), [Mathieu Salzmann](http://ttic.uchicago.edu/~salzmann/) <Br>
**[Pages]**  https://vitalab.github.io/deep-learning/2017/08/22/boundary-aware.html<Br>
**[Description]**<Br>
1) 提出一种基于距离变换的instance segmentation方法, 可以克服bounding box不准确的问题. 包括三部分: 提取bounding box, 预测object mask (OMN), object分类, 整个网络都是可微的, 可端到端训练;
2) OMN基于目标内像素到其边界的距离变换, 设计网络得到K个boundary-aware object mask, 然后decode成完整的object mask;
3) 仿照MNC, 采用multi-stage策略: 根据上一阶段得到的object mask, 对bounding box进行refine;
4) 实验及一些具体实现没研究, 如object mask与bounding box feature是如何结合起来的等;
	


# Personal-Instance-Segmentation-Paper-Record
# Under construction!
# Table of Contents
- [Deep Learning Methods](#deep-learning-methods)

# Rank
- Deep Learning Methods <Br>
  - ★★★ <Br>
  **[Mask R-CNN]** <Br>
  - ★★ <Br>
  **[MNC]**, **[InstanceFCN]**, **[Dynamically Instantiated Network]**, **[FCIS]**, **[PANet]**<Br>
  - ★ <Br>
  **[MPA]**, **[DWT]**, **[BAIS]**, **[MaskLab]**, **[InstanceCut]** <Br>

# Deep Learning Methods

### **SDS**
**[Paper]**  Simultaneous Detection and Segmentation <Br>
**[Year]** ECCV 2014 <Br>
**[Authors]** 	[Bharath Hariharan](http://home.bharathh.info/), [Pablo Arbelaez](https://biomedicalcomputervision.uniandes.edu.co/),	[Ross Girshick](http://www.rossgirshick.info/), [Jitendra Malik](https://people.eecs.berkeley.edu/~malik/) <Br> 
**[Pages]** https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/shape/sds/  <Br>
**[Description]** <Br>

### **Hypercolumns**
**[Paper]**  Hypercolumns for Object Segmentation and Fine-grained Localization <Br>
**[Year]** CVPR 2015 Oral<Br>
**[Authors]** 	[Bharath Hariharan](http://home.bharathh.info/), [Pablo Arbelaez](https://biomedicalcomputervision.uniandes.edu.co/),	[Ross Girshick](http://www.rossgirshick.info/) <Br> 
**[Pages]** https://github.com/bharath272/sds  <Br>
**[Description]** <Br>

### **CFM**
**[Paper]**  Convolutional Feature Masking for Joint Object and Stuff Segmentation <Br>
**[Year]** CVPR 2015 <Br>
**[Authors]** 	[Jifeng Dai](http://www.jifengdai.org/), [Kaiming He](http://kaiminghe.com/),	[Jian Sun](http://www.jiansun.org/)  <Br>
**[Pages]**   <Br>
**[Description]** <Br>
	

### ***Monocular Object Instance Segmentation and Depth Ordering with CNNs***
**[Paper]**  Monocular Object Instance Segmentation and Depth Ordering with CNNs <Br>
**[Year]** ICCV 2015 <Br>
**[Authors]** 	[Ziyu Zhang](https://ziyu-zhang.github.io/), [Alexander Schwing](http://alexander-schwing.de/), [Sanja Fidler](http://www.cs.utoronto.ca/~fidler/), [Raquel Urtasun](http://www.cs.toronto.edu/~urtasun/) <Br>
**[Pages]**   <Br>
**[Description]** <Br>
	
### **DeepMask**
**[Paper]**  Learning to segment object candidates <Br>
**[Year]** NIPS 2015 <Br>
**[Authors]** 	[Pedro O. Pinheiro](http://www.pedro.opinheiro.com/), [Tsung-Yi Lin](https://scholar.google.de/citations?user=_BPdgV0AAAAJ&hl=en&oi=sra), [Ronan Collobert](https://scholar.google.de/citations?user=32w7x1cAAAAJ&hl=en&oi=sra), [Piotr Dollàr](https://pdollar.github.io/) <Br>
**[Pages]**   <Br>
**[Description]** <Br>

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
	
### **SharpMask**
**[Paper]**  ILearning to Refine Object Segments <Br>
**[Year]** ECCV 2016 Spotlight <Br>
**[Authors]** [Pedro O. Pinheiro](http://www.pedro.opinheiro.com/), [Tsung-Yi Lin](https://scholar.google.de/citations?user=_BPdgV0AAAAJ&hl=en&oi=sra), [Ronan Collobert](https://scholar.google.de/citations?user=32w7x1cAAAAJ&hl=en&oi=sra), [Piotr Dollàr](https://pdollar.github.io/) <Br> 
**[Pages]**  <Br>
**[Description]** <Br>

### **InstanceFCN ★★**
**[Paper]**  Instace-sensitive Fully Convolutional Networks <Br>
**[Year]** ECCV 2016 <Br>
**[Authors]** 	[Jifeng Dai](http://www.jifengdai.org/), [Kaiming He](http://kaiminghe.com/),	[Yi Li](https://liyi14.github.io/), [Shaoqing Ren](http://shaoqingren.com/), [Jian Sun](http://www.jiansun.org/) <Br> 
**[Pages]**  <Br>
**[Description]** <Br>
1) DL用于Instance-segmentation较早的一篇paper. 在FCN的基础上提出instance-sensitive的InstanceFCN, 通过将每个pixel相对于某instance的relative position进行assemble, 得到output instance candidate.
2) **Instance-sensitive score maps**: 将FCN的"each output pixel is a classifier of an object category"改造成"each output pixel is a classifier of relative positions of instances". **Instance assembling module**: 将每个sliding window划分成k x k的网格, 对应于k^2个relative position. 相同网格中的像素取对应于相同位置的score map进行assemble.
3) **优点:** InstanceFCN具有local coherence的优点, 且没有任何high-dimensional layer. **缺点:**  inference时将输入进行多尺度缩放来处理multi-scale问题, 感觉有点简单粗暴; 模型的输出只能分辨每个instance mask, 但不能得出每个instance的类别.
4) 没有找到开源代码, 对training和inference的具体实现没仔细研究.

### **MPA ★**
**[Paper]**  Multi-scale Patch Aggregation (MPA) for Simultaneous Detection and Segmentation <Br>
**[Year]** CVPR 2016 Oral <Br>
**[Authors]** 	[Shu Liu](http://shuliu.me/), [Xiaojuan Qi](http://kaiminghe.com/),	[Jianping Shi](http://shijianping.me/), Hong Zhang, [Jiaya Jia](http://www.cse.cuhk.edu.hk/leojia/) <Br> 
**[Pages]**  <Br>
**[Description]** <Br>
1) 粗读, 提出了一种基于patch的instance segmentation方法, 其中patch对应的是目标的一部分, 而不是整个目标. <Br>
2) 经过若干层特征提取后(VGG16), 在feature map上提取四个尺度的patch, 类似于ROI pooling那一套, 再将patch align到相同尺寸, 分别送入分类和分割两支, 得到label和segmentation mask. patch的真值是根据一系列规则确定的. <Br>
3) 得到patch的label和mask后, 对相同尺度的patch在水平和竖直方向进行aggregate, 聚合相同label的patch的mask. <Br>
4) 几点疑问: 密集取patch送入后面的两个网络是否会造成inference速度很慢? 四个尺度的path且没经过坐标修正, 鲁棒性够强吗? mask预测一支对于部分目标能很好地分割吗, 会不会存在混淆(比如patch里包括的是两个人的衣服, 能否准确把其中的一件衣服分为前景)

### ***Reversible Recursive Instance-Level Object Segmentation***
**[Paper]**  Reversible Recursive Instance-Level Object Segmentation <Br>
**[Year]** CVPR 2016  <Br>
**[Authors]** [Xiaodan Liang](http://www.cs.cmu.edu/~xiaodan1/), [Yunchao Wei](https://weiyc.github.io/), [Xiaohui Shen](http://users.eecs.northwestern.edu/~xsh835/), [Zequn Jie](http://jiezequn.me/), [Jiashi Feng](https://sites.google.com/site/jshfeng/), Liang Lin, [Shuicheng Yan](https://www.ece.nus.edu.sg/stfpage/eleyans/) <Br> 
**[Pages]**  <Br>
**[Description]** <Br

### **MultiPathNet**
**[Paper]**  A MultiPath Network for Object Detection <Br>
**[Year]** BMVC 2016 <Br>
**[Authors]** 	[Sergey Zagoruyko](http://imagine.enpc.fr/~zagoruys/), [Adam Lerer](http://www.pedro.opinheiro.com/),	[Tsung-Yi Lin](https://vision.cornell.edu/se3/people/tsung-yi-lin/), [Pedro O. Pinheiro](http://www.pedro.opinheiro.com/) <Br> 
**[Pages]** https://github.com/facebookresearch/multipathnet  <Br>
**[Description]** <Br>

### ***Dynamically Instantiated Network* ★★**
**[Paper]**  Pixelwise Instance Segmentation with a Dynamically Instantiated Network <Br>
**[Year]** CVPR 2017 <Br>
**[Authors]**  [Anurag Arnab](http://www.robots.ox.ac.uk/~aarnab/), [Philip Torr](http://www.robots.ox.ac.uk/~phst/)<Br>
**[Pages]** http://www.robots.ox.ac.uk/~aarnab/instances_dynamic_network.html  <Br>
**[Description]** <Br>
1) 粗读. 本篇paper的思路比较新奇, 在全图上做segmentation和detection, 再用instance CRF区分instance, 整个模型可用类似CRFasRNN的思路端到端完成. <Br>
2) semantic segmentation部分用semantic CRF. <Br>
3) Instance CRF的unary potential包括三部分, box term, global term和shape term. 其中box term和global term是根据detection和segmentation共同确定的, shape term是根据预先定义的shape exemplar决定的. Pairwise term是一densely-connected Gaussian potentials. <Br>
4) loss采用cross entropy loss. 这里groundtruth和prediction的对应是根据最大IoU得到的. <Br>

### **FCIS ★★**
**[Paper]**   Fully Convolutional Instance-aware Semantic Segmentation <Br>
**[Year]** CVPR 2017 Spotlight <Br>
**[Authors]** 	[Yi Li](https://liyi14.github.io/), 	[Haozhi Qi](https://xjqi.github.io/), [Jifeng Dai](http://www.jifengdai.org/), Xiangyang Ji, [Yichen Wei](https://www.microsoft.com/en-us/research/people/yichenw/)  <Br>
**[Pages]** https://github.com/msracver/FCIS  <Br>
**[Description]** <Br>
1) 基于InstanceFCN中position sensitive score map的概念, 提出了end to end的可区分类别的实例分割方法. <Br>
2) backbone为resnet-101,从conv4开始分为两支, RPN一支产生ROI, 另一支产生2K^2(C+1)个位置敏感score map. 之后对每个ROI进行根据K*K个相对位置进行assemble, 每类输出ROI inside 和ROI outside两个score map. 根据inside和outside score map的大小组合可以得到一个pixel的两个信息: 1.它是否位于某个目标的相应位置上; 2.它是否属于该目标分割的前景. 最后通过max和softmax操作得到ROI的类别和segmentation mask. <Br>
3) 个人总结, 这种encode K*K的相对位置的策略有几个好处, 1.对位置敏感, 这正是instance任务需要的; 2.对ROI的偏移有一定程度的鲁棒性; 3.可以使需要对每个ROI分别进行的subnetwork变得更小, 节省时间.  <Br>
	
### **DWT ★**
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


### **BAIS ★**
**[Paper]**  Boundary-aware Instance Segmentation<Br>
**[Year]** CVPR 2017 <Br>
**[Authors]** [Zeeshan Hayder](https://scholar.google.com.au/citations?user=K2INPyYAAAAJ&hl<Br>=en), [Xuming He](https://xmhe.bitbucket.io/), [Mathieu Salzmann](http://ttic.uchicago.edu/~salzmann/) <Br>
**[Pages]**  https://vitalab.github.io/deep-learning/2017/08/22/boundary-aware.html<Br>
**[Description]**<Br>
1) 提出一种基于距离变换的instance segmentation方法, 可以克服bounding box不准确的问题. 包括三部分: 提取bounding box, 预测object mask (OMN), object分类, 整个网络都是可微的, 可端到端训练;
2) OMN基于目标内像素到其边界的距离变换, 设计网络得到K个boundary-aware object mask, 然后decode成完整的object mask;
3) 仿照MNC, 采用multi-stage策略: 根据上一阶段得到的object mask, 对bounding box进行refine;
4) 实验及一些具体实现没研究, 如object mask与bounding box feature是如何结合起来的等;

### **Instancecut**
**[Paper]**  Instancecut: From edges to instances with multicut <Br>
**[Year]** CVPR 2017 <Br>
**[Authors]** Alexander Kirillov, Evgeny Levinkov, [Bjoern Andres](http://www.andres.sc/), Bogdan Savchynskyy, [Carsten Rother](https://hci.iwr.uni-heidelberg.de/vislearn/) <Br>
**[Pages]** <Br>
**[Description]**<Br>
1) 粗读. 提出一种通过instanc-agnostic的segmentation和edge detection做instance segmentation的方法, . <Br>
2) 分割和边缘检测使用的都是FCN, 其中边缘检测的网络为了得到sharp edge做了一些改动. 最后通过image partition block使用segmentation和edge的信息得到instance的分割结果, 为了加速是在superpixel上进行的, 这段没仔细看<Br>
3) 此方法的一个很大缺点是不适用于不连续的instance, 此外其性能与soa相比还有很大差距. 这一类的方法虽然思路上比较新奇, 但实际应用上还是有很大局限性, 并且它们大多只在cityscape上进行实验, 可能是这种方法只对这种目标较小且形状较为简单的情况效果好? <Br>

### **Mask R-CNN ★★★**
**[Paper]**  Mask R-CNN <Br>
**[Year]** ICCV 2017 <Br>
**[Authors]** [Kaiming He](http://kaiminghe.com/), [Georgia Gkioxari](https://gkioxari.github.io/), [Piotr Dollár](https://pdollar.github.io/), [Ross Girshick](http://www.rossgirshick.info/) <Br>
**[Pages]** <Br>
https://github.com/facebookresearch/Detectron <Br>
https://github.com/matterport/Mask_RCNN <Br>
**[Description]**<Br>
1) 在Faster R-CNN基础上加入了Mask Head, 用于预测每个目标的mask. <Br>
2) 骨干网络换成了ResNeXt+FPN, 根据ROI的尺寸选择从C2到C5的某个level中取feature作为Mask和Cls Head的输入. <Br>
3) Mask Head为每个类别都预测一个mask, 最后取Cls Head中预测的概率最大的类别作为此目标分割结果, 这么做的好处是避免了类间竞争, 对instance segmentation任务来说能带来明显的性能提升. <Br>
4) ROI Align代替ROI pooling, 用双线性插值计算相应位置的feature而不是简单地取整. 这可以大大减小feature的位置偏差, 对预测mask起重要作用. <Br>
5) Mask R-CNN已集成到Deterctron中, 应掌握这个框架. <Br>
	
### **SGN**
**[Paper]**  SGN: Sequential grouping networks for instance segmentation <Br>
**[Year]** ICCV 2017 <Br>
**[Authors]** [Shu Liu](http://shuliu.me/), [Jiaya Jia](http://www.cse.cuhk.edu.hk/~leojia/), [Sanja Fidler](https://www.cs.utoronto.ca/~fidler/), [Raquel Urtasun](http://www.cs.toronto.edu/~urtasun/) <Br>
**[Pages]** <Br>
**[Description]**<Br>

### **BlitzNet**
**[Paper]**  BlitzNet: A Real-Time Deep Network for Scene Understanding <Br>
**[Year]** ICCV 2017 <Br>
**[Authors]**  [Nikita Dvornik](http://lear.inrialpes.fr/people/mdvornik/), [Konstantin Shmelkov](http://lear.inrialpes.fr/people/kshmelko/), [Julien Mairal](http://lear.inrialpes.fr/people/mairal/), [Cordelia Schmid](http://lear.inrialpes.fr/people/schmid)<Br>
**[Pages]** http://thoth.inrialpes.fr/research/blitznet/<Br>
**[Description]**<Br>
	
### **PANet ★★**
**[Paper]**  Path Aggregation Network for Instance Segmentation <Br>
**[Year]** CVPR 2018 Spotlight <Br>
**[Authors]** [Shu Liu](http://shuliu.me/), Lu Qi, Haifang Qin, [Jianping Shi](http://shijianping.me/), [Jiaya Jia](http://www.cse.cuhk.edu.hk/~leojia/) <Br>
**[Pages]** <Br>
**[Description]**<Br>
1) Mask R-CNN的改进, COCO 2017 Instance冠军. paper偏工程, 很多东西理论性不强, 但实际工程中可以借鉴<Br>
2) 在FPN后面加了一Bottom-up Path Augmentation, 就是给高层特征加入底层语义信息. <Br>
3) Adaptive Feature Pooling, 就是从每个level都取特征, 在后续网络的某个位置用MAX或SUM融合起来, box一支在fc1和fc2之间融合效果好, mask一支貌似没提到. <Br>
4) mask一支加入了一全连接层, 并与原来的结果fuse起来. 道理说服力不强, 但从结果来看提升了效果. <Br>
	
### **MaskLab ★**
**[Paper]**  MaskLab: Instance Segmentation by Refining Object Detection with Semantic and Direction Features  <Br>
**[Year]** CVPR 2018 <Br>
**[Authors]** [Liang-Chieh Chen](http://liangchiehchen.com/), [Alexander Hermans](https://www.vision.rwth-aachen.de/person/10/), [George Papandreou](http://ttic.uchicago.edu/~gpapan/), [Florian Schroff](http://www.florian-schroff.de/), [Peng Wang](https://scholar.google.de/citations?user=7lLdhrIAAAAJ&hl=en&oi=ao), Hartwig Adam <Br>
**[Pages]** <Br>
**[Description]**<Br>
1) Liang-Chieh Chen的xxLab系列又一弹, detection-based和segmentation-based结合做instance. 粗读, 和DeepLab一样还是偏工程的. <Br>
2) 分为box detection, semantic segmentation logits和direction prediction logits三部分. box detection负责检测目标的bounding box和其类别; semantic segmentation负责得到整张图中的分割label map; direction prediction负责得到每个pixel相对于其所属instance中心的方向. <Br>
3) 检测出某一目标的bbox和类别后, 从semantic和direction的feature中分别crop出相应区域, 将direction进行assemble, 做法与instanceFCN基本相同, 然后把semantic和direction的feature map concat起来完成最后的分割. <Br>
4) 用了Hypercolumn, atrous, deform conv等多种技术, 目前看来效果不如mask r-cnn(20180721). <Br>

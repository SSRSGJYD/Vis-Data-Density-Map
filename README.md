# Vis-Data-Density-Map

This repository implements naive density map to visualize high-dimension data.

This repository includes:

+ Dimension reduction of high-dimension data
+ Common kernel functions
+ Density map generation by kernel estimation
+ Visualization of density map



## Dimension Reduction

We use common dimension reduction method, i.e., TSNE. However, when dimension grows larger, TSNE is much slower and does not work well. Therefore, we first use PCA to reduce dimension to 100, then use TSNE to reduce dimension to 2.



## Density Map from Kernel Density

For efficiency issue, we divide 2-D Euclidean space into <img src="svgs/dc0d766062f00b3bf4ff7af4097991d4.svg?invert_in_darkmode" align=middle width=52.83089789999999pt height=22.465723500000017pt/> grids and count number of points in each grid by class. For <img src="svgs/9b325b9e31e85137d1de765f43c0f8bc.svg?invert_in_darkmode" align=middle width=12.92464304999999pt height=22.465723500000017pt/> classes, there should be <img src="svgs/9b325b9e31e85137d1de765f43c0f8bc.svg?invert_in_darkmode" align=middle width=12.92464304999999pt height=22.465723500000017pt/> counting results which are called grid maps. Then we do kernel density estimation on those  <img src="svgs/9b325b9e31e85137d1de765f43c0f8bc.svg?invert_in_darkmode" align=middle width=12.92464304999999pt height=22.465723500000017pt/> grid maps. We notice that KDE can be implemented by convolution operation on the grid maps.

Kernel function is used in kernel density estimation. We implement some common kernel functions:

+ Uniform: <img src="svgs/b53eac71230c188b6a7e64c821d3f314.svg?invert_in_darkmode" align=middle width=153.50939505pt height=27.77565449999998pt/>
+ Triangular: <img src="svgs/c2917ba5cf49eace32d82f7489699aa1.svg?invert_in_darkmode" align=middle width=172.4934189pt height=33.20539859999999pt/>
+ Gaussian: <img src="svgs/b2b87742b851d6e3d6427b63648cf3f5.svg?invert_in_darkmode" align=middle width=118.25701964999998pt height=40.95817769999997pt/>

A 2-D kernel function can be calculated by multiply 1-D kernel functions of x and y dimensions:
<p align="center"><img src="svgs/66a0ea1187b413f8d08fc713f619cc5d.svg?invert_in_darkmode" align=middle width=167.67521759999997pt height=16.438356pt/></p>
In our implementation, we limit the range of Gaussian density estimation by using a Gaussian kernel. All kernel size should be odd number.



## Density Map Visualization

For <img src="svgs/9b325b9e31e85137d1de765f43c0f8bc.svg?invert_in_darkmode" align=middle width=12.92464304999999pt height=22.465723500000017pt/> classes and grid map of <img src="svgs/dc0d766062f00b3bf4ff7af4097991d4.svg?invert_in_darkmode" align=middle width=52.83089789999999pt height=22.465723500000017pt/>, we visualize it on a <img src="svgs/dc0d766062f00b3bf4ff7af4097991d4.svg?invert_in_darkmode" align=middle width=52.83089789999999pt height=22.465723500000017pt/> image where each pixel represent a grid. 

Numbers of points in each class may vary greatly. To balance between classes, we first normalize grid maps by class so that maximum value is 1 and minimum value is 0 in each class:
<p align="center"><img src="svgs/f604181f350fd9005e49fd5ad9f6dc54.svg?invert_in_darkmode" align=middle width=377.0290128pt height=36.0951987pt/></p>
where
<p align="center"><img src="svgs/9b004964749a076e39f9c16366ede485.svg?invert_in_darkmode" align=middle width=709.57478295pt height=25.1935035pt/></p>
To represent density of points, we define saturation value of each pixel according to total density of the corresponding grid:
<p align="center"><img src="svgs/416bea9543453555d4605bce2ebbdfd6.svg?invert_in_darkmode" align=middle width=210.90709364999998pt height=47.60747145pt/></p>
where <img src="svgs/6bac6ec50c01592407695ef84f457232.svg?invert_in_darkmode" align=middle width=13.01596064999999pt height=22.465723500000017pt/> is used to ensure that all saturation value is valid:
<p align="center"><img src="svgs/8a24aaa0c10c91f1aa02a7427af6aa4d.svg?invert_in_darkmode" align=middle width=259.6908897pt height=47.60747145pt/></p>
To show distribution of each class, we use per-pixel color interpolation in HSV color space. For each of the <img src="svgs/9b325b9e31e85137d1de765f43c0f8bc.svg?invert_in_darkmode" align=middle width=12.92464304999999pt height=22.465723500000017pt/> classes, we assign a specific Hue value. Then we calculate hue value of each pixel as weighted sum of all class hue values:
<p align="center"><img src="svgs/41caa49522b9544ac8687d8ef4e4ea71.svg?invert_in_darkmode" align=middle width=312.26328209999997pt height=46.72166729999999pt/></p>
We set value of V to 1 for all pixels.



## Get Started

Example data of 1000 MNIST images has been put in `./data` . 

First, create a Python 3.x environment and install following Python packages:

+ matplotlib
+ numpy
+ opencv-python
+ scipy
+ sklearn

You can install by this shell command:

```shell
pip3 install -r requirements.txt
```

To do dimension reduction, execute:

```shell
python dimension_reduction.py
```

TSNE results are saved into `./tsne` folder. 

To visualize density map, execute:

```shell
python density_map.py
```

Density maps are saved into `./result` folder. 



## Experiment Results

### Dimension Reduction

#### Experiment 1: Perplexity & Iteration of TSNE

| iter | perplexity=10                                   | perplexity=30                                   | perplexity=50                                    |
| ---- | ----------------------------------------------- | ----------------------------------------------- | ------------------------------------------------ |
| 250  | <img src="tsne/10_250.png" style="zoom:30%;" /> | <img src="tsne/30_250.png" style="zoom:30%;" />   | <img src="tsne/50_250.png" style="zoom:30%;" /> |
| 500  | <img src="tsne/10_500.png" style="zoom:30%;" /> | <img src="tsne/30_500.png" style="zoom:30%;" /> | <img src="tsne/50_500.png" style="zoom:30%;" /> |
| 1000 | <img src="tsne/10_1000.png" style="zoom:30%;" /> | <img src="tsne/30_1000.png" style="zoom:30%;" /> | <img src="tsne/50_1000.png" style="zoom:30%;" /> |
| 2000 | <img src="tsne/10_2000.png" style="zoom:30%;" /> | <img src="tsne/30_2000.png" style="zoom:30%;" /> | <img src="tsne/50_2000.png" style="zoom:30%;" /> |
| 3500 | <img src="tsne/10_3500.png" style="zoom:30%;" /> | <img src="tsne/30_3500.png" style="zoom:30%;" /> | <img src="tsne/50_3500.png" style="zoom:30%;" /> |
| 5000 | <img src="tsne/10_5000.png" style="zoom:30%;" /> | <img src="tsne/30_5000.png" style="zoom:30%;" /> | <img src="tsne/50_5000.png" style="zoom:30%;" /> |

Above results indicate that:

+ perplexity=10 is not suitable, perplexity=50 works well.
+ number of iterations should be larger than 2000.

#### Experiment 2: PCA Preprocessing

We fix perplexity to 30 and 50 respectively.

Perplexity=30:

| Iter | No PCA | PCA (dim=100) | PCA (dim=50) |
| ---- | ------ | ------------- | ------------ |
| 2000 | <img src="tsne/30_2000.png" style="zoom:30%;" /> | <img src="tsne/pca100_30_2000.png" style="zoom:30%;" /> | <img src="tsne/pca50_30_2000.png" style="zoom:30%;" /> |
| 3500 | <img src="tsne/30_3500.png" style="zoom:30%;" /> | <img src="tsne/pca100_30_3500.png" style="zoom:30%;" /> | <img src="tsne/pca50_30_3500.png" style="zoom:30%;" /> |
| 5000 | <img src="tsne/30_5000.png" style="zoom:30%;" /> | <img src="tsne/pca100_30_5000.png" style="zoom:30%;" /> | <img src="tsne/pca50_30_5000.png" style="zoom:30%;" /> |

Perplexity=50:

| Iter | No PCA | PCA (dim=100) | PCA (dim=50) |
| ---- | ------ | ------------- | ------------ |
| 2000 | <img src="tsne/50_2000.png" style="zoom:30%;" /> | <img src="tsne/pca100_50_2000.png" style="zoom:30%;" /> | <img src="tsne/pca50_50_2000.png" style="zoom:30%;" /> |
| 3500 | <img src="tsne/50_3500.png" style="zoom:30%;" /> | <img src="tsne/pca100_50_3500.png" style="zoom:30%;" /> | <img src="tsne/pca50_50_3500.png" style="zoom:30%;" /> |
| 5000 | <img src="tsne/50_5000.png" style="zoom:30%;" /> | <img src="tsne/pca100_50_5000.png" style="zoom:30%;" /> | <img src="tsne/pca50_50_5000.png" style="zoom:30%;" /> |

It indicates that PCA preprocessing slightly help separate points of different classes. 

We compare efficiency of different methods (in second, perplexity=50):

<table>
    <tr>
        <td rowspan="2">iter</td>
        <td>TSNE</td>
        <td colspan="3">PCA (dim=100)</td>
        <td colspan="3">PCA (dim=50)</td>
    </tr>
    <tr>
        <td>Total</td>
        <td>PCA</td>
        <td>TSNE</td>
        <td>Total</td>
        <td>PCA</td>
        <td>TSNE</td>
        <td>Total</td>
    </tr>
    <tr>
        <td>2000</td>
        <td>10.05</td>
        <td rowspan="3">0.93</td>
        <td>8.98</td>
        <td>9.91</td>
        <td rowspan="3">0.61</td>
        <td>8.93</td>
        <td>9.54</td>
    </tr>
    <tr>
        <td>3500</td>
        <td>16.00</td>
        <td>14.88</td>
        <td>15.81</td>
        <td>14.77</td>
        <td>15.38</td>
    </tr>
    <tr>
        <td>5000</td>
        <td>21.76</td>
        <td>20.65</td>
        <td>21.58</td>
        <td>20.57</td>
        <td>21.18</td>
    </tr>
</table>

It indicates that PCA preprocessing is a litter bit faster than pure TSNE method.



### Density Map Visualization

#### Comparison of Kernel Functions

We set <img src="svgs/121c390773b2ac8a4c60b6ef7c49568c.svg?invert_in_darkmode" align=middle width=101.3011164pt height=22.465723500000017pt/> and compare visualization results from different kernel functions:

| kernel size | Uniform                                                      | Gaussian                                                     | Triangular                                                   |
| ----------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 7           | <img src="result/uniform,ksize=7,normclass=True(200).png" style="zoom:100%;" /> | <img src="result/gaussian,ksize=7,normclass=True(200).png" style="zoom:100%;" /> | <img src="result/triangular,ksize=17,normclass=True(200).png" style="zoom:100%;" /> |
| 17          | <img src="result/uniform,ksize=17,normclass=True(200).png" style="zoom:100%;" /> | <img src="result/gaussian,ksize=17,normclass=True(200).png" style="zoom:100%;" /> | <img src="result/triangular,ksize=7,normclass=True(200).png" style="zoom:100%;" /> |
| 27          | <img src="result/uniform,ksize=27,normclass=True(200).png" style="zoom:100%;" /> | <img src="result/gaussian,ksize=27,normclass=True(200).png" style="zoom:100%;" /> | <img src="result/triangular,ksize=27,normclass=True(200).png" style="zoom:100%;" /> |

It indicates that uniform kernel produces smoothest density map, while triangular kernel produces sharpest density map.

#### Influence of Sampling Resolution and Kernel Size

We use uniform kernel:

| kernel size | <img src="svgs/c987ea897aa6c55cfb9a2743f83e5ae2.svg?invert_in_darkmode" align=middle width=52.968029399999985pt height=21.18721440000001pt/> | <img src="svgs/422da1937d16c5751312b249c2e04fcd.svg?invert_in_darkmode" align=middle width=69.40644809999999pt height=21.18721440000001pt/> | <img src="svgs/e0ab29f0a625e71b180d1eee3252be89.svg?invert_in_darkmode" align=middle width=69.40644809999999pt height=21.18721440000001pt/>                                  |
| ----------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 7           | <img src="result/uniform,ksize=7,normclass=True(50).png" style="zoom:100%;" /> | <img src="result/uniform,ksize=7,normclass=True(100).png" style="zoom:100%;" /> | <img src="result/uniform,ksize=7,normclass=True(200).png" /> |
| 17          | <img src="result/uniform,ksize=17,normclass=True(50).png" style="zoom:100%;" /> | <img src="result/uniform,ksize=17,normclass=True(100).png" style="zoom:100%;" /> | <img src="result/uniform,ksize=17,normclass=True(200).png" /> |
| 27          | <img src="result/uniform,ksize=27,normclass=True(50).png" style="zoom:100%;" /> | <img src="result/uniform,ksize=27,normclass=True(100).png" style="zoom:100%;" /> | <img src="result/uniform,ksize=27,normclass=True(200).png" /> |

It indicates that using smaller <img src="svgs/7b9a0316a2fcd7f01cfd556eedf72e96.svg?invert_in_darkmode" align=middle width=14.99998994999999pt height=22.465723500000017pt/> and <img src="svgs/84c95f91a742c9ceb460a83f9b5090bf.svg?invert_in_darkmode" align=middle width=17.80826024999999pt height=22.465723500000017pt/> (larger grid size) makes density maps smoother. Moreover, with different <img src="svgs/7b9a0316a2fcd7f01cfd556eedf72e96.svg?invert_in_darkmode" align=middle width=14.99998994999999pt height=22.465723500000017pt/> and <img src="svgs/84c95f91a742c9ceb460a83f9b5090bf.svg?invert_in_darkmode" align=middle width=17.80826024999999pt height=22.465723500000017pt/>, suitable kernel size is also changing. Generally, when <img src="svgs/7b9a0316a2fcd7f01cfd556eedf72e96.svg?invert_in_darkmode" align=middle width=14.99998994999999pt height=22.465723500000017pt/> and <img src="svgs/84c95f91a742c9ceb460a83f9b5090bf.svg?invert_in_darkmode" align=middle width=17.80826024999999pt height=22.465723500000017pt/> becomes larger, the suitable kernel size becomes larger and vice versa.



## Reference

Maaten, Laurens van der, and Geoffrey Hinton. "Visualizing data using t-SNE." *Journal of machine learning research* 9.Nov (2008): 2579-2605.


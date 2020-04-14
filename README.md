# IGOS
This is a simple pytorch demo of the Integrated-Gradients Optimized Saliency (I-GOS) described in
>**Visualizing Deep Networks by Optimizing with Integrated Gradients** ([PDF](https://arxiv.org/abs/1905.00954))<br>
Zhongang Qi, Saeed Khorram, Fuxin Li, in arXiv preprint arXiv:1905.00954. 

For project website, please see:<br>
https://igos.eecs.oregonstate.edu/

## Quick Start
**IGOS_generate_video.py**: utilize I-GOS to generate saliency maps for the images in **‘./Images/’** on the pretrained VGG19 networks for ImageNet from the PyTorch model zoo, and then write video for each image to demonstrate how the deletion and insertion curves change.  

Here are some example results:

Saliency maps for dalmatian:

![](Results/dalmatian_IGOS.png) ![](Results/dalmatian_heatmap.png) 

Videos:

![](Results/dalmatian.gif)

The intuition behind the **Deletion** metric is that the removal of the pixels most relevant to a class will cause the original class score dropping sharply;
The intuition behind the **Insertion** metric is that only keeping the most relevant pixels will retain the original score as much as possible.


## Dependencies
All code is written in Python 3.6. To install the dependencies, first install and activate a virtual environment:
```
python -m venv env
source env/bin/activate
```
and then you can install the dependencies using `pip`:
```
pip install -r requirements.txt
```
Now, you just need to run the `IGOS_generate_video.py` to generate saliency map and insertion/deletion video:
```
python IGOS_generate_video.py
```
<br> 

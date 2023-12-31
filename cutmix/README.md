# CutMix Implementation

## 1. What is CutMix ?
In short, CutMix is an image augmentation technique that aims to generate image by blending to images with a ratio.
Instead of replacing a random patch with black, white pixels, this technique fill it with a patch extracted from a different image in the train dataset.
![image](https://github.com/Mikyx-1/Deep_Learning_Techniques/assets/92131994/f346f8e8-37d5-46a7-98f8-2c4de77df8b4)

## 2. How does it work ?
![Screenshot from 2023-12-31 17-15-02](https://github.com/Mikyx-1/Deep_Learning_Techniques/assets/92131994/31988f8c-61ed-4c32-b1b4-412e63bdbf9c)



## 3. What are the shortcomings ?
This technique is risky as it can either improve or deteriorate performance. For example, it can take the irrelevant patch and 
place it in the crucial part of the image, this will result in a huge confusion of a model

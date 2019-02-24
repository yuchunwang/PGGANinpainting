# PGGANinpainting
High Resolution Image Inpainting with Progressive Growing of GAN

(image-size:256x256 pixel)(hiding-size:128x128 pixel)

 ## How to train the model:
 
 1.   Download the celebA dataset.
 2.   Training the model
 
 step:
 
     python create.py celebA
     python main.py --path=256-train_val_test/training/folder1/ --celeba=True
     
     
     
The picture shows Our model’s performance.
(image-size:256x256 pixel)(hiding-size:128x128 pixel)

![image](https://github.com/yuchunwang/PGGANinpainting/blob/master/testperformance.png)
     
     
   
> Overview of our method to generate the inpainting image progressively.

![image](https://github.com/yuchunwang/PGGANinpainting/blob/master/architecture.png)
        
 

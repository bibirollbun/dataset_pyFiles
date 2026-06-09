# Model 1
https://www.kaggle.com/code/saisaingkham/cp3501-efficient-b0
# Model 2
https://www.kaggle.com/code/banyarsawthit/resnet50-opt

# Model 3
https://www.kaggle.com/code/minkhantkyawfreddy/cp3501-2024-tr2-using-fastai-lessons-v1/notebook
# Model 4
https://www.kaggle.com/code/minkhantkyawfreddy/cp3501-2024-tr2-using-fastai-lessons-v1

# Model 5
https://www.kaggle.com/code/haymanhninaye/mobilenetv2
# Model 6
https://www.kaggle.com/code/haymanhninaye/visiontransformer

# Model 7
https://www.kaggle.com/code/banyarsawthit/cp3501-2025-tr1s-using-fastai-vgg16
# Model 8
https://www.kaggle.com/code/banyarsawthit/cp3501-2024-tr2-using-fastai-densenet-121


# Model 1: 0.64750 (EfficientNet-B0)
# Model 2: 0.71750 (RestNet50)
# Model 3: 0.84750 (ResNet34)
# Model 4: 0.85250 (ResNet18)
# Model 5: 0.99250 (VGG-16)
# Model 6: 1.27250 (DenseNet-121)
# Model 7: 0.95000 (MobileNetV2)
# Model 8: 0.80750 (Vision Transformer(Vit))


# Model 1 resnet50 improved
https://www.kaggle.com/code/banyarsawthit/resnet50my5

# Model 2 Efficient B0 improved
https://www.kaggle.com/code/saisaingkham/efficient-b0


import pandas as pd
# let's sort it as per given submission sample
sub = pd.read_csv('/kaggle/input/cp-3501-retinamnist-v-2024/sample_submission.csv')
sub


# Your final best submission
sub.to_csv('submission.csv', index=False)
!head submission.csv





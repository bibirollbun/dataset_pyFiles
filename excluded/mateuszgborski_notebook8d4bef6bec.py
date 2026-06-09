!pip install fastai


from keras.datasets import mnist
from fastai.vision.all import *
from torchvision.models import resnet34


(x_train, y_train), (x_test, y_test) = mnist.load_data()
train_dataset = list(zip(x_train, y_train))


class GetX(ItemTransform): 
    def encodes(self, x): return PILImage.create(x[0])

class GetY(ItemTransform): 
    def encodes(self, x): return x[1]


dblock = DataBlock(blocks=(ImageBlock, CategoryBlock),
                   get_x=GetX,
                   get_y=GetY) 


dls = dblock.dataloaders(train_dataset)


dls.show_batch(max_n=10)


learn = vision_learner(dls, resnet34, metrics=error_rate)


learn.lr_find()


learn.fine_tune(2, 3e-3)


learn.show_results()


interp = Interpretation.from_learner(learn)
interp.plot_top_losses(9, figsize=(15,10))


predictions, _ = learn.get_preds(dl=learn.dls.test_dl(x_test))

predicted_labels = predictions.argmax(dim=1)
actual_labels = torch.tensor(y_test)

missed = (predicted_labels != actual_labels).sum().item()
accuracy = (len(y_test) - missed) / len(y_test)


print(f"Number of misclassifications: {missed}, Size of test dataset: {len(y_test)}")
print(f"Accuracy: {accuracy:.2%}")





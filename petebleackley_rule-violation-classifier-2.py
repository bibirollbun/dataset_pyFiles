# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import itertools
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
import tqdm
import kagglehub
import logging
import traceback

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


from transformers import PreTrainedModel, PretrainedConfig, AutoModel, AutoTokenizer
import accelerate


class GlobalAttentionPoolingHead(torch.nn.Module):
    
    def __init__(self,config):
        """
        Creates the layer
        Parameters
        ----------
        config : transformers.RobertaConfig
                 the configuration of the model

        Returns
        -------
        None.

        """
        size = config.hidden_size
        super(GlobalAttentionPoolingHead,self).__init__()
        self.global_projection = torch.nn.parameter.Parameter(torch.empty((size,32),
                                                                         dtype=torch.float16))
        self.local_projection =  torch.nn.parameter.Parameter(torch.empty((size,32),
                                                                         dtype=torch.float16))
        sigma = (3.0/(4.0*size))**0.25
        torch.nn.init.normal_(self.global_projection,0.0,sigma)
        torch.nn.init.normal_(self.local_projection,0.0,16*sigma)
        self.cosine = torch.nn.CosineSimilarity(dim=2,eps=1.0e-4)
        
    
        
    def forward(self,X,attention_mask=None):
        """
        

        Parameters
        ----------
        X : torch.Tensor
            Base model vectors to apply pooling to.
        attention_mask: tensorflow.Tensor, optional
            mask for pad values
        

        Returns
        -------
        torch.Tensor
            The pooled value.

        """
        if attention_mask is None:
            size=X.size()
            attention_mask = torch.ones((size[0],size[1],1),dtype=torch.float16)
        else:
            attention_mask = attention_mask.unsqueeze(2)
        Xa = X*attention_mask
        gp = torch.einsum('ijk,kl->il',Xa,self.global_projection).unsqueeze(1)
        lp = torch.einsum('ijk,kl->ijl',Xa,self.local_projection)
        attention = self.cosine(lp,gp)*attention_mask.squeeze()
        return torch.einsum('ij,ijk->ik',attention,Xa)


class EncoderModel(PreTrainedModel):
    
    def __init__(self,checkpoint):
        """
        Creates the encoder model

        Parameters
        ----------
        base_model : transformers.ModernBertModel
            The base model

        Returns
        -------
        None.

        """
        self.config = PretrainedConfig.from_pretrained(checkpoint)
        super(EncoderModel,self).__init__(self.config)
        self.encoder = AutoModel.from_pretrained(checkpoint,
                                                 torch_dtype=torch.float16,
                                                 attn_implementation="sdpa")
        self.head = GlobalAttentionPoolingHead(self.config)
        
        
    def forward(self,input_ids,
             attention_mask=None):
        """
        Vectorizes a tokenised text

        Parameters
        ----------
        inputs : tensorflow.Tensor
            tokenized text to endode

        Returns
        -------
        tensorflow.Tensor
            Vector representing the document

        """

        if attention_mask is None and 'attention_mask' in input_ids:
            (input_ids,attention_mask) = (input_ids['input_ids'],input_ids['attention_mask'])
        return self.head(self.encoder(input_ids,
                                      attention_mask).last_hidden_state,
                         attention_mask)


class Classifier(torch.nn.Module):
    def __init__(self,path):
        super(Classifier,self).__init__()
        checkpoint = kagglehub.model_download(path)
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.encoder = EncoderModel(checkpoint)
        self.activation = torch.nn.Sigmoid()
        

    def vectorize(self,text):
        tokens = self.tokenizer(text,
                                padding=True,
                                return_tensors='pt').to('cuda')
        return self.encoder(tokens)

    def forward(self,sample):
        vectors = {key:self.vectorize(sample[key].to_list())
                   for key in ('body',
                               'rule',
                               'positive_example_1',
                               'positive_example_2',
                               'negative_example_1',
                               'negative_example_2')}
        evidence = ((vectors['rule']+vectors['positive_example_1']+vectors['positive_example_2'])
                    -(vectors['negative_example_1']+vectors['negative_example_2']))
        return self.activation(torch.einsum('ij,ij->i',vectors['body'],evidence))
        
            


class DataIterator(torch.utils.data.IterableDataset):
    def __init__(self):
        super(DataIterator,self).__init__()
        self.data = pd.read_csv('/kaggle/input/jigsaw-agile-community-rules/train.csv')
        self.n = self.data.shape[0]

    def __iter__(self):
        shuffled = self.data.sample(self.n)
        for i in range(0,self.n,16):
            batch = shuffled.iloc[i:i+16]
            yield (batch[['body',
                          'rule',
                          'positive_example_1',
                          'positive_example_2',
                          'negative_example_1',
                          'negative_example_2']],
                  batch['rule_violation'])

    def __len__(self):
        return self.n//16 + 1



model = Classifier("answer-ai/modernbert/pytorch/base")
corpus = DataIterator()
optimizer = torch.optim.Rprop(model.parameters(),lr=5.0e-5)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=0.9)
accelerator = accelerate.Accelerator()
(model,optimizer,scheduler) = accelerator.prepare(model,optimizer,scheduler,
                                                  device_placement=[True,True,True])
loss_fn = torch.nn.BCELoss()
for epoch in range(100):
    print("Epoch",epoch)
    for (X,Y) in tqdm.tqdm(corpus):
        prediction = model(X)
        Y = torch.tensor(Y.array,dtype=torch.float16,device=accelerator.device)
        loss = loss_fn(prediction,Y)
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()
    scheduler.step()


def infer(test_data):
    return pd.DataFrame({'row_id':test_data['row_id'],
                               'rule_violation':model(test_data).numpy(force=True)})

def chunks(data):
    rows = data.index.values
    n = data.shape[0]
    for i in range(0,n,16):
        sequence = rows[i:i+16]
        yield data.loc[sequence]
    
    


logger = logging.getLogger(__name__)
data_path = "/kaggle/input/jigsaw-agile-community-rules/test.csv" 

test_data = pd.read_csv(data_path)

predictions = pd.concat([infer(chunk)
                         for chunk in chunks(test_data)])
predictions.to_csv('submission.csv',index=False)



predictions





# miXed

Automatic mixed precision training and inference on GPU. 

As the deep learning models like transformers getting larger and larger, its very difficult to train and perform inference on those models. You might feel the urge of training large language models but whenever you try to run that in your GPU, you might face cuda memory error. Training with `batch_size=1` is also not a good solution. So there is need of larger VRAM, but with AMP(automatic mixed precision) you can optimize the gpu memory usage by 2x, you can increase the `batch_size` which will provide you extra throughput. But the question is how it works, what are the principles behind AMP.

Whenever we create some deep learning model, the datatypes are usually in Float32, but in AMP its a extension to pytorch which tries to optimize the vram by using mixed precision (Float32 and Float16 both). But why both? Its intuitive that using float16 alone will optimize the entire process. But why do we need to use mixed precision. Before answering that lets see the structure of these `FP16` and `FP32` datatypes.


<img src="https://imgur.com/xIPuhut.png" height=200>

#### Advantages of using FP16 over FP32
|                        | Range FP32                  | Range FP16               |
|------------------------|-----------------------------|--------------------------|
|  Range                 | `1.4x10^-45` to `3.4x10^38` | `5.96x10^-8` to `655504` |
|  Compute throughput    | 1x                          | 8x (Depends on the gpu)  |
|  Memory throughput     | 1x                          | 2x                       |
|  Memory stroage        | 1x                          | 2x                       |

#### Some drawbacks of `FP16`

Due to less precision `FP16` cannot capture small accumulations, but `FP32` can.
Example:
```python3
# half precision
torch.ones((65536,), device='cuda', dtype=torch.half).sum()
>>> tensor(inf, device='cuda:0', dtype=torch.float16) 

# single precision
torch.ones((65536,), device='cuda', dtype=torch.float32).sum()
>>> tensor(65536., device='cuda:0')
```

Also weight update requires high precision, FP16 can't capture. Lets try `1+0.0001`
```python3 
# half precision
torch.tensor(1, device='cuda', dtype=torch.half) + 0.0001
>>> tensor(1., device='cuda:0', dtype=torch.float16)

# single precision
torch.tensor(1, device='cuda') + 0.0001
>>> tensor(1.0001, device='cuda:0')
```
These are the cases where `FP32` performs better than `FP16`

Usecases of `FP16`: In case of matmul, convolution operations, tensor add, pointwise tensor mul.

Usecases of `FP32`:During weight updates, reductions like loss functions, softmax, norms etc.

### Gradient scalling
This is the most important thing about AMP. Most of the forward propagation task will run on `FP16` mode other than some activation functions. But when you get the loss of a certain batch the backpropagation starts. As I have shown in the `1+0.0001` example `FP16` can't capture small accumulations so as we go up the network calculating the gradients, those gradients becomes very small and the model stops training. To mitigate this there is a feature named `gradient scalling` which scales the loss so with the chain rule it flows to the gradients of all the layers which tries to uplift the gradients in certain range. In pytorch this will be handled by `GradScaler()`. After it computes the grads, the weights update will take place in `FP32` mode.


### Lets train a model with both single precision and AMP
Before start writing the script, it has been tested that AMP doesn't decrease the accuracy of the model compared to a model trained on single precision. In this demonstration we would only focus on the execution time, not the model quality. 

Model we will be using is `google/bert_uncased_L-12_H-768_A-12` from 🤗 and we are going to add a linear layer on top of the `[CLS]` output vector for classification.

```python3
import time
import torch
import datasets
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast

class ModelDataset(Dataset):
    """Custom dataset"""
    def __init__(self, data):
        self.data = data
        self.tokenizer = AutoTokenizer.from_pretrained("google/bert_uncased_L-12_H-768_A-12")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        o = self.tokenizer(
                self.data[idx]['text'], 
                max_length=512, 
                padding='max_length', 
                truncation=True, 
                return_tensors='pt'
            )
        label = torch.tensor([self.data[idx]['label']]).type(torch.float32)
        return o['input_ids'][0], o['attention_mask'][0], label

class ClassificationModel(torch.nn.Module):
    def __init__(self):
        super(ClassificationModel, self).__init__()
        self.bert = AutoModel.from_pretrained('google/bert_uncased_L-12_H-768_A-12') 
        self.dropout = torch.nn.Dropout(0.1)
        self.fc = torch.nn.Linear(768,1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, ids, mask): 
        o = self.bert(ids, mask)['last_hidden_state'][:,0,:]
        o = self.dropout(o)
        o = self.fc(o)
        return self.sigmoid(o)
    
# prepare dataset    
tweet_dataset = datasets.load_dataset('tweet_eval','offensive')
custom_dataset = ModelDataset(tweet_dataset['train'])
 
# change the batch_size to 8 if you are using same experimental setup
loader = DataLoader(custom_dataset, shuffle=True, batch_size=16)

# load model, loss, and optimizer
model = ClassificationModel().to('cuda')
criterion = torch.nn.functional.binary_cross_entropy_with_logits
optim = torch.optim.AdamW(model.parameters(), lr=2e-5)
```

### Difference between single precision vs mixed-precision training loop

<table>
<tr>
<td>

  ```python3
# without mixed precision
t0 = time.time()
model.train()
  
for i in range(5):
  
    avg_loss = 0.0
    for batch in loader:
  
        ids, mask, label = batch
        ids = ids.to('cuda')
        mask = mask.to('cuda')
        label = label.to('cuda')
        optim.zero_grad()
        loss = criterion(model(ids, mask), label)
        loss.backward()
        optim.step()
        avg_loss += loss
    print(loss/len(loader))
  
print(time.time()-t0)
  ```
</td>
<td>

  ```python3
# with mixed precision
scaler = GradScaler()
t0 = time.time()
model.train()
for i in range(5):
    avg_loss = 0.0
    for batch in loader:
        ids, mask, label = batch
        ids = ids.to('cuda')
        mask = mask.to('cuda')
        label = label.to('cuda')
        optim.zero_grad()
        with autocast():
            loss = criterion(model(ids, mask), label)

        scaler.scale(loss).backward()
        scaler.step(optim)
        scaler.update()
        avg_loss += loss
    print(loss/len(loader))
print(time.time() - t0)
  ```
</td>
</tr>
</table>

I have done the experiment in `1xTesla T4`

Training time for single precision: `5149 seconds`

Training time for mixed precision: `2163 seconds`

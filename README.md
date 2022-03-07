# miXed

Automatic mixed precision training and inference on GPU. 

<img src="https://imgur.com/xIPuhut.png" height=200>

|                        | Range FP32                  | Range FP16               |
|------------------------|-----------------------------|--------------------------|
|  Range                 | `1.4x10^-45` to `3.4x10^38` | `5.96x10^-8` to `655504` |
|  Compute throughput    | 1x                          | 8x (Depends on the gpu)  |
|  Memory throughput     | 1x                          | 2x                       |
|  Memory stroage        | 1x                          | 2x                       |

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
So when to use half precision:
In case of matmul, convolution operations, tensor add, pointwise tensor mul.

When to use single precision:
During weight updates, reductions like loss functions, softmax, norms etc.


Before Starting going through this documentation I am expecting that you have some level of idea regarding deep learning model training and inference on `torch`. Even if you don't have any knowledge regarding that here I have presented a very basic 🤗's `transformers` based classification model with custom training loop in `torch`.

```python3
import time
import torch
import datasets
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast

class ModelDataset(Dataset):
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

I have done the experiment in 1xTesla T4
Training time for single precision -> 5149 seconds
Training time for mixed precision -> 2163 seconds

# miXed

Automatic mixed precision training and inference on GPU. 

<img src="https://imgur.com/xIPuhut.png" height=200>

|                        | Range FP32                  | Range FP16               |
|------------------------|-----------------------------|--------------------------|
|  Range                 | `1.4x10^-45` to `3.4x10^38` | `5.96x10^-8` to `655504` |
|  Compute throughput    | 1x                          | 8x (Depends on the gpu)  |
|  Memory throughput     | 1x                          | 2x                       |
|  Memory stroage        | 1x                          | 2x                       |

Before Starting going through this documentation I am expecting that you have some level of idea regarding deep learning model training and inference on `torch`. Even if you don't have any knowledge regarding that here I have presented a very basic 🤗's `transformers` based classification model with custom training loop in `torch`.

```python3
import torch
from transformers import AutoModel, AutoTokenizer

# dataloader = 

class ClassificationModel(torch.nn.Module):
  def __init__(self):
    super(ClassificationModel, self).__init__()
    self.bert = AutoModel.from_pretrained('bert-base-uncased') 
    self.dropout = torch.nn.Dropout(0.1)
    self.fc = torch.nn.Linear(512,1)
    self.sigmoid = torch.nn.Sigmoid()
    
  def forward(self, sent_id, mask): 
    _, o = self.bert(sent_id, mask)[0]
    o = self.dropout(o)
    o = self.fc(o)
    return self.sigmoid(o)
    
model = ClassificationModel().to('cuda')
criterion = torch.loss.BCELoss()
optim = torch.optim.AdamW(model.parameters(), lr=2e-5)

model.train()
for i in range(5):
  for id, mask, label in loader:
    optim.zero_grad()
    loss = criterion(model(id,mask), label)
    loss.backward()
    optim.step()
```

### Difference between single precision vs mixed-precision training loop

<table>
<tr>
<td>

  ```python3
  model = Model().to('cuda')
criterion = BCELoss()

model.train()
for i in range(5):
  for id, mask, label in loader:
    optim.zero_grad()
    o = model(id,mask)
    loss=criterion(o, label)
    loss.backward()
    optim.step()
  ```
</td>
<td>

  ```python3
model = Model().to('cuda')
criterion = BCELoss()

model.train()
for i in range(5):
  for id, mask, label in loader:
    optim.zero_grad()
    o = model(id,mask)
    loss=criterion(o, label)
    loss.backward()
    optim.step()
  ```
</td>
<td>
  Variables defined with <code>def</code> cannot be changed once defined. This is similar to <code>readonly</code> or <code>const</code> in C# or <code>final</code> in Java. Most variables in Nemerle aren't explicitly typed like this.
</td>
</tr>
</table>






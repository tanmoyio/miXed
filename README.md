# miXed

Automatic mixed precision training and inference on GPU. 

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

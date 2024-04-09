import os
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
import unidecode
# from sklearn import metrics
import torch
from torch.utils.data import Dataset, DataLoader 
from transformers import DistilBertTokenizer, DistilBertModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

path = '/home/Kaggle-challenge/workspace/data/'


train_d = pd.read_csv(path + 'train.csv')
# train_d = pd.read_csv('make_train.csv')
# train_d = pd.read_csv('IDENTITY_fill.csv')
test = pd.read_csv(path + 'test.csv')
submission = pd.read_csv(path + 'sample_submissions.csv')

MAX_LEN = 512
TRAIN_BATCH_SIZE = 32 #32 #16 
EPOCHS = 3 # 2 # 1 
LEARNING_RATE = 1e-05
NUM_WORKERS = 2

def remove_emoji(string):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', string)
    text = re.sub("[\{\}\[\]\/?.,;:|\)*~`!^\-_+<>@\#$%&\\\=\(\'\"]", " ", text)
    return text

def preprocess(df, text_column="comment_text"):
    url_pattern = r"https?://\S+|www\.\S+"
    # remove url
    df[text_column] = df[text_column].str.replace(url_pattern, " ")

    # apply unidecode
    df[text_column] = df[text_column].map(unidecode.unidecode)
    
    # remove emoji
    df[text_column] = df[text_column].map(remove_emoji)

    # apply lower
    df[text_column] = df[text_column].str.lower()
    
    return df

train_d = preprocess(train_d)

col = ['toxicity', 'severe_toxicity', 'obscene',
       'threat', 'insult', 'identity_attack', 'sexual_explicit']

# col = ['toxicity','female',
#        'male', 'christian', 'white', 'muslim', 'black',
#        'homosexual_gay_or_lesbian', 'jewish', 'psychiatric_or_mental_illness',
#        'asian']


value_train = train_d[col]
for i in col:
    value_train[i] = (value_train[i] >= 0.5).astype(int)

train_d['target'] = value_train.iloc[:, :].values.tolist()



class MultiLabelDataset(Dataset):

    def __init__(self, dataframe, tokenizer, max_len: int, eval_mode: bool = False):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.text = dataframe.comment_text
        self.eval_mode = eval_mode 
        if self.eval_mode is False:
            self.targets = self.data.target
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text.iloc[index])
        text = " ".join(text.split())

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        output = {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
        }
                
        if self.eval_mode is False:
            output['targets'] = torch.tensor(self.targets.iloc[index], dtype=torch.float)
                
        return output

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', truncation=True, do_lower_case=True)
training_set = MultiLabelDataset(train_d, tokenizer, MAX_LEN)




train_params = {'batch_size': TRAIN_BATCH_SIZE,
                'shuffle': True,
                'num_workers': NUM_WORKERS
                }
training_loader = DataLoader(training_set, **train_params)



class DistilBERTClass(torch.nn.Module):
    
    def __init__(self):
        super().__init__()
        self.l1 = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.classifier_1 = torch.nn.Linear(768,512)
        self.classifier_2 = torch.nn.Linear(512,64)
        self.classifier_3 = torch.nn.Linear(64,768)
        self.dropout = torch.nn.Dropout(0.1)
        self.classifier = torch.nn.Linear(768, 7)

    def forward(self, input_ids, attention_mask, token_type_ids):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.Tanh()(pooler)
        pooler = self.dropout(pooler)
        pooler = self.classifier_1(pooler)
        pooler = torch.nn.Tanh()(pooler)
        pooler = self.dropout(pooler)
        pooler = self.classifier_2(pooler)
        pooler = torch.nn.Tanh()(pooler)
        pooler = self.dropout(pooler)
        pooler = self.classifier_3(pooler)
        pooler = torch.nn.Tanh()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output
    
model = DistilBERTClass()
model.to(device)


def loss_fn(outputs, targets):
    return torch.nn.BCEWithLogitsLoss()(outputs, targets)

optimizer = torch.optim.Adam(params =  model.parameters(), lr=LEARNING_RATE)



def train(epoch):
    
    model.train()
    for _,data in tqdm(enumerate(training_loader, 0)):
        ids = data['ids'].to(device, dtype = torch.long)
        mask = data['mask'].to(device, dtype = torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
        targets = data['targets'].to(device, dtype = torch.float)

        outputs = model(ids, mask, token_type_ids)

        optimizer.zero_grad()
        loss = loss_fn(outputs, targets)
        if _%5000==0:
            print(f'Epoch: {epoch}, Loss:  {loss.item()}')
        
        loss.backward()
        optimizer.step()
        
        
for epoch in range(EPOCHS):
    train(epoch)
    
    


test_set = MultiLabelDataset(test, tokenizer, MAX_LEN, eval_mode = True)
testing_params = {'batch_size': TRAIN_BATCH_SIZE,
               'shuffle': False,
               'num_workers': 2
                }
test_loader = DataLoader(test_set, **testing_params)


all_test_pred = []

def test(epoch):
    model.eval()
    
    with torch.inference_mode():
    
        for _, data in tqdm(enumerate(test_loader, 0)):


            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            outputs = model(ids, mask, token_type_ids)
            probas = torch.sigmoid(outputs)

            all_test_pred.append(probas)
    return probas


probas = test(model)


all_test_pred = torch.cat(all_test_pred)
all_test_pred = all_test_pred.cpu()

submission['prediction'] = all_test_pred[:,0].cpu()

submission.to_csv('submit_5hd3.csv', index=False)
from transformers import RobertaTokenizer, RobertaConfig, RobertaModel
import torch
import torch.nn as nn
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
model = RobertaModel.from_pretrained("python_model")

query = "set a variable as hello world"
query_vec = model(tokenizer(query,return_tensors='pt')['input_ids'])[1]
code_1="print('hello world')"
code1_vec = model(tokenizer(code_1,return_tensors='pt')['input_ids'])[1]
code_2="s = 'hello world'"
code2_vec = model(tokenizer(code_2,return_tensors='pt')['input_ids'])[1]
code_3="hello world"
code3_vec = model(tokenizer(code_3,return_tensors='pt')['input_ids'])[1]
code_4 ="cout<<'hello word'<<endl"
code4_vec = model(tokenizer(code_4,return_tensors='pt')['input_ids'])[1]

code_vecs=torch.cat((code1_vec,code2_vec,code3_vec,code4_vec),0)
codes = [code_1,code_2,code_3,code_4]
scores=torch.einsum("ab,cb->ac",query_vec,code_vecs)
print(f"score of the vector multiplication before applying softmax {scores}")
scores=torch.softmax(scores,-1)
print("Query:",query)
for i in range(4):
    print("Code:",codes[i])
    print("Score:",scores[0][i].item())
    


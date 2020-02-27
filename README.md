# ConvLSTM
 Convolutional LSTM in Pytorch  
 Learning some craftsmanship from https://github.com/ndrplz/ConvLSTM_pytorch  
 Original paper: [Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting](https://papers.nips.cc/paper/5955-convolutional-lstm-network-a-machine-learning-approach-for-precipitation-nowcasting.pdf)
 
 ### Start
 ```
 clstm = ConvLSTM(input_size=(height, width),  
                  input_dim=img_channel,  
                  hidden_dim=3, (or a list, whose length is num_layers)  
                  kernel_size=3, (or a tuple, or a list)  
                  num_layers=4,  
                  stride=1,  
                  dilation=1,  
                  padding="SAME", (or a integer, or a tuple, or a list)  
                  bias=True,  
                  batch_first=True  
                  )  
```
### Attention
 Input contains two variables, **Input**'s shape is (batch_size, time_step, channel, height, width); and initial **state**, default None
 Output also contains two variables: **output** (batch_size, time_steps, channel, height, weight); and a tuple **state** contain **c** and **h**, shape  (batch_size, channel, height, weight)

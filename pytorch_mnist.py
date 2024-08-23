import numpy as np                    	# 수학 계산 관련 라이브러리
import matplotlib.pyplot as plt    	# 그래프 (및 그림) 표시를 위한 library

import torch                          	# 파이토치 관련 라이브러리
import torch.nn as nn                 	# neural network 관련 라이브러리

import torchvision.datasets as dset   	# 다양한 데이터셋 (MNIST, COCO, ...) 관련 라이브러리
import torchvision.transforms as transforms   # 입력/출력 데이터 형태, 크기 등을 조정
from torch.utils.data import DataLoader        # 데이터를 적절한 배치 사이즈로 load 할 수 있도록 함.

bs = 64                	# batch_size 는 대개 2^n 형태의 값으로 설정함.
learning_rate = 0.002      # 최적 학습률은 최적화 알고리즘 및 batch_size 에 따라 달라짐.
num_epochs = 20	# 학습 반복 횟수	

mnist_train = dset.MNIST(root="./", train=True,  transform=transforms.ToTensor(), download=True)
mnist_test  = dset.MNIST(root="./", train=False, transform=transforms.ToTensor(), download=True)

mnist_train = torch.utils.data.Subset(mnist_train, range(0, 6000))   # 0번부터 599번까지 600개
mnist_test  = torch.utils.data.Subset(mnist_test , range(0, 1000))   # 0번부터  99번까지 100개

train_loader = DataLoader(mnist_train, batch_size=bs, shuffle=True, drop_last=True)
test_loader  = DataLoader(mnist_test,  batch_size=bs, shuffle=True, drop_last=True)

class My_Model(nn.Module):
    def __init__(self):
        super().__init__()		# parent class 인 nn.Module의 생성자/초기화 함수를 상속함.
        self.layer = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),		# nn.Sigmoid() 함수를 사용할 수도 있음.
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 10),
        )
        
        # 가중치 초기화
        for m in self.layer.modules() :
            if isinstance(m, nn.Linear) :
                nn.init.kaiming_uniform_(m.weight)    
        
    def forward(self, x):			# x.shape = (bs, 1, 28, 28)
        in_data = x.view(bs, -1)		# in_data.shape = (bs, 784)
        out_data = self.layer(in_data)		# out_data.shape = (bs, 10)
        return out_data

model = My_Model()
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)   # Stochastic Gradient Descent
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")	# GPU 관련 내용은 생략함.



def train(train_loader):
    for (data, target) in train_loader:
        output = model(data)                        	# 순방향 전파를 통해 모델의 출력 계산
        loss = loss_func(output, target)      	# nn.CrossEntropyLoss() = [Softmax + CEL]
        optimizer.zero_grad()                    	# optimizer 기울기값 초기화
        loss.backward()                   		# 오차 역전파를 이용하여 기울기값 (편미분값) 계산
        optimizer.step()                   		# 파라미터 업데이트

def evaluate(test_loader):
    correct = 0			# 정답 수 초기화
    
    for (data, target) in test_loader:
        output = model(data)        		# output.shape = (bs, 10)
        output = output.detach().numpy()      	# output.shape = (bs, 10)
        pred = np.argmax(output, axis=1)      	# pred.shape = (bs,)
        target = target.detach().numpy()    	# target.shape = (bs,)

        correct += np.sum(pred == target)	# 정답 수 업데이트

    num_test_data = len(test_loader.dataset) - (len(test_loader.dataset) % bs)
    
    test_accuracy = 100. * correct / num_test_data
    return test_accuracy

train_acc_list = [10]		# 학습 데이터에 대한 정확도 저장을 위한 list (초기값: 10%)
test_acc_list = [10]		# 시험 데이터에 대한 정확도 저장을 위한 list (초기값: 10%)

for epoch in range(1, num_epochs + 1):
    train(train_loader)			# 학습 실시
    train_accuracy = evaluate(train_loader)	# 학습 정확도 계산
    test_accuracy = evaluate(test_loader)	# 테스트 정확도 계산

    train_acc_list.append(train_accuracy)	# 학습 데이터에 대한 정확도 리스트 갱신
    test_acc_list.append(test_accuracy)	# 시험 데이터에 대한 정확도 리스트 갱신

    print(f'Epoch:{epoch:2d}   Train Acc: {train_accuracy:6.2f}%   Test Acc: {test_accuracy:5.2f}%')

x = np.arange(len(train_acc_list))

plt.plot(x, train_acc_list, label='train acc')		# 학습 데이터 정확도 출력
plt.plot(x, test_acc_list, label='test acc', linestyle='--')	# 시험 데이터 정확도 출력

plt.xlabel("epoch_num")		# x축 제목 표시
plt.ylabel("accuracy (%)")	# y축 제목 표시

plt.ylim(0, 100.0)		# y축 범위 지정
plt.legend(loc='lower right')	# 범례 표시 및 위치 지정
plt.show()


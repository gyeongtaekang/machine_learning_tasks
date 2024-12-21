from torch import nn

class SRCNN(nn.Module):
    def __init__(self, num_channels=1):
        super(SRCNN, self).__init__()
        # SRCNN 구조에 따라 3개의 컨볼루션 계층을 정의
        # 첫 번째 Conv 계층: 9x9 커널, 출력 채널 수:64
        # 두 번째 Conv 계층: 1x1 커널, 출력 채널 수:32
        # 세 번째 Conv 계층: 5x5 커널, 출력 채널 수:1
        # 입력 채널 수는 1, 출력 채널 수도 1 (밝기 정보만 활용)
        
        # 첫 번째 계층
        # 커널 크기가 9이므로 padding=9//2 =4 적용하여 입력출력 크기 동일 유지
        self.conv1 = nn.Conv2d(in_channels=num_channels, out_channels=64, kernel_size=9, padding=4)
        self.relu1 = nn.ReLU(inplace=True)
        
        # 두 번째 계층
        # 커널 크기가 1이므로 padding=0 적용
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, padding=0)
        self.relu2 = nn.ReLU(inplace=True)
        
        # 세 번째 계층
        # 커널 크기가 5이므로 padding=5//2 =2 적용
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=5, padding=2)

    def forward(self, x):
        # 전방 계산: conv1 -> relu1 -> conv2 -> relu2 -> conv3 순서로 진행
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.conv3(x)
        return x

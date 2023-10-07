from torch import nn

class ClassificationNetwork(nn.Module):
    def __init__(self, input_shape: int, num_classes: int):
        super(self.__class__, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_shape, num_classes),
            nn.LogSoftmax(dim=-1)
        )
        
    def forward(self, input):       
        out = self.model(input)
        return out
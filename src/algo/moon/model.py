from src.models import *
from torch import nn 

class MoonTypeModel(nn.Module): 
    
    def __init__(self, model_name, num_channels=1, im_size=(28, 28), num_classes=10, hidden_size=200):
        
        super().__init__()
        self.use_lstm = False
        if model_name == 'mlp':
            self.features = MLP_header(im_size=im_size, hidden_size=hidden_size) 
            num_ftrs = hidden_size

        elif model_name in ['resnet18', 'resnet50', 'resnet34', 'resnet101']:
            
            if model_name == 'resnet18':
                model = ResNet18(num_channel=num_channels, num_classes=num_classes)
            elif model_name == 'resnet34':
                model = ResNet34(num_channel=num_channels, num_classes=num_classes)
            elif model_name == 'resnet50':
                model = ResNet50(num_channel=num_channels, num_classes=num_classes)
            elif model_name == 'resnet101':
                model = ResNet101(num_channel=num_channels, num_classes=num_classes)

            self.features = nn.Sequential(*list(model.resnet.children())[:-1])
            num_ftrs = model.resnet.fc.in_features

        elif model_name == 'cnn_text': 
            # self.use_lstm = True
            # self.features = LSTM_Header()
            self.features = CNN_Text_Header()

            num_ftrs = 96

        self.l3 = nn.Linear(num_ftrs, num_classes)

    def forward(self, x): 
        h = self.features(x)
        if not self.use_lstm:
            h = h.view(h.size(0), -1)

        y = self.l3(h)
        return h, h, y
    
def get_moon_model(model_name, num_channels=1, im_size=(28, 28), num_classes=10, hidden_size=200): 
    model = MoonTypeModel(model_name, num_channels, im_size, num_classes, hidden_size)
    return model
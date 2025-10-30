import torch
from torch.optim import SGD
import copy
from torch import nn 

from src.algo import NTD_Loss, MoonTypeModel

def train(net, trainloader, learning_rate: float, DEVICE, get_grad_norm = False, proximal_mu: float = None, epochs: int = 1, use_ntd_loss: bool = False, tau = None, beta = None):

    if use_ntd_loss:
        last_layer = list(net.modules())[-1]
        num_classes = last_layer.out_features
        criterion = NTD_Loss(num_classes=num_classes, tau=tau, beta=beta)
        global_net = copy.deepcopy(net)
        global_net.to(DEVICE)
    else: 
        criterion = nn.CrossEntropyLoss()

    if get_grad_norm: 
        full_grad = [torch.zeros_like(p) for p in net.parameters() if p.requires_grad]
        total_grad_samples = 0

    optimizer = SGD(net.parameters(), lr=learning_rate)
    net.train() 
    global_params = [p.detach().clone().to(DEVICE) for p in net.parameters()]
    
    loss_, acc = 0.0, 0
    
    for ep in range(epochs):
        running_loss, running_corrects, tot = 0.0, 0, 0

        for images, labels in trainloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = net(images)
            
            if use_ntd_loss:
                dg_outputs = global_net(images)
                loss = criterion(outputs, labels, dg_outputs)
            else: 
                loss = criterion(outputs, labels)
            if proximal_mu is not None:
                prox_term = 0.0
                for local_w, global_w in zip(net.parameters(), global_params):
                    prox_term += torch.square((local_w - global_w).norm(2))

                loss = loss + (proximal_mu / 2) * prox_term
            
            loss.backward()

            if get_grad_norm and ep == 0: 
                with torch.no_grad(): 
                    for i, p in enumerate(net.parameters()):
                        if p.grad is not None:
                            full_grad[i] += p.grad.detach() * images.size(0) 
                total_grad_samples += images.size(0)

            optimizer.step()
            
            preds = outputs.argmax(dim=1)
            tot += images.size(0)
            running_corrects += torch.sum(preds == labels).item()
            running_loss += loss.item() * images.size(0)

        loss = running_loss / tot
        accuracy = running_corrects / tot

        loss_ += running_loss
        acc += accuracy 
    
    loss_ /= epochs
    acc /= epochs

    if get_grad_norm:
        full_grad_ = [full_grad[i] / total_grad_samples for i in range(len(full_grad))]
        norm_grad = sum(g.norm()**2 for g in full_grad_).item()
        return {'loss': loss_, 'accuracy': acc, 'grad_norm': norm_grad}

    return {'loss': loss_, 'accuracy': acc}


def test(net, testloader):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net.to(device)
    net.eval()
    criterion = nn.CrossEntropyLoss()
    corrects, total_loss, tot = 0, 0.0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)

            if isinstance(net, MoonTypeModel): 
                _, _, outputs = net(images)
            else:
                outputs = net(images)

            loss = criterion(outputs, labels)

            if isinstance(net, MoonTypeModel):
                _, preds = torch.max(outputs.data, 1)
            else:  
                preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            corrects += torch.sum(preds == labels).item()
            total_loss += loss.item() * images.size(0)
            tot += images.size(0)

    total_loss /= tot
    accuracy = corrects / tot

    return total_loss, accuracy
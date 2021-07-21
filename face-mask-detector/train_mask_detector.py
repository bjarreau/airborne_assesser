import torch
import torchvision
from dataset import ImageClassificationDataset
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader, ConcatDataset

transform = transforms.Compose([
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),
    transforms.resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
mask_set = ImageClassificationDataset('./dataset/with_mask', transform)
naked_set = ImageClassificationDataset('./dataset/without_mask', transform)
dataset = ConcatDataset(mask_set, naked_set)
dataloader = DataLoader(dataset, batch_size = 32, shuffle=True)

samples, labels = iter(dataloader).next()

device = torch.device('cuda')
model = torchvision.models.resnet18(pretrained=True)
model.fc = torch.nn.Linear(512, 2)
model = model.to(device)
model = model.train()

train_loss=[]
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
for e in range(15):
    running_loss=0
    for images, labels in dataloader:
        inputs, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        img = model(inputs)
        loss = nn.NLLLoss(img, labels)
        running_loss+=loss
        loss.backward()
        optimizer.step()
    print("Epoch : {}:{}..".format(e+1, epochs), "Training Loss: {:.6f}".format(running_loss/len(dataloader)))
    train_loss.append(running_loss)

plt.plot(train_loss, label="Training Loss")
plt.show()

path = "C:/Users/bjarreau/git/mask_5_0/UI/model/mask_detect/torchmask.pth"
torch.save(model.state_dict(), path)

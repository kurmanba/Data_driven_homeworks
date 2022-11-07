import torch.nn.functional as F
from torch.utils.data import DataLoader
from ann_model import Autoencoder
from preprocess_data import *
from tqdm import tqdm
import json

mpl.use('macosx')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

network_parameters = json.load(open('/Users/alisher/IdeaProjects/Ann_practice/for_submission/simulation.json'))
torch.manual_seed(network_parameters["random_seed"])
model = Autoencoder(num_features=network_parameters["num_features"])
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=network_parameters["learning_rate"])
cost_t = []

custom_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset, test_dataset = prepare()
train_loader = DataLoader(dataset=train_dataset, batch_size=network_parameters["batch_size"], shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=network_parameters["batch_size"], shuffle=False)


for epoch in tqdm(range(network_parameters["num_epochs"])):
    for batch_index, (features, targets) in enumerate(train_loader):

        features = features.view(-1, 28 * 28).to(device)
        decoded = model(features)
        cost = F.mse_loss(decoded, features)
        optimizer.zero_grad()
        cost_t.append(float(cost))
        cost.backward()
        optimizer.step()

PATH = "/Users/alisher/IdeaProjects/Ann_practice/Scripts/Models/ann_{}".format(network_parameters["num_features"][1])
torch.save(model.state_dict(), PATH)

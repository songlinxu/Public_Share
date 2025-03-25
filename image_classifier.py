import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
import codecs

# Define the dataset class
class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.label_map = {}
        self.subset_map = {}  # Map each image to its respective subset
        self._load_dataset()
    
    def _load_dataset(self):
        sets = [d for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d))]
        label_id = 0
        
        for set_folder in sets:
            if set_folder == '.DS_Store': continue
            full_set_path = os.path.join(self.root_dir, set_folder)
            image_folder = os.path.join(full_set_path, 'images')
            index_folder = os.path.join(full_set_path, 'index')
            
            for img_name in os.listdir(image_folder):
                if img_name.endswith('.png'):
                    img_path = os.path.join(image_folder, img_name)
                    index_path = os.path.join(index_folder, img_name.replace('.png', '.txt'))
                    
                    with codecs.open(index_path, 'r', encoding='utf-8', errors='ignore') as f:
                        label = f.read().strip()
                        
                    if label not in self.label_map:
                        self.label_map[label] = label_id
                        label_id += 1
                    
                    self.image_paths.append(img_path)
                    self.labels.append(self.label_map[label])
                    self.subset_map[img_path] = set_folder  # Track which set this image belongs to
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# Define the CNN model
class ImageClassifier(nn.Module):
    def __init__(self, num_classes):
        super(ImageClassifier, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # Handle variable image sizes
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
    
    def forward(self, x):
        return self.model(x)

# Training and evaluation pipeline
class Trainer:
    def __init__(self, dataset, batch_size=16, epochs=10, learning_rate=0.001):
        self.dataset = dataset
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = learning_rate
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def train_and_evaluate(self):
        subsets = list(set(self.dataset.subset_map.values()))  # ['Set 1', 'Set 2', 'Set 3', 'Set 4']
        
        for fold, test_set in enumerate(subsets):
            print(f'Fold {fold + 1} - Testing on {test_set}')
            train_indices = [i for i, path in enumerate(self.dataset.image_paths) if self.dataset.subset_map[path] != test_set]
            test_indices = [i for i, path in enumerate(self.dataset.image_paths) if self.dataset.subset_map[path] == test_set]
            
            train_subset = Subset(self.dataset, train_indices)
            test_subset = Subset(self.dataset, test_indices)
            
            train_loader = DataLoader(train_subset, batch_size=self.batch_size, shuffle=True)
            test_loader = DataLoader(test_subset, batch_size=self.batch_size, shuffle=False)
            
            num_classes = len(set(self.dataset.labels))
            model = ImageClassifier(num_classes).to(self.device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=self.lr)
            
            for epoch in range(self.epochs):
                model.train()
                running_loss = 0.0
                for images, labels in train_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                
                print(f'Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}')
            
            self.evaluate(model, test_loader)
    
    def evaluate(self, model, test_loader):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        print(f'Test Accuracy: {100 * correct / total:.2f}%')

# Main execution
if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # Resize without extreme distortion
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    data_path = './DataSet-for-train/'
    dataset = ImageDataset(root_dir=data_path, transform=transform)
    trainer = Trainer(dataset, batch_size=16, epochs=20, learning_rate=0.0007)
    trainer.train_and_evaluate()


    

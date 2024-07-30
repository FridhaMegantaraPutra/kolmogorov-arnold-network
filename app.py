import torch
from torchvision import transforms
from PIL import Image
from kan.kan import *

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
                         )
])


model = KAN([32 * 32 * 3, 64, 3])
model.load_state_dict(torch.load('kan.pth'))
model.eval()


class_names = ['covid', 'normal', 'pneumonia']


def predict_image(image_path):

    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)

    print(f"Image shape: {image.shape}")

    if isinstance(model.layers[0], KANLinear):
        image = image.view(-1, 32 * 32 * 3)

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    return class_names[predicted.item()]


image_path = 'covid_1.jpeg'
prediction = predict_image(image_path)
print(f"Predicted class: {prediction}")

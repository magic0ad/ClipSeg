import torch
import requests
from models.clipseg import CLIPDensePredT
from PIL import Image
from torchvision import transforms
from matplotlib import pyplot as plt
import glob
import os

# load model
model = CLIPDensePredT(version='ViT-B/16', reduce_dim=64)
model.eval();

# non-strict, because we only stored decoder weights (not CLIP weights)
model.load_state_dict(torch.load('weights/rd64-uni.pth', map_location=torch.device('cpu')), strict=False);



## For waterbird folder

waterbird_dir = '/home/mila/j/jaewoo.lee/scratch/dataset/waterbirds/waterbird_complete95_forest2water2'
image_list = glob.glob(f'{waterbird_dir}/**/*.jpg', recursive=True)

for i in range(len(image_list)):

    folder_name = '/home/mila/j/jaewoo.lee/projects/text_prompt_sam/clipseg/waterbird_results/' + image_list[i].split('waterbird_complete95_forest2water2/')[-1].split('/')[0]
    file_name = '/home/mila/j/jaewoo.lee/projects/text_prompt_sam/clipseg/waterbird_results/' + image_list[i].split('waterbird_complete95_forest2water2/')[-1]
    os.makedirs(folder_name, exist_ok=True)

    input_image = Image.open(image_list[i])
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Resize((352, 352)),
    ])
    

    img = transform(input_image).unsqueeze(0)
    
    prompt = ['a bird']
    
    # predict
    with torch.no_grad():
        preds = model(img, prompt)[0]
    
    # visualize prediction
    output_image = torch.sigmoid(preds[0][0])
    
    # Save the result to 'output.png'
    plt.imsave(file_name, output_image.numpy(), cmap='gray')

    print('Saved: ', file_name)





#
## load and normalize image
#input_image = Image.open('/home/mila/j/jaewoo.lee/scratch/dataset/waterbirds/waterbird_complete95_forest2water2/001.Black_footed_Albatross/Black_Footed_Albatross_0023_796059.jpg')
#
## or load from URL...
## image_url = 'https://farm5.staticflickr.com/4141/4856248695_03475782dc_z.jpg'
## input_image = Image.open(requests.get(image_url, stream=True).raw)
#
#transform = transforms.Compose([
#    transforms.ToTensor(),
#    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#    transforms.Resize((352, 352)),
#])
#img = transform(input_image).unsqueeze(0)
#
#prompt = ['a bird']
#
## predict
#with torch.no_grad():
#    preds = model(img, prompt)[0]
#
## visualize prediction
#output_image = torch.sigmoid(preds[0][0])
#
## Save the result to 'output.png'
#plt.imsave('output.png', output_image.numpy(), cmap='gray')

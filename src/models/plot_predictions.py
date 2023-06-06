import matplotlib.pyplot as plt
import torch

from src.models.model import CNN_model, VGG_19_model
from src.data.dataset import HotDogDataset

import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

from tqdm import tqdm
import numpy as np

def main():
    
    ### Hardcoded constants!!!! #####
    saved_weights_path = '/zhome/39/c/174709/git/Hotdog_or_NotHotodog/models/testing_save_model.pt'
    data_path = '/dtu/datasets1/02514/hotdog_nothotdog'
    batch_size = 64
    seed = 7
    model_name = 'cnn'
    
    # Device
    device = torch.device('cuda'if torch.cuda.is_available() else 'cpu')
    print(f"Runs on: {device}")

    # Get the dataset
    # Get a set of test images
    test_transformation = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    testset_to_be_split = HotDogDataset(data_path, train=False, transform=test_transformation)

    generator1 = torch.Generator().manual_seed(seed)
    _, testset =  random_split(testset_to_be_split, [0.2, 0.8], generator=generator1)

    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    # Get the model
    model = CNN_model(3, 2, 224, 224)
    # model.load_state_dict(torch.load(saved_weights_path)['model'])
    model.to(device)
    model.eval()

    n_images = int(np.ceil(len(testset)/batch_size))
    bad_images_tsnr = []
    bad_probs = []
    bad_labels = []
    true_labels = []
    # Get the predictions
    for _, (data, target) in tqdm( enumerate(test_loader), total=len(test_loader) ):
        
        data, target = data.to(device), target.to(device)


        with torch.no_grad():
            output = model(data)

        predicted = output.argmax(1)



        # Find those that are not classified correct, alongside their probs
        b = predicted != target
        mask = torch.where(b, True, False)

        probs_of_wrong, _ = output[mask].max(1)

        # Get the max wrong prob
        max_output_probs, _ = output.max(1)
        b =(max_output_probs  >= probs_of_wrong.max() ) & (predicted != target)
        mask_p = torch.where(b, True, False)
        

        # Get the image tensor and the correct label
        bad_img = data[mask_p]

        bad_label = predicted[mask_p]
        true_label = target[mask_p]

        bad_images_tsnr.append(bad_img)
        bad_labels.append(bad_label)
        true_labels.append(true_label)
        bad_probs.append(probs_of_wrong.max())

    
    n_to_plot = 4
    # Now pick the worst n images, based on the probs.
    idx = []
    probs_sorted = sorted(bad_probs, reverse=True)

    for i in range(n_to_plot):
        idx.append(bad_probs.index(probs_sorted[i]))

    # Now plot these indices

    fig, ax = plt.subplots(1,n_to_plot)
    fig.subplots_adjust(hspace=0.5)
    for i in range(len(idx)):
        img_ = bad_images_tsnr[i].squeeze()
        img_ = torch.permute(img_, (1,2,0)).cpu().numpy()

        ax[i].imshow(img_)
        ax[i].axis(False)

        a = bad_labels[i].cpu().numpy()
        bad_pred = testset_to_be_split.id_to_label[bad_labels[i].cpu().numpy()[0]]
        true_pred = testset_to_be_split.id_to_label[true_labels[i].cpu().numpy()[0]]
        # bad_pred = testset_to_be_split.labels[bad_labels[i].cpu().numpy()]
        # true_pred = testset_to_be_split.labels[true_labels[i].cpu().numpy()]
        ax[i].set_title(f"Predicted: {bad_pred}\nTrue: {true_pred}")

    plt.savefig('test_plot.pdf')
    plt.show()

    

    

    


if __name__=='__main__':
    main()
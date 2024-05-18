# Description:
# This file should be used for performing inference on a network
# Usage: inference.py <path_2_dataset> <path_2_model> (<int_number_of_samples>)
import sys
import numpy as np
from dataset import SampleDataset
import torch
from skimage import io
from skimage.util import img_as_ubyte
import albumentations as A
from albumentations.pytorch import ToTensorV2     
from training import custom_accuracy_score
from dataset import extract_seg_dataset
import matplotlib.pyplot as plt

transforms = A.Compose([
                         # preprocessing
                         A.SmallestMaxSize (128),
                         A.CenterCrop(128, 128),
                         A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                         ToTensorV2(),
                        ]   
                    )
loss_fn = torch.nn.CrossEntropyLoss()
    
# sample function for performing inference for a whole dataset
def infer_all(net, batch_size, dataloader, device):
    # do not calculate the gradients
    net.eval()
    counter = 0
    with torch.no_grad():
        for i, (xb, yb) in enumerate(dataloader):
            xb, yb = xb.to(device), yb
            pred = net(xb).to(dtype = torch.float32)
            pred = np.array(pred.cpu().detach())
            pred = np.argmax(np.array(pred), axis=1)
            for i in range(pred.shape[0]):
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
                ax1.imshow(pred[i])
                ax1.set_title("Predicted")
                ax2.imshow(yb[i])
                ax2.set_title("Mask")
                plt.savefig(f"./output_predictions/prediction_{counter}.png")
                plt.clf()
                counter+=1
    return


# declaration for this function should not be changed
def inference(dataset_path, model_path, n_samples):
    """
    inference(dataset_path, model_path='model.pt') performs inference on the given dataset;
    if n_samples is not passed or <= 0, the predictions are performed for all data samples at the dataset_path
    if  n_samples=N, where N is an int positive number, then only N first predictions are performed
    saves:
    - predictions to 'output_predictions' folder

    Parameters:
    - dataset_path (string): path to a dataset
    - model_path (string): path to a model
    - n_samples (int): optional parameter, number of predictions to perform

    Returns:
    - None
    """
    # Check for available GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Computing with {}!'.format(device))

    # loading the model
    model = torch.load(model_path)
    model.eval()

    batch_size = 4
    df = extract_seg_dataset(dataset_path)
    print(df.shape)
    ds = SampleDataset(df, transforms)
    testloader = torch.utils.data.DataLoader(ds,
                      batch_size=batch_size,
                      shuffle=False,
                      num_workers=1)


    # if n_samples <= 0 -> perform predictions all data samples at the dataset_path
    if n_samples <= 0:
        infer_all(model, batch_size, testloader, device)
    else:
        model.eval()
        with torch.no_grad():
        # perform predictions only for the first n_samples images
            for i in range(0, n_samples):
                # generate a random image and save t to output_predictions
                sample, mask = ds[i]
                mask = mask.unsqueeze(0)
                out = model(sample.unsqueeze(0).to(device))
                out = np.array(out.cpu().detach())
    
                out = np.argmax(np.array(out), axis=1)
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
                ax1.imshow(out.squeeze(0))
                ax1.set_title("Predicted")
                ax2.imshow(mask.squeeze(0))
                ax2.set_title("Mask")
                plt.savefig(f"./output_predictions/prediction_{i}.png")
                plt.clf()

    return


def get_arguments():
    if len(sys.argv) == 3:
        dataset_path = sys.argv[1]
        model_path = sys.argv[2]
        number_of_samples = 0
    elif len(sys.argv) == 4:
        try:
            dataset_path = sys.argv[1]
            model_path = sys.argv[2]
            number_of_samples = int(sys.argv[3])
        except Exception as e:
            print(e)
            sys.exit(1)
    else:
        print("Usage: inference.py <path_2_dataset> <path_2_model> (<int_number_of_samples>)")
        sys.exit(1)

    return dataset_path, model_path, number_of_samples


if __name__ == "__main__":
    path_2_dataset, path_2_model, n_samples_2_predict = get_arguments()
    inference(path_2_dataset, path_2_model, n_samples_2_predict)

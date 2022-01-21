import argparse
from tqdm import tqdm
import cv2
import torch
import random
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# from petroleum import PetroleumDataset, collate_skip_empty, colors_pre_class
# from petroleum_dataset import PetroleumDataset, collate_skip_empty, colors_per_class
from petroleum_dataset import PetroleumDataset, collate_skip_empty, colors_per_class
from resnet import ResNet101
import torchvision.models as modes


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']  # 提取网络结构
    model.load_state_dict(checkpoint['model_state_dict'])  # 加载网络权重参数
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.eval()
    return model

def fix_random_seeds():
    seed = 10
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_features(dataset, batch, num_images):
    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    # initialize our implementation of ResNet
    # model = modes.vgg16(pretrained=False)
    model = ResNet101(pretrained=True)
    # model = load_checkpoint(trained_model)
    model.eval()
    model.to(device)

    # read the dataset and initialize the data loader
    dataset = PetroleumDataset(dataset, num_images)
    print('======================')
    print(dataset)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch, collate_fn=collate_skip_empty, shuffle=True)
    # print(dataloader)
    # print('++++++++++++++++++++++++')

    # we'll store the features as NumPy array of size num_images x feature_size
    features = None

    # we'll also store the image labels and paths to visualize them later
    labels = []
    image_paths = []

    for batch in tqdm(dataloader, desc='Running the model inference'):
        # print(batch)
        images = batch['image'].to(device)
        labels += batch['label']
        image_paths += batch['image_path']
        # print('+++++++++++=====================')
        # print(images)
        # print(labels)
        # print(set(labels))
        # print(image_paths)
        # print('==================+++++++++++++++++++')

        with torch.no_grad():
            output = model.forward(images)

        current_features = output.cpu().numpy()
        # print('=================================')
        # print(current_features)
        if features is not None:
            features = np.concatenate((features, current_features))
        else:
            features = current_features

    return features, labels, image_paths


# scale and move the coordinates so they fit [0; 1] range
def scale_to_01_range(x):
    # compute the distribution range
    value_range = (np.max(x) - np.min(x))

    # move the distribution so that it starts from zero
    # by extracting the minimal value from all its values
    # print('rrrrrrrrrrrrrrrrrrrrrrrrrr')
    # print(x)
    starts_from_zero = x - np.min(x)
    # print(starts_from_zero)

    # make the distribution fit [0; 1] by dividing by its range
    return starts_from_zero / value_range


def scale_image(image, max_image_size):
    image_height, image_width, _ = image.shape

    scale = max(1, image_width / max_image_size, image_height / max_image_size)
    image_width = int(image_width / scale)
    image_height = int(image_height / scale)

    # 对图片调整大小
    image = cv2.resize(image, (image_width, image_height))
    return image


def draw_rectangle_by_class(image, label):
    image_height, image_width, _ = image.shape

    # get the color corresponding to image class
    color = colors_per_class[label]
    image = cv2.rectangle(image, (0, 0), (image_width - 1, image_height - 1), color=color, thickness=5)

    return image

# 计算绘图坐标
def compute_plot_coordinates(image, x, y, image_centers_area_size, offset):
    image_height, image_width, _ = image.shape

    # compute the image center coordinates on the plot
    center_x = int(image_centers_area_size * x) + offset

    # in matplotlib, the y axis is directed upward
    # to have the same here, we need to mirror the y coordinate
    center_y = int(image_centers_area_size * (1 - y)) + offset

    # knowing the image center, compute the coordinates of the top left and bottom right corner
    tl_x = center_x - int(image_width / 2)
    tl_y = center_y - int(image_height / 2)

    br_x = tl_x + image_width
    br_y = tl_y + image_height

    return tl_x, tl_y, br_x, br_y

# 可视化tsne图像
def visualize_tsne_images(tx, ty, images, labels, plot_size=1000, max_image_size=100):
    # we'll put the image centers in the central area of the plot
    # and use offsets to make sure the images fit the plot
    offset = max_image_size // 2
    image_centers_area_size = plot_size - 2 * offset

    tsne_plot = 255 * np.ones((plot_size, plot_size, 3), np.uint8)

    # now we'll put a small copy of every image to its corresponding T-SNE coordinate
    for image_path, label, x, y in tqdm(
            zip(images, labels, tx, ty),
            desc='Building the T-SNE plot',
            total=len(images)
    ):
        image = cv2.imread(image_path)

        # scale the image to put it to the plot
        image = scale_image(image, max_image_size)

        # draw a rectangle with a color corresponding to the image class
        image = draw_rectangle_by_class(image, label)

        # compute the coordinates of the image on the scaled plot visualization
        tl_x, tl_y, br_x, br_y = compute_plot_coordinates(image, x, y, image_centers_area_size, offset)

        # put the image to its TSNE coordinates using numpy subarray indices
        tsne_plot[tl_y:br_y, tl_x:br_x, :] = image

    plt.imshow(tsne_plot[:, :, ::-1])
    plt.show()


def visualize_tsne_points(tx, ty, labels):
    # initialize matplotlib plot
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # for every class, we'll add a scatter plot separately
    for label in colors_per_class:
        # find the samples of the current class in the data
        indices = [i for i, l in enumerate(labels) if l == label]

        # extract the coordinates of the points of this class only
        current_tx = np.take(tx, indices)
        current_ty = np.take(ty, indices)

        # convert the class color to matplotlib format:
        # BGR -> RGB, divide by 255, convert to np.array
        color = np.array([colors_per_class[label][::-1]], dtype=np.float) / 255

        # add a scatter plot with the correponding color and label
        ax.scatter(current_tx, current_ty, c=color, label=label)

    # build a legend using the labels we set previously
    ax.legend(loc='best')

    # finally, show the plot
    plt.show()


def visualize_tsne(tsne, images, labels, plot_size=1000, max_image_size=100):
    # extract x and y coordinates representing the positions of the images on T-SNE plot
    # tsne是二维数组
    tx = tsne[:, 0]
    # print(tx)

    ty = tsne[:, 1]
    # print(ty)
    # print(np.min(ty))
    # print(np.max(ty))
    # print('++++++++++=======++++++++')

    # scale and move the coordinates so they fit [0; 1] range
    tx = scale_to_01_range(tx)
    ty = scale_to_01_range(ty)

    # visualize the plot: samples as colored points
    visualize_tsne_points(tx, ty, labels)

    # visualize the plot: samples as images
    visualize_tsne_images(tx, ty, images, labels, plot_size=plot_size, max_image_size=max_image_size)


def main():
    parser = argparse.ArgumentParser()
    # D:\中石油工作\image_classfication-master\image_classfication-master\picture        ../archive/raw-img
    parser.add_argument('--path', type=str, default='D:\\uu\\pytorch_classification-master\\data\\data_uu')   
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--num_images', type=int, default=500)
    # parser.add_argument('--trained_model', type=str, default="D:\\中石油工作\\pytorch_classification-master\\weightsresnext101_32x32d\\epoch_oil_20.pth")
    args = parser.parse_args()

    fix_random_seeds()
    # print(args.num_images)
    features, labels, image_paths = get_features(
        dataset=args.path,
        batch=args.batch,
        num_images=args.num_images,
        # trained_model=args.trained_model,
    )
    # print('===========================+')
    # print(features)
    # print(labels)
    # print(len(labels))
    # print(image_paths)
    tsne = TSNE(n_components=2).fit_transform(features)
    # print(tsne)
    # print('++++++++++++++++++++++++++=')

    visualize_tsne(tsne, image_paths, labels)

if __name__ == '__main__':
    main()
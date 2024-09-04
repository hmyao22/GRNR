from torch.nn import functional as F
from torchvision import transforms
import numpy as np
import torch
import cv2
import os
from PIL import Image
import pytorch_lightning as pl
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

# imagenet
mean_train = [0.485, 0.456, 0.406]
std_train = [0.229, 0.224, 0.225]


class RNR(pl.LightningModule):
    def __init__(self):
        super(RNR, self).__init__()

        self.init_features()
        self.image_size = 320

        self.beltas = [torch.rand([int(self.image_size / 8) * int(self.image_size / 8), 512, 9]),
                       torch.rand([int(self.image_size / 16) * int(self.image_size / 16), 1024, 9]),
                       torch.rand([int(self.image_size / 32) * int(self.image_size / 32), 2048, 9]),
        ]

        def hook_t(module, input, output):
            self.features.append(output)


        self.model = torch.hub.load('pytorch/vision:v0.9.0', 'wide_resnet50_2', pretrained=True)
        # self.model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet18', pretrained=True)


        for param in self.model.parameters():
            param.requires_grad = False



        self.model.layer2[-1].register_forward_hook(hook_t)
        self.model.layer3[-1].register_forward_hook(hook_t)

        self.criterion = torch.nn.MSELoss(reduction='sum')

        self.data_transforms = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size), Image.ANTIALIAS),
            transforms.ToTensor(),
            transforms.CenterCrop(self.image_size),
            transforms.Normalize(mean=mean_train,
                                 std=std_train)])

        self.inv_normalize = transforms.Normalize(mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.255],
                                                  std=[1 / 0.229, 1 / 0.224, 1 / 0.255])



    def init_features(self):
        self.features = []

    def init_neighbors(self, index=0):
        scale = 2**(3+index)
        batch, height, width = 1,int(self.image_size/scale), int(self.image_size/scale)
        neighbor_indices = np.array([
            [(-1, -1), (-1, 0), (-1, 1)],
            [(0, -1), (0, 0), (0, 1)],
            [(1, -1), (1, 0), (1, 1)]
        ])
        i, j = np.indices((height, width))
        i = i.reshape(-1, 1, 1)
        j = j.reshape(-1, 1, 1)
        return np.concatenate((i, j), axis=2)[:, np.newaxis] + neighbor_indices[np.newaxis, :, :, :]

    def get_neighbors(self, matrix, index=0):
        scale = 2**(3+index)
        indices = self.init_neighbors(index)
        padded_matrix = np.pad(matrix, ((0, 0), (0, 0), (1, 1), (1, 1)), mode='reflect')
        neighbors = padded_matrix[:, :, indices[:, :, :, 0] + 1, indices[:, :, :, 1] + 1]
        neighbors = np.transpose(neighbors, (0, 2, 1, 3, 4))
        neighbors = neighbors.reshape(1 * int(self.image_size/scale) * int(self.image_size/scale), -1, 9)
        return neighbors


    def interpolate_scoremap(self, heatMap, cut=3):
        imgshape = self.image_size
        blank = torch.ones_like(heatMap[:, :]) * heatMap[:, :].min()
        blank[cut:heatMap.shape[1] - cut, cut:heatMap.shape[1] - cut] = heatMap[cut:heatMap.shape[1] - cut,
                                                                    cut:heatMap.shape[1] - cut]
        return F.interpolate(blank[:, :].unsqueeze(0).unsqueeze(0), size=imgshape, mode='bilinear', align_corners=False)



    def cal_ano_map(self, image, filter_number=40, lamda=0):
        self.model.eval().cuda()
        self.init_features()
        _ = self.model(image)

        embeddings = []
        for feature in self.features:
            m = torch.nn.AvgPool2d(3, 1, 1)
            embeddings.append(m(feature))

        anomaly_maps=[]
        for i, embedding_temp in enumerate(embeddings):
            scale = 2 ** (3 + i)
            embedding = embedding_temp.permute(0, 2, 3, 1).contiguous()
            embedding = embedding.view(-1, embedding_temp.shape[1])
            self.embedding_list = embedding


            ######## feature filter ######
            distances = torch.cdist(self.embedding_list, self.embedding_list).cpu().numpy()
            np.fill_diagonal(distances, np.inf)
            average_distances = np.mean(distances, axis=1)
            top_k_indices = np.argsort(average_distances.flatten())[:filter_number]
            embedding_list_filtered = self.embedding_list[top_k_indices]

            global_mu = torch.tile(torch.unsqueeze(embedding_list_filtered, 0),
                                   [int(self.image_size / scale) * int(self.image_size / scale), 1, 1]).cuda()


            ###### neighbor sample #####

            neighbor_feature = torch.from_numpy(self.get_neighbors(embedding_temp.cpu(), index=i))
            neighbor_feature = neighbor_feature + self.beltas[i] * 0.001
            Sc = torch.transpose(neighbor_feature, 1, 2)
            Sc_local = torch.cat((Sc[:, :4, :], Sc[:, 5:, :]), dim=1).cuda()


            ###### regression #####
            Q = torch.unsqueeze(self.embedding_list, 1).cuda()

            temp2 = 0

            temp = torch.bmm(Sc_local, torch.transpose(Sc_local, 2, 1)) + \
                   lamda * global_mu.shape[1]*torch.bmm(Sc_local, torch.transpose(Sc_local, 2, 1))
            for j in range(global_mu.shape[1]):
                mu = global_mu[:, j, :].unsqueeze(1)
                temp2 += lamda * torch.bmm(mu, torch.transpose(Sc_local, 2, 1))


            W = torch.bmm((torch.bmm(Q, torch.transpose(Sc_local, 2, 1)) + temp2), torch.linalg.inv(temp))
            Q_hat = torch.bmm(W, Sc_local)

            score_patches = torch.abs(Q - Q_hat)

            score_patches = torch.norm(score_patches, dim=2)

            score_patches = score_patches.T
            anomaly_map = score_patches.reshape(int(self.image_size / scale), int(self.image_size / scale))
            anomaly_map = self.interpolate_scoremap(anomaly_map, cut=3-i)
            anomaly_map = gaussian_filter(anomaly_map.squeeze().cpu().detach().numpy(), sigma=3)
            cut_surrounding = 32
            anomaly_map = anomaly_map[cut_surrounding:self.image_size - cut_surrounding,
                          cut_surrounding:self.image_size - cut_surrounding]

            anomaly_maps.append(anomaly_map)

        return anomaly_maps[0], anomaly_maps[1]



if __name__ == '__main__':
    print(torch.__version__)
    torch.backends.cudnn.benchmark = True
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image = Image.open(r'E:\YHM\dataset\Zero-texture\Mvtec-Texture\carpet\test\hole\008.png').convert('RGB')
    model = RNR()
    image_tensor = torch.unsqueeze(model.data_transforms(image), 0).cuda()

    result = model.cal_ano_map(image_tensor, filter_number=40,  lamda=5)


    shown_image = cv2.resize(np.array(image), [320, 320])
    shown_image = shown_image[32:-32, 32:-32]
    shown_image = cv2.cvtColor(shown_image, cv2.COLOR_BGR2RGB)

    cv2.imwrite('shown_image.png', shown_image)
    score_map = result[0] * result[1]
    score_map =255*(score_map-score_map.min())/(score_map.max()-score_map.min())

    plt.figure(figsize=(12,12))
    plt.subplot(141)
    plt.imshow(shown_image)
    plt.subplot(142)
    plt.imshow(result[0], cmap='jet')
    plt.title('Scale 1 result')
    plt.subplot(143)
    plt.imshow(result[1], cmap='jet')
    plt.title('Scale 2 result')
    plt.subplot(144)
    plt.imshow(result[0]*result[1], cmap='jet')
    plt.title('Fuse result')
    plt.show()
























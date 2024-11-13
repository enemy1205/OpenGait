import torch.nn as nn
import numpy as np
from .base import BaseLoss
import torch
import cv2

import torch
import torch.nn.functional as F
import torchvision.transforms as T

class Single_Frame_EdgeDetector:
    def __init__(self):
        pass

    def get_edge(self, img_tensor):
        # 确保图像是二值的
        binary_tensor = (img_tensor > 0.5).float() * 255

        # 查找轮廓
        contours = self.find_contours(binary_tensor)

        # 创建输出图像
        output_tensor = torch.zeros_like(binary_tensor)

        # 绘制轮廓
        for contour in contours:
            output_tensor = self.draw_contour(output_tensor, contour)

        return output_tensor

    def find_contours(self, binary_tensor):
        # 使用 Sobel 滤波器来检测边缘
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).cuda()
        sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).cuda()

        edges_x = F.conv2d(binary_tensor.unsqueeze(0), sobel_x, padding=1)
        edges_y = F.conv2d(binary_tensor.unsqueeze(0), sobel_y, padding=1)

        edges = torch.sqrt(edges_x.pow(2) + edges_y.pow(2)).squeeze(0)

        # 阈值处理，找到边缘点
        edges = (edges > 0.1).float() * 255

        # 找到轮廓点
        contours = []
        for y in range(edges.shape[1]):
            for x in range(edges.shape[2]):
                if edges[0, y, x] > 0:
                    contours.append((y, x))

        return contours

    def draw_contour(self, output_tensor, contour):
        y, x = contour
        output_tensor[0, y, x] = 255
        return output_tensor
    
class EdgeDetector:
    def __init__(self):
        pass

    def get_edge(self, img_tensor):
        # 确保图像是二值的
        binary_tensor = (img_tensor > 0.5).float() * 255

        # 查找轮廓
        contours = self.find_contours(binary_tensor)

        # 创建输出图像
        output_tensor = torch.zeros_like(binary_tensor)

        # 绘制轮廓
        # for i in range(binary_tensor.shape[0]):
        #     for contour in contours[i]:
        #         output_tensor = self.draw_contour(output_tensor, i, contour)
        # 绘制轮廓
        self.draw_contours(output_tensor, contours)

        return output_tensor

    def find_contours(self, binary_tensor):
        # 使用 Sobel 滤波器来检测边缘
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).cuda()
        sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).cuda()

        edges_x = F.conv2d(binary_tensor, sobel_x, padding=1)
        edges_y = F.conv2d(binary_tensor, sobel_y, padding=1)

        edges = torch.sqrt(edges_x.pow(2) + edges_y.pow(2))

        # 阈值处理，找到边缘点
        edges = (edges > 0.1).float() * 255

        # 找到轮廓点
        contours = []
        for i in range(binary_tensor.shape[0]):
            non_zero_indices = torch.nonzero(edges[i, 0])
            contours.append(non_zero_indices.tolist())

        return contours

    # def draw_contour(self, output_tensor, index, contour):
    #     y, x = contour
    #     output_tensor[index, 0, y, x] = 255
    #     return output_tensor
    def draw_contours(self, output_tensor, contours):
        for i, contour in enumerate(contours):
            if len(contour) > 0:
                # 将 contour 转换为张量
                contour_tensor = torch.tensor(contour, device=output_tensor.device)
                y, x = contour_tensor[:, 0], contour_tensor[:, 1]
                output_tensor[i, 0, y, x] = 255



class EdgeLoss(BaseLoss):
    def __init__(self,loss_term_weight=1.0):
        super(EdgeLoss, self).__init__(loss_term_weight)
        # self.l1_loss = nn.L1Loss()
        # self.l2_loss = nn.MSELoss()
        self.huber_loss = nn.HuberLoss()
        self.edge_det = EdgeDetector()

    def edge_loss(self, output, target):
        output_edge = self.edge_det.get_edge(output.detach())
        gt_edge = self.edge_det.get_edge(target.detach())
        # cv2.imwrite('test1.png',output_edge.detach().cpu().numpy().astype(np.uint8).squeeze(0))
        # cv2.imwrite('test2.png',gt_edge.detach().cpu().numpy().astype(np.uint8).squeeze(0))
        # loss = self.l1_loss(output_edge, gt_edge)
        loss = self.huber_loss(output_edge, gt_edge)
        return loss, output_edge, gt_edge

    def forward(self, pred_silt_video, gt_silt_video,b_s):
        mean_image_loss = []
        # output_edges = []
        # target_edges = []
        
        # for batch_idx in range(pred_silt_video.size(0)):
        #     edges_o = []
        #     edges_t = []
        #     for frame_idx in range(pred_silt_video.size(1)):
        #         loss, output_edge, target_edge = self.edge_loss(
        #             gt_silt_video[batch_idx, frame_idx],
        #             pred_silt_video[batch_idx, frame_idx]
        #         )
        #         mean_image_loss.append(loss)
        #         edges_o.append(output_edge)
        #         edges_t.append(target_edge)
        #     output_edges.append(torch.cat(edges_o, dim=0))
        #     target_edges.append(torch.cat(edges_t, dim=0))

        # mean_image_loss = torch.stack(mean_image_loss, dim=0).mean(dim=0)
        
        loss, output_edges, target_edges = self.edge_loss(gt_silt_video, pred_silt_video)
        mean_image_loss = loss/b_s
        # self.current_output_edges = output_edges
        # self.current_target_edges = target_edges
        
        self.info.update({
            'image_edge_loss':mean_image_loss.detach().clone()
        })
        return mean_image_loss,self.info


if __name__ == '__main__':
    image_path = '/home/sp/datasets/CASIA_B/Yolov8Seg/GaitDatasetB-silh_occ_random_r40/003/cl-01/036/003-cl-01-036-054.png'  # 替换为你的图像路径
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    transform = T.Compose([
    T.ToTensor()  # 将图像转换为 [0, 1] 范围内的浮点数
    ])
    img_tensor = transform(image).cuda()
    detector = EdgeDetector()
    edge_tensor = detector.get_edge(img_tensor)
    edge_image = edge_tensor.squeeze(0).cpu().numpy().astype(np.uint8)
    cv2.imwrite('Edge_Image.png', edge_image)
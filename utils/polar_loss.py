# 来自github-claude
# 修改V4的代码，使其输入是法线图和aolp,dolp，输出是法线图的方位角和偏振aolp和dolp的方位角之间的loss的值，不用管dolp<0.3的区域。
# 0.3<dolp<0.7的区域要考虑邻域一致性约束，dolp>0.7的区域只用考虑pi的歧义性，即只用考虑aolp_angle % (2 * np.pi)和aolp_angle + np.pi，也要考虑邻域一致性约束。
# 必须是可微分的,且在GPU上进行运算

import torch
import torch.nn as nn
import torch.nn.functional as F


class AzimuthLoss(nn.Module):
    def __init__(self, window_size=3):
        super(AzimuthLoss, self).__init__()
        self.window_size = window_size
        # Create Gaussian kernel for neighborhood consistency
        kernel = self._create_gaussian_kernel(window_size)
        self.register_buffer('kernel', kernel)

    def _create_gaussian_kernel(self, window_size, sigma=1.0):
        """Creates a 2D Gaussian kernel"""
        coords = torch.arange(window_size).float() - window_size // 2
        x, y = torch.meshgrid(coords, coords)
        kernel = torch.exp(-(x.pow(2) + y.pow(2)) / (2 * sigma ** 2))
        kernel = kernel / kernel.sum()
        return kernel.view(1, 1, window_size, window_size)

    def _angular_difference(self, angle1, angle2):
        """Computes the minimum angular difference considering periodicity"""
        diff = torch.abs(angle1 - angle2)
        return torch.min(diff, 2 * torch.pi - diff)

    def _neighborhood_consistency(self, angles, dolp_mask):
        """Computes neighborhood consistency using convolution"""
        # Convert angles to complex form for periodic handling
        complex_angles = torch.exp(2j * angles.float())
        real = complex_angles.real
        imag = complex_angles.imag

        # Apply convolution separately to real and imaginary parts
        avg_real = F.conv2d(real.unsqueeze(1), self.kernel, padding=self.window_size // 2)
        avg_imag = F.conv2d(imag.unsqueeze(1), self.kernel, padding=self.window_size // 2)

        # Calculate consistency score (magnitude of the average complex vector)
        consistency = torch.sqrt(avg_real.pow(2) + avg_imag.pow(2)).squeeze(1)

        # Only apply consistency where dolp_mask is True
        consistency = consistency * dolp_mask.float()
        return consistency

    def forward(self, normal_azimuth, aolp_angle, dolp):
        """
        Compute the loss between normal map azimuth and AOLP angles

        Args:
            normal_azimuth: Tensor of shape [B, H, W] - azimuth angles from normal map
            aolp_angle: Tensor of shape [B, H, W] - AOLP angles
            dolp: Tensor of shape [B, H, W] - DOLP values

        Returns:
            total_loss: Scalar tensor representing the total loss
        """
        batch_size = normal_azimuth.size(0)
        device = normal_azimuth.device

        # Create masks for different DOLP regions
        high_dolp_mask = dolp > 0.7
        mid_dolp_mask = (dolp >= 0.3) & (dolp <= 0.7)

        # Initialize loss components
        angle_loss = torch.zeros(1, device=device)
        consistency_loss = torch.zeros(1, device=device)

        # For high DOLP regions (ambiguity pi)
        if high_dolp_mask.any():
            # Consider both original angle and angle + pi
            angle_diff1 = self._angular_difference(normal_azimuth, aolp_angle)
            angle_diff2 = self._angular_difference(normal_azimuth, aolp_angle + torch.pi)

            # Take minimum of the two differences
            min_diff = torch.min(angle_diff1, angle_diff2)
            angle_loss += (min_diff * high_dolp_mask.float()).mean()

            # Add neighborhood consistency for high DOLP
            consistency_loss += (1 - self._neighborhood_consistency(aolp_angle, high_dolp_mask)).mean()

        # For mid DOLP regions
        if mid_dolp_mask.any():
            # Regular angle difference
            mid_diff = self._angular_difference(normal_azimuth, aolp_angle)
            angle_loss += (mid_diff * mid_dolp_mask.float()).mean()

            # Stronger neighborhood consistency for mid DOLP
            consistency_loss += 2 * (1 - self._neighborhood_consistency(aolp_angle, mid_dolp_mask)).mean()

        # Combine losses
        total_loss = angle_loss + consistency_loss
        return total_loss

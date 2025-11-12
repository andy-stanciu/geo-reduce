import torch
import torch.nn as nn

class HaversineLoss(nn.Module):
    """
    Haversine loss function for geographic coordinate prediction.
    
    Calculates great-circle distance on Earth's surface in kilometers.
    Fully differentiable and suitable for gradient-based optimization.
    """
    
    def __init__(self, earth_radius_km=6371.0):
        super().__init__()
        self.earth_radius = earth_radius_km
    
    def forward(self, pred, target):
        """
        Args:
            pred: [batch, 2] tensor with predicted [lat, lon] (in degrees or normalized)
            target: [batch, 2] tensor with ground truth [lat, lon]
        
        Returns:
            loss: Mean Haversine distance in kilometers
        """
        # Extract lat/lon (assume they're already denormalized to degrees)
        lat1 = pred[:, 0]
        lon1 = pred[:, 1]
        lat2 = target[:, 0]
        lon2 = target[:, 1]
        
        # Convert to radians
        lat1 = torch.deg2rad(lat1)
        lon1 = torch.deg2rad(lon1)
        lat2 = torch.deg2rad(lat2)
        lon2 = torch.deg2rad(lon2)
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = torch.sin(dlat / 2) ** 2 + \
            torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2) ** 2
        
        # Clamp to avoid numerical errors with arcsin
        a = torch.clamp(a, 0.0, 1.0)
        
        c = 2 * torch.asin(torch.sqrt(a))
        
        # Distance in kilometers
        distance = self.earth_radius * c
        
        # Return mean distance as loss
        return distance.mean()


class HaversineLossNormalized(nn.Module):
    """
    Haversine loss for NORMALIZED coordinates [0, 1].
    
    Automatically denormalizes before computing distance.
    """
    
    def __init__(self, lat_min, lat_max, 
                 lon_min, lon_max, 
                 earth_radius_km=6371.0):
        super().__init__()
        self.lat_min = lat_min
        self.lat_max = lat_max
        self.lon_min = lon_min
        self.lon_max = lon_max
        self.earth_radius = earth_radius_km
    
    def denormalize(self, coords):
        """Denormalize from [0, 1] to actual lat/lon"""
        lat_norm = coords[:, 0]
        lon_norm = coords[:, 1]
        
        lat = lat_norm * (self.lat_max - self.lat_min) + self.lat_min
        lon = lon_norm * (self.lon_max - self.lon_min) + self.lon_min
        
        return lat, lon
    
    def forward(self, pred, target):
        """
        Args:
            pred: [batch, 2] normalized predictions [0, 1]
            target: [batch, 2] normalized ground truth [0, 1]
        """
        # Denormalize
        pred_lat, pred_lon = self.denormalize(pred)
        target_lat, target_lon = self.denormalize(target)
        
        # Convert to radians
        lat1 = torch.deg2rad(pred_lat)
        lon1 = torch.deg2rad(pred_lon)
        lat2 = torch.deg2rad(target_lat)
        lon2 = torch.deg2rad(target_lon)
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = torch.sin(dlat / 2) ** 2 + \
            torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2) ** 2
        
        a = torch.clamp(a, 0.0, 1.0)
        c = 2 * torch.asin(torch.sqrt(a))
        
        distance = self.earth_radius * c
        
        return distance.mean()

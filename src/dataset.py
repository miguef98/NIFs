import numpy as np
import torch
from torch.utils.data import IterableDataset
from src.obj import load
from scipy.spatial import KDTree

def sampleTrainingData(
        surface_pc: np.array,
        surface_normals: np.array,
        samplesOnSurface: int,
        samplesOffSurface: int,
        domainBounds: tuple = ([-1, -1, -1], [1, 1, 1]),
):
    samples = np.random.randint(0, surface_pc.shape[0], samplesOnSurface)
    surfacePoints = surface_pc[ samples, : ]
    surfaceNormals = surface_normals[ samples, :]

    domainPoints = np.random.uniform(
        domainBounds[0], domainBounds[1],
        (samplesOffSurface, 3)
    )

    fullSamples = torch.row_stack((
        torch.from_numpy( surfacePoints ),
        torch.from_numpy( domainPoints )
    ))
    fullNormals = torch.row_stack((
        torch.from_numpy( surfaceNormals ),
        torch.from_numpy( np.zeros_like( domainPoints ) )
    ))
    fullSDFs = torch.row_stack((
        torch.zeros( (samplesOnSurface, 1) ),
        torch.ones( (samplesOffSurface, 1) )
    ))

    return fullSamples.float().unsqueeze(0), fullNormals.float().unsqueeze(0), fullSDFs.float().unsqueeze(0)

class PointCloud(IterableDataset):
    def __init__(self, pointCloudPath: str,
                 batchSize: int,
                 samplingPercentiles: list,
                 batchesPerEpoch : int ):
        super().__init__()

        print(f"Loading mesh \"{pointCloudPath}\".")

        self.points, self.normals = load(pointCloudPath + '_pc.obj')
        
        self.batchSize = batchSize
        self.samplesOnSurface = int(self.batchSize * samplingPercentiles[0])
        self.samplesFarSurface = int(self.batchSize * samplingPercentiles[1])
        
        print(f"Fetching {self.samplesOnSurface} on-surface points per iteration.")
        print(f"Fetching {self.samplesFarSurface} far from surface points per iteration.")

        self.batchesPerEpoch = batchesPerEpoch
        
    def __iter__(self):
        for _ in range(self.batchesPerEpoch):
            yield sampleTrainingData(
                surface_pc=self.points,
                surface_normals=self.normals,
                samplesOnSurface=self.samplesOnSurface,
                samplesOffSurface=self.samplesFarSurface
            )

def shortestDistance( P, X ):
    sqnormP = torch.sum( P * P, dim=1)
    sqnormX = torch.sum( X * X, dim=1)

    shDistances, _ = torch.min( sqnormX.repeat( P.shape[0], 1 ) - 2 * ( P @ X.T ), dim=1 )

    return torch.sqrt( shDistances + sqnormP )

def sampleTrainingData2D(
        surface_pc: np.array,
        surface_normals: np.array,
        medial_axis: np.array,
        samplesOnSurface: int,
        samplesOffSurface: int,
        device:torch.device,
        domainBounds: tuple = ([-1, -1], [1, 1]),
):
    samples = np.random.randint(0, surface_pc.shape[0], samplesOnSurface)
    surfacePoints = surface_pc[ samples, : ]
    surfaceNormals = surface_normals[ samples, :]

    domainPoints = torch.from_numpy( np.random.uniform(
        domainBounds[0], domainBounds[1],
        (samplesOffSurface, 2)
    )).to(device)
    
    medialAxisPoints = torch.from_numpy( medial_axis ).to(device)

    domainPointsMADistance = shortestDistance( domainPoints, medialAxisPoints )
    eikonalEquationMask = ( (domainPointsMADistance > 0.001) * 2) - 1


    fullSamples = torch.row_stack((
        torch.from_numpy( surfacePoints ).to(device),
        domainPoints,
        medialAxisPoints
    ))
    fullNormals = torch.row_stack((
        torch.from_numpy( surfaceNormals ).to(device),
        torch.zeros_like( domainPoints  ).to(device),
        torch.zeros_like( medialAxisPoints ).to(device)
    ))
    fullSDFs = torch.row_stack((
        torch.zeros( (samplesOnSurface, 1) ).to(device),
        eikonalEquationMask.reshape( (samplesOffSurface, 1) ),
        torch.ones( (len(medial_axis), 1) ).to(device) * -1 
    ))

    return fullSamples.float().unsqueeze(0), fullNormals.float().unsqueeze(0), fullSDFs.float().unsqueeze(0)

class PointCloud2D(IterableDataset):
    def __init__(self, pointCloudPath: str,
                 batchSize: int,
                 samplingPercentiles: list,
                 batchesPerEpoch : int,
                 device: torch.device ):
        super().__init__()

        print(f"Loading point cloud \"{pointCloudPath}\".")

        file = np.load( pointCloudPath + '_pc.npz' )
        self.points, self.normals, self.medial_axis = file['points'], file['normals'], file['medial_axis']
        
        self.batchSize = batchSize
        self.samplesOnSurface = int(self.batchSize * samplingPercentiles[0])
        self.samplesFarSurface = int(self.batchSize * samplingPercentiles[1])
        self.device = device
        print(f"Fetching {self.samplesOnSurface} on-surface points per iteration.")
        print(f"Fetching {self.samplesFarSurface} far from surface points per iteration.")

        self.batchesPerEpoch = batchesPerEpoch
        
    def __iter__(self):
        for _ in range(self.batchesPerEpoch):
            yield sampleTrainingData2D(
                surface_pc=self.points,
                surface_normals=self.normals,
                medial_axis=self.medial_axis,
                samplesOnSurface=self.samplesOnSurface,
                samplesOffSurface=self.samplesFarSurface,
                device=self.device
            )
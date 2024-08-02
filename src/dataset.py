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

def sampleTrainingDataHarmonic(
        surface_pc: np.array,
        surface_normals: np.array,
        samplesOnSurface: int,
        samplesDomain: int,
        samplesBoundary: int,
        queryTree: KDTree,
        domainBounds: tuple = ([-1, -1, -1], [1, 1, 1]),
):
    samples = np.random.randint(0, surface_pc.shape[0], samplesOnSurface)
    surfacePoints = surface_pc[ samples, : ]
    surfaceNormals = surface_normals[ samples, :]

    domainPoints = np.random.uniform(
        domainBounds[0], domainBounds[1],
        (samplesDomain, 3)
    )

    boundaryPoints = np.random.uniform(
        domainBounds[0], domainBounds[1],
        (samplesBoundary, 3)
    )

    sel = np.random.randint(0, 3, samplesBoundary)
    boundaryPoints[ np.arange(len(boundaryPoints)), sel] = np.random.choice( [-1,1], samplesBoundary)
    boundaryPointsDistance, _ = queryTree.query( boundaryPoints, k=1 )

    fullSamples = torch.row_stack((
        torch.from_numpy( surfacePoints ),
        torch.from_numpy( domainPoints ),
        torch.from_numpy( boundaryPoints )
    ))
    fullNormals = torch.row_stack((
        torch.from_numpy( surfaceNormals ),
        torch.from_numpy( np.zeros_like( domainPoints ) ),
        torch.from_numpy( np.zeros_like( boundaryPoints ) )
    ))
    fullSDFs = torch.row_stack((
        torch.zeros( (samplesOnSurface, 1) ),
        -1 * torch.ones( (samplesDomain, 1) ),
        torch.from_numpy( boundaryPointsDistance.reshape( (samplesBoundary,1) ) )
    ))

    return fullSamples.float().unsqueeze(0), fullNormals.float().unsqueeze(0), fullSDFs.float().unsqueeze(0)


class PointCloudHarmonic(IterableDataset):
    def __init__(self, pointCloudPath: str,
                 batchSize: int,
                 samplingPercentiles: list,
                 batchesPerEpoch : int ):
        super().__init__()

        print(f"Loading mesh \"{pointCloudPath}\".")

        self.points, self.normals = load(pointCloudPath + '_pc.obj')
        self.queryTree = KDTree( self.points )

        self.batchSize = batchSize
        self.samplesOnSurface = int(self.batchSize * samplingPercentiles[0])
        self.samplesFarSurface = int(self.batchSize * samplingPercentiles[1])
        self.samplesBoundary = int(self.batchSize * samplingPercentiles[2])

        print(f"Fetching {self.samplesOnSurface} on-surface points per iteration.")
        print(f"Fetching {self.samplesFarSurface} far from surface points per iteration.")

        self.batchesPerEpoch = batchesPerEpoch
        
    def __iter__(self):
        for _ in range(self.batchesPerEpoch):
            yield sampleTrainingDataHarmonic(
                surface_pc=self.points,
                surface_normals=self.normals,
                samplesOnSurface=self.samplesOnSurface,
                samplesDomain=self.samplesFarSurface,
                samplesBoundary=self.samplesBoundary,
                queryTree=self.queryTree
            )
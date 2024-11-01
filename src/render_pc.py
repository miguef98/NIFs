import torch
import numpy as np
from src.model import SIREN
from src.evaluate import evaluate
from src.util import normalize
import warnings
import tqdm
from src.inverses import inverse

class Sampler:
    def __init__(self, n_in_features=3, hidden_layers=[256,256,256,256], w0=30, ww=None, checkpoint = None, device =0):
        self.decoder = SIREN(
            n_in_features= n_in_features,
            n_out_features=1,
            hidden_layer_config=hidden_layers,
            w0=w0,
            ww=ww
        )
        self.features = n_in_features
        self.device = torch.device(device)
        self.decoder.to( self.device )
        self.decoder.eval()

        self.decoder.load_state_dict( torch.load(checkpoint, map_location=self.device))

    def generate_point_cloud(self, gt_mode, alpha, num_steps = 5, num_points = 20000, surf_thresh = 0.01, max_iter=1000 ):

        for param in self.decoder.parameters():
            param.requires_grad = False

        surface_points = np.zeros((0, 3))
        normals = np.zeros((0,3))
        for iterations in tqdm.tqdm(range(max_iter)):

            if len(surface_points) != 0:
                samples = surface_points[ np.random.uniform(0, len(surface_points), num_points // 2).astype(np.uint32) ] + np.random.normal(0, 0.1, (num_points // 2, 3))
                samples = np.concatenate( [samples, np.random.uniform(-1, 1, (num_points // 2, 3) )] )
            else:
                samples = np.random.uniform(-1, 1, (num_points, 3) )

            gradients = np.zeros( (num_points, 3 ) )
            udfs = None
            for step in range(num_steps):
                if step == num_steps - 1:
                    hessians = np.zeros( (num_points, 3, 3))
                    udfs = evaluate( self.decoder, samples, gradients=gradients, hessians=hessians, device=self.device )
                else:
                    udfs = evaluate( self.decoder, samples, gradients=gradients, device=self.device )

                udfs = evaluate( self.decoder, samples, gradients=gradients, device=self.device )
                steps = inverse(gt_mode, udfs, alpha, min_step=0)

                samples -= steps * normalize(gradients)
        
            mask_points_on_domain = np.prod( np.logical_and( samples >= -1, samples <= 1 ), axis=1 ).astype(np.bool8)
            mask_points_on_surf = (  steps.flatten() < surf_thresh ) * mask_points_on_domain

            if np.sum(mask_points_on_surf) > 0:
                samples_near_surf = samples[ mask_points_on_surf ]
                surface_points = np.vstack((surface_points, samples_near_surf))

                if gt_mode == 'siren':
                    normals = np.vstack( ( normals, normalize(gradients)[mask_points_on_surf]) )
                else:
                    normals = np.vstack( ( normals, [ np.linalg.eigh(hessian)[1][:,2] for hessian in hessians[mask_points_on_surf] ]) )
            
            if len(surface_points) >= num_points:
                break

        if len(surface_points) < num_points:
            warnings.warn( '\033[93m' + f'Max iterations reached. Only sampled {len(surface_points)} surface points.' + '\033[0m', RuntimeWarning )

        return surface_points, normals

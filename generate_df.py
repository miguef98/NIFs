import numpy as np
import torch
import argparse
from src.model import SIREN
from src.evaluate import evaluate
from PIL import Image
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from src.util import normalize

def imagen_dist( axis, distancias, niveles, eps=0.0005, color_map='br', min_val=-1, max_val=1, contour=False):
    masked_distancias = distancias
    for v in niveles:
        masked_distancias = np.ma.masked_inside( masked_distancias, v / max_val - eps, v / max_val + eps )
    
    pos = axis.imshow( 
        masked_distancias.reshape(np.sqrt(len(distancias)).astype(np.uint32), np.sqrt(len(distancias)).astype(np.uint32)), 
        cmap=color_map, 
        interpolation='none', 
        vmin=min_val, 
        vmax=max_val
    )

    if contour:
        axis.contour(
            masked_distancias.reshape(np.sqrt(len(distancias)).astype(np.uint32), np.sqrt(len(distancias)).astype(np.uint32)),
            levels= np.linspace(min_val,max_val,23), colors='black', linewidths=0.5)
        pos = axis.contourf(
            masked_distancias.reshape(np.sqrt(len(distancias)).astype(np.uint32), np.sqrt(len(distancias)).astype(np.uint32)),
            levels= np.linspace(min_val,max_val,23), cmap=color_map)
        
    axis.set_xticks([])
    axis.set_yticks([])

    return pos
    
def generate_df( model_path, output_path, options ):

    model = SIREN(
            n_in_features= 3,
            n_out_features=1,
            hidden_layer_config=options['hidden_layer_nodes'],
            w0=options['weight0'],
            ww=None,
            activation=options.get('activation', 'sine')
    )
    model.load_state_dict( torch.load(model_path, weights_only=True))

    SAMPLES = options['width'] ** 2
    BORDES = [1, -1]
    EJEPLANO = [0,1,2]
    OFFSETPLANO = 0.0

    device_torch = torch.device(options['device'])
    model.to(device_torch)

    ranges = np.linspace(BORDES[0], BORDES[1], options['width'])
    i_1, i_2 = np.meshgrid( ranges, ranges )
    samples = np.concatenate(
            np.concatenate( np.array([np.expand_dims(i_1, 2), 
                                np.expand_dims(i_2, 2), 
                                np.expand_dims(np.ones_like(i_1) * OFFSETPLANO, 2)])[EJEPLANO]
                        , axis=2 ),
            axis=0)

    gradients = np.zeros((SAMPLES, 3))
    pred_distances = evaluate( model, samples, device=device_torch, gradients=gradients )
    pred_grad_norm = np.linalg.norm( gradients , axis=1 ).reshape((SAMPLES, 1))

    gradients = normalize(gradients)

    normals = gradients
    grad_map = ( normals + np.ones_like(normals) ) / 2

    plt.rcParams.update({'font.size': 16})
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10,4), dpi=500)

    pos1 = imagen_dist( axes[0] , pred_distances, [0], color_map='bwr_r', eps=options['surf_thresh'], contour=True)
    pos2 = axes[1].imshow( pred_grad_norm.reshape((options['width'], options['width'])), cmap='plasma', vmin=0, vmax=np.max(pred_grad_norm) )

    axes[0].set_title(r'$f$')
    plt.colorbar(pos1,ax=axes[0])
    axes[1].set_title(r'$\left \| \nabla f \right \|$')
    plt.colorbar(pos2,ax=axes[1])

    fig.tight_layout()
    fig.savefig(output_path + 'distance_fields.png')

    im = Image.fromarray((grad_map.reshape(np.sqrt(SAMPLES).astype(np.uint32), np.sqrt(SAMPLES).astype(np.uint32), 3) * 255).astype(np.uint8))
    im.save( output_path +'pred_grad.png', 'PNG')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate dense point cloud from trained model')
    parser.add_argument('mesh_path', metavar='path/to/mesh.ply', type=str,
                        help='path to input preprocessed mesh')
    parser.add_argument('model_path', metavar='path/to/pth', type=str,
                        help='path to input model')
    parser.add_argument('output_path', metavar='path/to/output/', type=str,
                        help='path to output folder')
    parser.add_argument('-d', '--device', type=int, default=0, help='torch device')
    parser.add_argument('-w0', '--weight0', type=float, default=30, help='w0 parameter of SIREN')
    parser.add_argument('-w', '--width', type=int, default=512, help='width of generated image')
    parser.add_argument('-t', '--surf_thresh', type=float, default=1e-3, help='on surface threshold')

    args = parser.parse_args()
    d = vars(args)
    d['hidden_layer_nodes'] = [256,256,256,256,256,256,256,256]
    d['activation']='relu'
    generate_df(args.model_path, args.mesh_path, args.output_path,d )


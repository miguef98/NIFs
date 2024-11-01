# coding: utf-8
import torch
import torch.nn.functional as F
import src.diff_operators as dif

def dirichlet_boundary(gt_sdf, pred_sdf):
    return torch.where(
        gt_sdf == 0,
        (pred_sdf) ** 2,
        torch.zeros_like(pred_sdf)
    )

def neumann_boundary(gt_sdf, gt_vectors, pred_vectors):
    return torch.where(
        gt_sdf == 0,
        torch.sum( (pred_vectors - gt_vectors) ** 2, dim=-1 )[...,None],
        #1 - F.cosine_similarity(pred_vectors / pred_vectors.norm(dim=-1)[...,None], gt_vectors.squeeze(0), dim=-1)[..., None],
        torch.zeros_like(gt_sdf)
    )

def eikonal_equation(gradient):
    return (gradient.norm(dim=-1) - 1.) ** 2
    
def weighted_eikonal_equation( gradient, gt_sdf ):
    return torch.where(
        gt_sdf == 0,
        ((gradient.norm(dim=-1) - 1.) ** 2 ) * 100,
        (gradient.norm(dim=-1) - 1.) ** 2
    )

def regularization_siren(gt_sdf, pred_sdf, radius=1e3):
    """
    This function penalizes the pred_sdf of points in gt_sdf!=0
    Used in SIREN's papers
    """
    return torch.where(
           gt_sdf != 0,
           torch.exp(- radius * torch.abs(pred_sdf)),
           torch.zeros_like(pred_sdf)
        )

def regularization_total_variation( gradient, coords ):
    return dif.gradient( gradient.norm( dim=-1 ), coords ).norm( dim=-1 )

def loss_siren( model, model_input, gt, loss_weights ):
    model_output = model(model_input)

    gt_sdf = gt['sdfs']
    gt_normals = gt['normals']

    coords = model_output['model_in']
    pred_sdf = model_output['model_out']

    gradient = dif.gradient(pred_sdf, coords).squeeze(0)

    return {
        'eikonal_equation': eikonal_equation(gradient).mean() * loss_weights[0],
        'dirichlet_boundary': dirichlet_boundary(gt_sdf, pred_sdf).mean() * loss_weights[1],
        'neumann_boundary': neumann_boundary(gt_sdf, gt_normals, gradient).mean() * loss_weights[2] ,
        #'regularization_siren': regularization_siren( gt_sdf, pred_sdf).mean() * loss_weights[3],
        'regularization_total_variation': regularization_total_variation( gradient, coords ).mean() * loss_weights[4]
    }

def eikonal_medial_axis( gt_sdf, gradient ):
    return torch.where(
        gt_sdf.flatten() == -1,
        gradient.norm(dim=-1) ** 2,
        (gradient.norm(dim=-1) - 1.) ** 2,
    )

#def regularization_total_variation( gradient, coords ):
#    return dif.gradient( gradient.norm( dim=-1 ), coords ).norm( dim=-1 )

def loss_medial_axis( model, model_input, gt, loss_weights ):
    model_output = model(model_input)

    gt_sdf = gt['sdfs']
    gt_normals = gt['normals']

    coords = model_output['model_in']
    pred_sdf = model_output['model_out']

    gradient = dif.gradient(pred_sdf, coords).squeeze(0)

    return {
        'eikonal_equation': eikonal_medial_axis(gt_sdf, gradient).mean() * loss_weights[0],
        'dirichlet_boundary': dirichlet_boundary(gt_sdf, pred_sdf).mean() * loss_weights[1],
        'neumann_boundary': neumann_boundary(gt_sdf, gt_normals, gradient).mean() * loss_weights[2] ,
        #'regularization_siren': regularization_siren( gt_sdf, pred_sdf).mean() * loss_weights[3],
        'regularization_total_variation': regularization_total_variation( gradient, coords ).mean() * loss_weights[4]
    }
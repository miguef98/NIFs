from src.render_mc import get_mesh_sdf
from src.model import SIREN
import torch
import argparse
import json

def generate_mc(model, device, N, output_path, from_file=None):

	if from_file is not None:
		model = SIREN(
			n_in_features= 3,
			n_out_features=1,
			hidden_layer_config=from_file["hidden_layer_nodes"],
			w0=from_file["w0"],
			ww=None,
			activation=from_file.get('activation','sine')
		)

		model.load_state_dict( torch.load(from_file["model_path"], weights_only=True))
		model.to(device)

	vertices, faces, meshSIREN = get_mesh_sdf(
		model,
		N=N,
		device=device
	)

	meshSIREN.export(output_path)
	print(f'Saved to {output_path}')

	return meshSIREN	

if __name__=='__main__':
	parser = argparse.ArgumentParser(description='Generate mesh through marching cubes from trained model')
	parser.add_argument('config_path', metavar='path/to/json', type=str,
					help='path to render config')

	args = parser.parse_args()

	with open(args.config_path) as config_file:
		config_dict = json.load(config_file)	

	device_torch = torch.device(config_dict["device"])

	model = SIREN(
		n_in_features= 3,
		n_out_features=1,
		hidden_layer_config=config_dict["hidden_layer_nodes"],
		w0=config_dict["w0"],
		ww=None
	)

	model.load_state_dict( torch.load(config_dict["model_path"], map_location=device_torch))
	model.to(device_torch)

	print('Generating mesh...')

	generate_mc(model, device_torch, config_dict['nsamples'], config_dict['output_path'])


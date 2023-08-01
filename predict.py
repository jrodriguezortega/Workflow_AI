import os
import argparse

import pytorch_lightning as pl

from project.datamodules import LULCDataModuleInference
from project.pl_modules import LitBRITS
from project.utils import save_predictions

# ------------
# Set up default hyperparameters
# ------------
default_config = {
    'output_dir': './outputs',
    'model': {
        'rnn_hid_size': 100,
        'impute_weight': 0.25,
        'label_weight': 0.75,
        'data_dim': 7,
        'seq_len': 12,
        'lr': 3e-3,
        'wd': 0,
        'optimizer': 'adam',
        'scheduler': 'cosine',
        'soft_labels': True,
    },
    'data': {
        'json_dir': './data/',
        'level': 'N1',
        'batch_size': 1,
        'data_dim': 7,
    },
}


def predict(config: dict, checkpoint_path: str):
    # ------------
    # Create Data Module
    # ------------
    data_module = LULCDataModuleInference(**config['data'])
    classes, class_to_idx, idx_to_class = data_module.get_class_vars()

    # ------------
    # Create Model
    # ------------
    model = LitBRITS.load_from_checkpoint(checkpoint_path, **config['model'], idx_to_class=idx_to_class)

    # ------------
    # Create trainer
    # ------------
    trainer = pl.Trainer(accelerator='cpu')

    print('Computing predictions...')
    predictions = trainer.predict(model, datamodule=data_module)

    csv_path = os.path.join(config['output_dir'], f'predictions_{config["data"]["level"]}.csv')
    save_predictions(
        predictions=predictions,
        csv_path=csv_path,
        idx_to_class=model.hparams.idx_to_class
    )
    print('--------------------------------------')
    print(f'Predictions saved in: {csv_path}')


def get_args():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--level', default='N1', type=str,
                        help='Classification level to use', choices=['N1', 'N2'])
    parser.add_argument('--output_dir', default='./outputs', type=str,
                        help='Output directory to save the results')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    level = args.level

    config = default_config

    config['output_dir'] = args.output_dir

    if not os.path.exists(config['output_dir']):
        os.makedirs(config['output_dir'])

    config['data']['json_dir'] = './data/json_dir'
    config['data']['level'] = level
    config['data']['batch_size'] = 1

    ckpt_path = f'./models/{level}/model.ckpt'

    predict(config=config, checkpoint_path=ckpt_path)

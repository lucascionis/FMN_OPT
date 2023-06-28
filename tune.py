import os
import math
import argparse
import pickle
import numpy as np

import torch
from ray import air
from ray.air import session
from ray import tune
from ray.tune.search.flaml import CFO
from ray.tune.schedulers import ASHAScheduler

from src.fmn_opt import FMNOpt
from utils.load_data import load_data
from configs.tuning_resources import TUNING_RES
from configs.search_spaces import OPTIMIZERS_SEARCH_TUNE, SCHEDULERS_SEARCH_TUNE
# global device definition
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["TUNE_DISABLE_AUTO_CALLBACK_LOGGERS"] = "1"
os.environ["TUNE_MAX_PENDING_TRIALS_PG"] = "1"

parser = argparse.ArgumentParser(description='Retrieve tuning params')
parser.add_argument('-opt', '--optimizer',
                    type=str,
                    default='SGD',
                    help='Provide the optimizer name (e.g. SGD, Adam)')
parser.add_argument('-sch', '--scheduler',
                    type=str,
                    default='CosineAnnealingLR',
                    help='Provide the scheduler name (e.g. CosineAnnealingLR)')
parser.add_argument('-m_id', '--model_id',
                    type=int,
                    default=0,
                    help='Provide the model ID')
parser.add_argument('-d_id', '--dataset_id',
                    type=int,
                    default=0,
                    help='Provide the dataset ID (e.g. CIFAR10: 0)')
parser.add_argument('-bs', '--batch_size',
                    type=int,
                    default=50,
                    help='Provide the batch size (e.g. 50, 100, 10000)')
parser.add_argument('-bn', '--batch_number',
                    type=int,
                    default=10,
                    help='Provide the batch number (e.g. 10, 50, 100)')
parser.add_argument('-s', '--steps',
                    type=int,
                    default=10,
                    help='Provide the steps number (e.g. 50, 100, 10000)')
parser.add_argument('-n', '--norm',
                    type=str,
                    default='inf',
                    help='Provide the norm (0, 1, 2, inf)')
parser.add_argument('-n_sample', '--num_samples',
                    type=int,
                    default=5,
                    help='Provide the number of samples for the tuning')
parser.add_argument('-ep', '--epochs',
                    type=int,
                    default=5,
                    help='Provide the number of epochs for the attack tuning \
                    (how many times the attack is run by the tuner)')
parser.add_argument('-l', '--loss',
                    type=int,
                    default=0,
                    help='Provide the selection of the loss computation 0 for logit, 1 for cross entropy\
                         (default: 0)')
parser.add_argument('-wp', '--working_path',
                    default='TuningExp')

args = parser.parse_args()


def tune_attack(config, model, dataset, attack_params, epochs=5):
    for epoch in range(epochs):
        attack = FMNOpt(
            model=model,
            dataset=dataset,
            norm=attack_params['norm'],
            steps=attack_params['steps'],
            batch_size=attack_params['batch_size'],
            batch_number=attack_params['batch_number'],
            optimizer=attack_params['optimizer'],
            scheduler=attack_params['scheduler'],
            optimizer_config=config['opt_s'],
            scheduler_config=config['sch_s'],
            device=device,
            logit_loss=True if attack_params['loss_selection'] == 0 else False
        )

        # distance, _ = attack.run()
        attack.run()
        median_distance = np.median([a['best_distance'] for a in attack.attack_data])
        session.report({"distance": median_distance})


def run_tuning(optimizer = 'SGD',
               scheduler = 'CosineAnnealingLR',
               model_id = 8,
               dataset_id = 0,
               batch_size=5,
               batch_number=10,
               steps=10,
               norm='inf',
               num_samples=5,
               epochs=2,
               loss = 0,
               working_path = 'tuning_exp'):
    # load search spaces
    optimizer_search = OPTIMIZERS_SEARCH_TUNE[optimizer]
    scheduler_search = SCHEDULERS_SEARCH_TUNE[scheduler]

    attack_params = {
        'batch_size': batch_size,
        'batch_number': batch_number,
        'steps': steps,
        'norm': norm,
        'optimizer': optimizer,
        'scheduler': scheduler,
        'loss_selection': loss
    }

    tune_config = {
        'num_samples': num_samples,
        'epochs': epochs
    }

    # load model and dataset
    model, dataset = load_data(model_id, dataset_id, attack_params['norm'])

    steps_keys = ['T_max', 'T_0', 'milestones']
    for key in steps_keys:
        if key in scheduler_search:
            scheduler_search[key] = scheduler_search[key](attack_params['steps'])
    search_space = {
        'opt_s': optimizer_search,
        'sch_s': scheduler_search
    }

    trainable_with_resources = tune.with_resources(
        tune.with_parameters(
            tune_attack,
            model=torch.nn.DataParallel(model).to(device),
            dataset=dataset,
            attack_params=attack_params,
            epochs=tune_config['epochs']
        ),
        resources=TUNING_RES
    )

    tune_scheduler = ASHAScheduler(mode='min', metric='distance', grace_period=2)
    algo = CFO(metric='distance', mode='min')

    # ./TuningExp/Modelname_dataset/...
    str_loss = ['LL', 'CE']
    # Defining experiment name
    tuning_exp_name = f"{optimizer}_{scheduler}_{str_loss[loss]}"
    tuning_exp_path = os.path.join(working_path, tuning_exp_name)
    tuner = tune.Tuner(
        trainable_with_resources,
        param_space=search_space,
        tune_config=tune.TuneConfig(
            num_samples=tune_config['num_samples'],
            search_alg=algo,
            scheduler=tune_scheduler
        ),
        run_config=air.RunConfig(tuning_exp_name,
                                 local_dir=working_path,
                                 checkpoint_config=air.CheckpointConfig(
                                     checkpoint_at_end=False,
                                     checkpoint_frequency=0,
                                     num_to_keep=None),
                                 log_to_file=False,
                                 )
    )

    results = tuner.fit()

    # Checking best result and best config
    best_result = results.get_best_result(metric='distance', mode='min')
    best_config = best_result.config
    print(f"best_distance : {best_result.metrics['distance']}\n, best config : {best_config}\n")

    best_result_packed = {
        'distance': best_result.metrics['distance'],
        'best_config': best_result.config
    }
    # highlighting end of run
    print("\n+++++COMPLETE LOG AT: {0}/{1}+++++\n".format(working_path, tuning_exp_name))

    filename = os.path.join(tuning_exp_path, "best_result.pkl")
    with open(filename, "wb") as file:
        pickle.dump(best_result_packed, file)


if __name__ == '__main__':
    optimizer = args.optimizer
    scheduler = args.scheduler
    model_id = args.model_id
    dataset_id = args.dataset_id
    loss = args.loss
    working_path = args.working_path

    batch_size = int(args.batch_size)
    batch_number = int(args.batch_number)
    steps = int(args.steps)
    norm = args.norm
    num_samples = int(args.num_samples)
    epochs = int(args.epochs)

    run_tuning(
        optimizer=optimizer,
        scheduler=scheduler,
        model_id=model_id,
        dataset_id=dataset_id,
        batch_size=batch_size,
        batch_number=batch_number,
        steps=steps,
        norm=norm,
        num_samples=num_samples,
        epochs=epochs,
        loss=loss,
        working_path=working_path)



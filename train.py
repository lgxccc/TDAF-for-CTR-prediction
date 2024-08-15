
import argparse
import collections
import json
import os
import random
import time
import pandas as pd
import numpy as np
import torch
import torch.utils.data
import sys
from deepctr_torch.inputs import SparseFeat, VarLenSparseFeat, build_input_features
sys.path.append('..')
sys.path.append('../..')
import hparams_registry
import algorithms
import misc
from fast_data_loader import InfiniteDataLoader, FastDataLoader
from torch.utils.data import TensorDataset


def get_model_input(data, sparse_features, dataset):
    if dataset=="ml-10m":
        model_input = {name: data[name] for name in sparse_features}
        model_input["movie_title"] = data.iloc[:, 5:32]
        model_input["genre"] = data.iloc[:, 32:40]
        model_input['env'] = data['env']
    else:
        model_input = {name: data[name] for name in sparse_features}
        model_input["actors"] = data.iloc[:, 31:61]
        model_input["directors"] = data.iloc[:, 61:91]
        model_input["genres"] = data.iloc[:, 91:100]
        model_input["languages"] = data.iloc[:, 116:135]
        model_input["regions"] = data.iloc[:, 135:160]
        model_input["tags"] = data.iloc[:, 100:116]
        model_input['env'] = data['env']
    return model_input

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    train_start = time.time()
    parser = argparse.ArgumentParser(description='Domain generalization')
    parser.add_argument('--algorithm', type=str, default="DeepFM")
    parser.add_argument('--task', type=str, default="domain_generalization")
    parser.add_argument('--hparams', type=str,
                        help='JSON-serialized hparams dict')
    parser.add_argument('--hparams_seed', type=int, default=0)
    parser.add_argument('--trial_seed', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0,
                        help='Seed for everything else')
    parser.add_argument('--IL', type=str, default=None,help='Seed for everything else')
    parser.add_argument('--gpu', type=int, default=0,
                        help='gpu index')
    parser.add_argument('--checkpoint_freq', type=int, default=100)
    parser.add_argument('--output_dir', type=str, default="train_output")
    parser.add_argument('--holdout_fraction', type=float, default=0.2)
    parser.add_argument('--uda_holdout_fraction', type=float, default=0)
    parser.add_argument('--skip_model_save', action='store_true')
    parser.add_argument('--save_model_every_checkpoint', action='store_true')
    parser.add_argument('--ctr_dataset', type=str, default="ml-10m",help='dataset for ctr')
    parser.add_argument('--model', type=str, default="deepfm", help='model for ctr')
    args = parser.parse_args()
    print("args:", args)

    # random.seed(args.seed)
    # np.random.seed(args.seed)
    # torch.manual_seed(args.seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # device = "cuda"
    # torch.cuda.set_device(args.gpu)

    start_step = 0
    algorithm_dict = None

    os.makedirs(args.output_dir, exist_ok=True)
    sys.stdout = misc.Tee(
        os.path.join(args.output_dir, args.dataset + '_' + args.algorithm + '_' + str(args.test_envs) + '_out.txt'))
    sys.stderr = misc.Tee(
        os.path.join(args.output_dir, args.dataset + '_' + args.algorithm + '_' + str(args.test_envs) + '_err.txt'))

    for k, v in sorted(vars(args).items()):
        print('\t{}: {}'.format(k, v))

    hparams = hparams_registry.default_hparams(args.algorithm, args.dataset)
    hparams.update(json.loads(args.hparams))
    for k, v in sorted(hparams.items()):
        print('\t{}: {}'.format(k, v))

    if not os.path.exists(args.output_dir + "/results"):
        os.makedirs(args.output_dir + "/results")
    random_string = str(random.randint(10, 99))
    t = time.localtime()
    time_str = str(t.tm_year) + "0" + str(t.tm_mon) + str(t.tm_mday) + str(t.tm_hour)

    file_name = args.output_dir + "/results/" + args.algorithm + "_" + args.dataset + "_" + str(args.test_envs) + "_" + args.hparams + "_" + time_str + random_string + ".txt"


    file_name = file_name.replace("\"", "")
    with open(file_name, 'a') as f:
        f.write(str(args))
        f.write(str(hparams))
        f.close()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = "cuda"
    torch.cuda.set_device(args.gpu)

    k=8
    if args.ctr_dataset == 'douban':
        train_envs = [14, 15, 16, 17, 18]
        # val_envs = [19, 20, 21, 22, 23]
        val_envs = [19]
        # test_envs = [20,21,22,23]
        test_envs = [20]
    elif args.ctr_dataset == 'ml-10m':
        train_envs = [13, 14, 15, 16, 17]
        # val_envs = [18]
        val_envs = [18,19,20,21,22]
        # test_envs = [19,20,21,22]
        test_envs = [19]
    else:
        # train_envs = [21,22,23,24,25]
        # val_envs = [26]
        # test_envs = [27]
        train_envs = [2, 3, 4, 5,6,7]
        val_envs = [8]
        test_envs = [9]

    if args.ctr_dataset == "ml-10m":
        data = pd.read_hdf("D:\pyproject\DIL-main\data\ml-10m.h5")
        sparse_features = ['user_id:token', 'item_id:token', 'release_year:token', 'hour', 'wday']
        target = ['rating:float']
        fixlen_feature_columns = [SparseFeat(feat, np.max(data[feat]) + 1, embedding_dim=k)
                                       for feat in sparse_features]
        varlen_feature_columns = [VarLenSparseFeat(SparseFeat('genre', vocabulary_size=20
                                                                   , embedding_dim=k), maxlen=8, combiner='mean'),
                                       VarLenSparseFeat(SparseFeat('movie_title', vocabulary_size=10855
                                                                   , embedding_dim=k), maxlen=27, combiner='mean')]

        user_feature_colummn = [SparseFeat('user_id:token', np.max(data['user_id:token']) + 1, embedding_dim=k)]
        train_data = data[data['env'].isin(train_envs)]
        model_input = get_model_input(train_data,sparse_features,args.ctr_dataset)
        val_data = data[data['env'].isin(val_envs)]
        val_model_input = get_model_input(val_data,sparse_features,args.ctr_dataset)
        test_data = data[data['env'].isin(test_envs)]
        test_model_input = get_model_input(test_data, sparse_features, args.ctr_dataset)

        dnn_feature_columns = varlen_feature_columns + fixlen_feature_columns
        linear_feature_columns = varlen_feature_columns + fixlen_feature_columns
        feature_index = build_input_features(linear_feature_columns + dnn_feature_columns)
        if isinstance(model_input, dict):
            x_temp = [model_input[feature] for feature in feature_index]
            x_temp.append(train_data['env'])
            x_temp.append(train_data['rating:float'])
            x = x_temp
        for i in range(len(x)):
            if len(x[i].shape) == 1:
                x[i] = np.expand_dims(x[i], axis=1)
        x = np.concatenate(x, axis=-1)

        val_temp = [val_model_input[feature] for feature in feature_index]
        val_temp.append(val_data['env'])
        val_temp.append(val_data['rating:float'])
        val = val_temp

        test_temp = [test_model_input[feature] for feature in feature_index]
        test_temp.append(test_data['env'])
        test_temp.append(test_data['rating:float'])
        test = test_temp
        for i in range(len(val)):
            if len(val[i].shape) == 1:
                val[i] = np.expand_dims(val[i], axis=1)
        val = np.concatenate(val, axis=-1)

        for i in range(len(test)):
            if len(test[i].shape) == 1:
                test[i] = np.expand_dims(test[i], axis=1)
        test = np.concatenate(test, axis=-1)

    elif args.ctr_dataset == "douban":
        data = pd.read_hdf("D:\pyproject\DIL-main\data\douban.h5")
        data_subset = data.head(10)
        data_subset.to_excel('data_subset.xlsx', index=False, engine='openpyxl')

        sparse_features = ['USER_MD5', 'MOVIE_ID',
                                'DOUBAN_SCORE', 'DOUBAN_VOTES',
                                'IMDB_ID', 'MINS', 'OFFICIAL_SITE',
                                'YEAR', 'RATING_MONGTH', 'RATING_WEEKDAY', 'RATING_HOUR',
                                'RELEASE_YEAR', 'RELEASE_MONTH']
        target = ['RATING']
        user_feature_colummn = [SparseFeat('USER_MD5', np.max(data['USER_MD5']) + 1, embedding_dim=k)]
        fixlen_feature_columns = [SparseFeat(feat, np.max(data[feat]) + 1, embedding_dim=k)
                                       for feat in sparse_features]
        varlen_feature_columns = [VarLenSparseFeat(SparseFeat('actors', vocabulary_size=85425
                                                                   , embedding_dim=k), maxlen=30, combiner='mean'),
                                       VarLenSparseFeat(SparseFeat('directors', vocabulary_size=19498
                                                                   , embedding_dim=k), maxlen=30, combiner='mean'),
                                       VarLenSparseFeat(SparseFeat('genres', vocabulary_size=54
                                                                   , embedding_dim=k), maxlen=9, combiner='mean'),
                                       VarLenSparseFeat(SparseFeat('languages', vocabulary_size=670
                                                                   , embedding_dim=k), maxlen=19, combiner='mean'),
                                       VarLenSparseFeat(SparseFeat('regions', vocabulary_size=379
                                                                   , embedding_dim=k), maxlen=25, combiner='mean'),
                                       VarLenSparseFeat(SparseFeat('tags', vocabulary_size=52457
                                                                   , embedding_dim=k), maxlen=16, combiner='mean')]
        train_data = data[data['env'].isin(train_envs)]
        model_input = get_model_input(train_data,sparse_features,args.ctr_dataset)
        val_data = data[data['env'].isin(val_envs)]
        val_model_input = get_model_input(val_data,sparse_features,args.ctr_dataset)
        test_data = data[data['env'].isin(test_envs)]
        test_model_input = get_model_input(test_data, sparse_features, args.ctr_dataset)

        dnn_feature_columns = varlen_feature_columns + fixlen_feature_columns
        linear_feature_columns =  varlen_feature_columns + fixlen_feature_columns

        feature_index = build_input_features(linear_feature_columns + dnn_feature_columns)
        x_temp = [model_input[feature] for feature in feature_index]
        x_temp.append(train_data['env'])
        x_temp.append(train_data['RATING'])
        x = x_temp
        for i in range(len(x)):
            if len(x[i].shape) == 1:
                x[i] = np.expand_dims(x[i], axis=1)
        x = np.concatenate(x, axis=-1)

        val_temp = [val_model_input[feature] for feature in feature_index]
        val_temp.append(val_data['env'])
        val_temp.append(val_data['RATING'])
        val = val_temp

        test_temp = [test_model_input[feature] for feature in feature_index]
        test_temp.append(test_data['env'])
        test_temp.append(test_data['RATING'])
        test = test_temp
        for i in range(len(val)):
            if len(val[i].shape) == 1:
                val[i] = np.expand_dims(val[i], axis=1)
        val = np.concatenate(val, axis=-1)

        for i in range(len(test)):
            if len(test[i].shape) == 1:
                test[i] = np.expand_dims(test[i], axis=1)
        test = np.concatenate(test, axis=-1)

    else: #aliec
        data = pd.read_hdf("D:\pyproject\DIL-main\data/AliEC.h5")
        env_counts = data['user'].value_counts().sort_index()
        sparse_features = ['user', 'adgroup_id', 'pid', 'cate_id', 'campaign_id', 'customer', 'cms_segid', 'cms_group_id',
                           'final_gender_code', 'age_level', 'shopping_level', 'occupation']
        target = ['clk']
        fixlen_feature_columns = [SparseFeat(feat, np.max(data[feat]) + 1, embedding_dim=k)
                                       for feat in sparse_features]
        varlen_feature_columns = []
        train_data = data[data['env'].isin(train_envs)]
        model_input = get_model_input(train_data,sparse_features,args.ctr_dataset)
        val_data = data[data['env'].isin(val_envs)]
        val_model_input = get_model_input(val_data,sparse_features,args.ctr_dataset)
        test_data = data[data['env'].isin(test_envs)]
        test_model_input = get_model_input(test_data, sparse_features, args.ctr_dataset)

        dnn_feature_columns = varlen_feature_columns + fixlen_feature_columns
        linear_feature_columns =  varlen_feature_columns + fixlen_feature_columns

        feature_index = build_input_features(linear_feature_columns + dnn_feature_columns)
        x_temp = [model_input[feature] for feature in feature_index]
        x_temp.append(train_data['env'])
        x_temp.append(train_data['clk'])
        x = x_temp
        for i in range(len(x)):
            if len(x[i].shape) == 1:
                x[i] = np.expand_dims(x[i], axis=1)
        x = np.concatenate(x, axis=-1)

        val_temp = [val_model_input[feature] for feature in feature_index]
        val_temp.append(val_data['env'])
        val_temp.append(val_data['clk'])
        val = val_temp

        test_temp = [test_model_input[feature] for feature in feature_index]
        test_temp.append(test_data['env'])
        test_temp.append(test_data['clk'])
        test = test_temp
        for i in range(len(val)):
            if len(val[i].shape) == 1:
                val[i] = np.expand_dims(val[i], axis=1)
        val = np.concatenate(val, axis=-1)

        for i in range(len(test)):
            if len(test[i].shape) == 1:
                test[i] = np.expand_dims(test[i], axis=1)
        test = np.concatenate(test, axis=-1)

    train_data_list = [x[x[:, -2] == i] for i in train_envs]
    val_data_list = [val[val[:, -2] == i] for i in val_envs]
    test_data_list = [test[test[:, -2] == i] for i in test_envs]
    def dataframe_to_tensordataset(array, domain_index, eval="train"):
        labels = torch.tensor(array[:,-1], dtype=torch.float32)
        features = torch.tensor(array[:,:-1], dtype=torch.float32)
        if eval=="train":
            domains = torch.full((len(array),), domain_index, dtype=torch.long)
        elif eval=="val":
            domains = torch.full((len(array),), len(train_envs) + domain_index, dtype=torch.long)
        elif eval=="test":
            domains = torch.full((len(array),), len(train_envs) + len(val_envs) + domain_index, dtype=torch.long)
        return TensorDataset(features, labels, domains)

    tensor_datasets = [dataframe_to_tensordataset(env, i) for i, env in enumerate(train_data_list)]
    train_loaders = [
        InfiniteDataLoader(
            dataset=env,
            weights=None,
            batch_size=hparams['batch_size'],
            domain=i,
            num_workers=0
        )
        for i, env in enumerate(tensor_datasets)
        if i not in args.test_envs
    ]

    val_tensor_datasets = [dataframe_to_tensordataset(env, i, "val") for i, env in enumerate(val_data_list)]
    eval_loaders = [FastDataLoader(
        dataset=env,
        batch_size=hparams['batch_size'],
        num_workers=0)
        for env in val_tensor_datasets]

    test_tensor_datasets = [dataframe_to_tensordataset(env, i, "test") for i, env in enumerate(test_data_list)]
    test_loaders = [FastDataLoader(
        dataset=env,
        batch_size=hparams['batch_size'],
        num_workers=0)
        for env in test_tensor_datasets]

    train_minibatches_iterator = zip(*train_loaders)

    if args.ctr_dataset == "ml-10m":
        eval_loader_names = [f'val:env{18}']
        # test_loader_names = [f'test:env{19}',f'test:env{20}',f'test:env{21}',f'test:env{22}']
        test_loader_names = [f'test:env{19}']
    elif args.ctr_dataset == 'douban':    # douban
        # eval_loader_names = [f'val:env{19}',f'val:env{20}',f'val:env{21}',f'val:env{22}',f'val:env{23}',f'val:env{24}',]
        eval_loader_names = [f'val:env{19}']
        test_loader_names = [f'test:env{20}']
        # test_loader_names = [f'test:env{20}', f'test:env{21}', f'test:env{22}', f'test:env{23}']
    else:
        eval_loader_names = [f'val:env{8}']
        test_loader_names = [f'test:env{9}']

    algorithm_class = algorithms.get_algorithm_class(args.algorithm)
    print("algorithm_class:", algorithm_class)
    input_shape = len(tensor_datasets[0][0][0])
    algorithm = algorithm_class(input_shape, 2, feature_index, len(train_envs), hparams, dnn_feature_columns,linear_feature_columns,args.model)
    # algorithm_dda = algorithm_class(input_shape, 2, feature_index, len(train_envs), hparams, dnn_feature_columns,
    #                             linear_feature_columns, args.model)

    if algorithm_dict is not None:
        algorithm.load_state_dict(algorithm_dict)

    algorithm.to(device)

    train_minibatches_iterator = zip(*train_loaders)

    checkpoint_vals = collections.defaultdict(lambda: [])

    steps_per_epoch = min([len(env) / hparams['batch_size'] for env in train_data_list])


    n_steps = args.steps
    print("total steps:", n_steps, "args.steps:", args.steps)
    checkpoint_freq = args.checkpoint_freq or data.CHECKPOINT_FREQ

    def save_checkpoint(filename):
        if args.skip_model_save:
            return
        save_dict = {
            "args": vars(args),
            "train_num_domains": train_envs,
            "val_num_domains": val_envs,
            "model_hparams": hparams,
            "model_dict": algorithm.cpu().state_dict()
        }
        torch.save(save_dict, os.path.join(args.output_dir, filename))

    last_results_keys = None
    best_val_auc = 0
    stop_patience = 0
    best_test_auc = 0
    best_test_loss = 0
    for step in range(start_step, n_steps):
        step_start_time = time.time()
        if hparams['dm_idx']:
            minibatches_device = [(x.to(device), y.to(device), d.to(device))
                                  for x, y, d in next(train_minibatches_iterator)]
        else:
            minibatches_device = [(x.to(device), y.to(device))
                                  for x, y in next(train_minibatches_iterator)]
        uda_device = None
        
        step_vals = algorithm.update(minibatches_device, dnn_feature_columns, feature_index, IL=args.IL)
        checkpoint_vals['step_time'].append(time.time() - step_start_time)

        if (step % 5 == 0) or (step == n_steps - 1):
            results = {
                'step': step,
                'epoch': step / steps_per_epoch,
            }

            for key, val in checkpoint_vals.items():
                results[key] = np.mean(val)

            val_auc_list = []
            evals = zip(eval_loader_names, eval_loaders)
            for name, loader in evals:
                val_auc, val_logloss = misc.accuracy(algorithm, loader,None, device, args.algorithm)
                results[name + 'auc'] = val_auc
                results[name + 'loss'] = val_logloss
                results[test_loader_names[0] + 'auc'] = best_test_auc
                results[test_loader_names[0] + 'loss'] = best_test_loss
                val_auc_list.append(val_auc)

            if np.mean(val_auc_list) > best_val_auc:
                stop_patience = 0
                best_val_auc = np.mean(val_auc_list)
                tests = zip(test_loader_names, test_loaders)
                for test_name, test_loader in tests:
                    test_auc, test_logloss = misc.accuracy(algorithm, test_loader, None, device, args.algorithm)
                    results[test_name + 'auc'] = test_auc
                    results[test_name + 'loss'] = test_logloss
                    best_test_auc,best_test_loss = test_auc, test_logloss
            else:
                stop_patience +=1

            results_keys = sorted(results.keys())
            if results_keys != last_results_keys:
                misc.print_row(results_keys, colwidth=14)
                last_results_keys = results_keys
            misc.print_row([results[key] for key in results_keys],
                           colwidth=14)

            results.update({
                'hparams': hparams,
                'args': vars(args)
            })

            epochs_path = os.path.join(args.output_dir, 'results.jsonl')
            with open(epochs_path, 'a') as f:
                f.write(json.dumps(results, sort_keys=True) + "\n")

            with open(file_name, 'a') as f:
                f.write("\n")
                for k in sorted(results):
                    if "acc" in k or "loss" in k or "step" in k:
                        f.write(k + ": " + str(results[k]) + " ")

            algorithm_dict = algorithm.state_dict()
            start_step = step + 1
            checkpoint_vals = collections.defaultdict(lambda: [])

            if args.save_model_every_checkpoint:
                save_checkpoint('model_step{}.pkl'.format(step))

            if stop_patience >=10:
                break



    save_checkpoint('model.pkl')

    with open(os.path.join(args.output_dir, 'done'), 'w') as f:
        f.write('done')
    print("running time: ", time.time() - train_start)


from torchmeta.modules import (MetaModule, MetaSequential, MetaLinear)
from torchmeta.utils.gradient_based import gradient_update_parameters
from deepctr_torch.inputs import build_input_features, SparseFeat, DenseFeat, VarLenSparseFeat, get_varlen_pooling_list, \
    create_embedding_matrix, varlen_embedding_lookup
import networks
from networks import *
from deepctr_torch.layers import AFMLayer, LogTransformLayer
from deepctr_torch.layers.interaction import CrossNet, InnerProductLayer
from collections import OrderedDict

def compute_irm_penalty(logits, labels):
    scale = torch.tensor(1.0, requires_grad=True)
    loss = nn.BCEWithLogitsLoss()(logits * scale, labels)
    grad = torch.autograd.grad(loss, [scale], create_graph=True)[0]
    return torch.sum(grad ** 2)

def concat_fun(inputs, axis=-1):
    if len(inputs) == 1:
        return inputs[0]
    else:
        return torch.cat(inputs, dim=axis)
def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]

class Algorithm(torch.nn.Module):
    def __init__(self, num_classes, num_domains, hparams):
        super(Algorithm, self).__init__()
        self.hparams = hparams

    def update(self, minibatches, unlabeled=None):
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError


class ERM(Algorithm):
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(ERM, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs + 1 if hparams["dm_idx"] else self.featurizer.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'], num_domains)

        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )

    def update(self, minibatches, unlabeled=None, return_z=False):
        self.device = "cuda" if minibatches[0][0].is_cuda else "cpu"
        T = len(minibatches)
        n = len(minibatches[0][0])

        d = torch.cat([torch.tensor([float(d)] * n).to(self.device) for d in range(T)])

        if self.hparams['dm_idx']:
            all_x = torch.cat([x for x, y, _ in minibatches])
            all_y = torch.cat([y for x, y, _ in minibatches])
            all_z = self.featurizer(all_x)
            all_z = torch.cat([all_z, d], dim=-1)
        else:
            all_x = torch.cat([x for x, y in minibatches])
            all_y = torch.cat([y for x, y in minibatches])
            all_z = self.featurizer(all_x)
        preds = self.classifier(all_z)
        loss = F.cross_entropy(preds, all_y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if return_z:
            return {'loss': loss.item()}, all_z.view(T, n, -1)[T - 1].detach().cpu().numpy(), all_y.view(T, n, -1)[T - 1].detach().cpu().numpy()
        else:
            return {'loss': loss.item()}  #, all_z.view(T, n, -1)[T - 1].detach().cpu().numpy(), all_y.view(T, n, -1)[T - 1].detach().cpu().numpy()

    def predict(self, x, d=None):
        if d!=None:
            z = self.featurizer(x)
            return self.classifier(torch.cat([z, d.unsqueeze(1)], dim=-1))
        else:
            return self.network(x)

    def predict_embs(self, x):
        z = self.featurizer(x)
        preds = self.classifier(z)
        return preds, z

class DeepFM(Algorithm):
    def __init__(self, input_shape, num_classes, feature_index, num_domains, hparams, dnn_feature_columns, linear_feature_columns,model="deepfm",
                 init_std=0.0001):
        super(DeepFM, self).__init__(num_classes, num_domains,hparams)
        self.gradient_update_parameters = gradient_update_parameters
        self.dnn_feature_columns,self.linear_feature_columns = dnn_feature_columns,linear_feature_columns
        self.embedding_dict_linear = create_embedding_matrix(self.linear_feature_columns, init_std, linear=True, sparse=False)
        for tensor in self.embedding_dict_linear.values():
            nn.init.normal_(tensor.weight, mean=0, std=init_std)
        self.embedding_dict_stable = create_embedding_matrix(self.dnn_feature_columns, init_std, sparse=False)  #相当于普通推荐模型中的嵌入层
        self.feature_index = feature_index
        self.criterion = nn.BCEWithLogitsLoss()
        a = 8 * len(self.dnn_feature_columns)
        self.classifier = MetaSequential(
            MetaLinear(a, 256),
            nn.ReLU(),
            MetaLinear(256, 128),
            nn.ReLU(),
            MetaLinear(128, 1))
        # self.classifier = MetaSequential(
        #     MetaLinear(a, 128),
        #     nn.ReLU(),
        #     MetaLinear(128, 64),
        #     nn.ReLU(),
        #     MetaLinear(64, 1))
        self.d = networks.ImprovedDomainAdaptor_Second(a, a, hparams, 8, networks.MLP(a, a, hparams))
        self.classifier_params = [None for _ in range(num_domains + 1)]
        self.linear_params = [None for _ in range(num_domains + 1)]
        if model=='dcn':
            self.crossnet = CrossNet(in_features=a,layer_num=2, parameterization='vector')

        self.optimizer = torch.optim.Adam(
            [{'params': list(self.embedding_dict_stable.parameters()) + list(self.classifier.parameters())+list(self.embedding_dict_linear.parameters())},
             {'params': self.d.parameters(), 'lr': self.hparams["beta"]}],
            lr=self.hparams["beta"],
            weight_decay=self.hparams['weight_decay'])
        self.hparams = hparams

    def input_from_feature_columns_linear(self, X):
        self.sparse_feature_columns = list(
            filter(lambda x: isinstance(x, SparseFeat), self.linear_feature_columns)) if len(self.linear_feature_columns) else []
        self.dense_feature_columns = list(
            filter(lambda x: isinstance(x, DenseFeat), self.linear_feature_columns)) if len(self.linear_feature_columns) else []
        self.varlen_sparse_feature_columns = list(
            filter(lambda x: isinstance(x, VarLenSparseFeat), self.linear_feature_columns)) if len(self.linear_feature_columns) else []
        sparse_embedding_list = [self.embedding_dict_linear[feat.embedding_name](
            X[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]].long()) for
            feat in self.sparse_feature_columns]

        dense_value_list = [X[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]] for feat in
                            self.dense_feature_columns]

        sequence_embed_dict = varlen_embedding_lookup(X, self.embedding_dict_linear, self.feature_index,
                                                      self.varlen_sparse_feature_columns)
        varlen_embedding_list = get_varlen_pooling_list(sequence_embed_dict, X, self.feature_index,
                                                        self.varlen_sparse_feature_columns, self.device)

        sparse_embedding_list += varlen_embedding_list
        sparse_embedding_cat = torch.cat(sparse_embedding_list, dim=-1)#这里相当于对x做了线性嵌入

        return sparse_embedding_cat

    def linear_forward(self, X):  #输入应当是sparse_embedding_cat
        linear_logit = torch.zeros([X.shape[0], 1]).to(self.device)
        sparse_feat_logit = torch.sum(X, dim=-1, keepdim=False)
        linear_logit += sparse_feat_logit

        return linear_logit

    def input_from_feature_columns(self, X, feature_columns, embedding_dict, feature_index, support_dense=True):
        sparse_feature_columns = list(
            filter(lambda x: isinstance(x, SparseFeat), feature_columns)) if len(feature_columns) else []
        dense_feature_columns = list(
            filter(lambda x: isinstance(x, DenseFeat), feature_columns)) if len(feature_columns) else []
        varlen_sparse_feature_columns = list(
            filter(lambda x: isinstance(x, VarLenSparseFeat), feature_columns)) if feature_columns else []
        if not support_dense and len(dense_feature_columns) > 0:
            raise ValueError(
                "DenseFeat is not supported in dnn_feature_columns")
        sparse_embedding_list = [embedding_dict[feat.embedding_name](
            X[:, feature_index[feat.name][0]:feature_index[feat.name][1]].long()) for
            feat in sparse_feature_columns]  # error

        sequence_embed_dict = varlen_embedding_lookup(X, embedding_dict, feature_index,
                                                      varlen_sparse_feature_columns)

        varlen_sparse_embedding_list = get_varlen_pooling_list(sequence_embed_dict, X, feature_index,
                                                               varlen_sparse_feature_columns, device='cuda')

        dense_value_list = [X[:, feature_index[feat.name][0]:feature_index[feat.name][1]] for feat in
                            dense_feature_columns]
        mlp_input = torch.cat(sparse_embedding_list + varlen_sparse_embedding_list, dim=1)
        return mlp_input, dense_value_list

    def deepfm_forward(self, x,z, c_params=None):
        #first_logit
        x = x.view(x.size(0), -1, len(self.linear_feature_columns))
        first_logit = self.linear_forward(x)
        #fm部分
        z = z.view(z.size(0), len(self.dnn_feature_columns), -1)
        square_of_sum = torch.pow(torch.sum(z, dim=1, keepdim=True), 2)
        sum_of_square = torch.sum(z * z, dim=1, keepdim=True)
        cross_term = square_of_sum - sum_of_square
        fm_second_order = 0.5 * torch.sum(cross_term, dim=2, keepdim=False)
        deep_input = torch.flatten(z, start_dim=1)
        deep_output = self.classifier(deep_input, params=c_params)
        # y_pred = fm_first_order + fm_second_order + deep_output
        y_pred = first_logit + fm_second_order + deep_output
        return y_pred

    def update(self, minibatches, dnn_feature_columns, feature_index, unlabeled=None, return_z=False, IL=None):
        self.device = "cuda" if minibatches[0][0].is_cuda else "cpu"
        T = len(minibatches)
        self.T = T
        n = len(minibatches[0][0])

        d = [torch.tensor([float(d)] * n).to(self.device) for d in range(T)]
        y = [y for _, y, _ in minibatches]
        z = []
        z_l = []
        x = []
        for x_, _, _ in minibatches:
            x_emb, dense_value_list = self.input_from_feature_columns(x_, dnn_feature_columns,
                                                                          self.embedding_dict_stable, feature_index)
            x_emb_linear = self.input_from_feature_columns_linear(x_)
            x_emb = x_emb.view(x_emb.size(0), -1)
            z.append(x_emb)
            z_l.append(x_emb_linear.view(x_emb.size(0), -1))

        if IL == 'irm':
            irm_lambda = 1
            total_loss = 0.0
            irm_penalty = 0.0
            for i in range(len(minibatches)):
                num_envs = len(minibatches)
                y_pred = self.deepfm_forward(z_l[i], z[i])
                loss = self.criterion(y_pred.view(-1), y[i].float())
                penalty = compute_irm_penalty(y_pred.view(-1), y[i].float())
                total_loss += loss
                irm_penalty += penalty
            total_loss /= num_envs
            irm_penalty /= num_envs
            final_loss = total_loss + irm_lambda * irm_penalty
            self.step_optim(final_loss)

        elif IL == 'dro':
            alpha = 0.5
            num_envs = len(minibatches)
            env_losses = []
            for i in range(len(minibatches)):
                y_pred = self.deepfm_forward(z_l[i], z[i])
                loss = self.criterion(y_pred.view(-1), y[i].float())
                env_losses.append(loss)
            worst_case_loss = torch.stack(env_losses).max()
            total_loss = (1 - alpha) * sum(env_losses) / num_envs + alpha * worst_case_loss
            self.step_optim(total_loss)

        elif IL == 'dda':
            z_tld = self.dm_adapt(z)
            # z_tld = z # ablation study用到
            objective = 0
            for i in range(len(minibatches)):

                c_params = None
                if i != 0:
                    c_params,inner_loss = self.inner_loop(z_l[i - 1], z[i - 1], z_tld[i - 1], int(d[i][0] - 1), y[
                        i - 1].float())  # 只更新分类模型：模型预测下一时刻样本特征时，使分类模型根据特征预测的结果接近y。不更新特征预测模型self.d
                    y_pred = self.deepfm_forward(z_l[i], z[i], c_params)
                    objective += self.criterion(y_pred.view(-1), y[i].float())  # 用内循环更新后的参数来对z_i预测，计算损失值
                    self.classifier_params[int(d[i][0])] = c_params  # 用内循环更新的参数替换原本参数
                elif i == 0:
                    # feedback
                    y_pred = self.deepfm_forward(z_l[i], z[i], c_params)
                    inner_c_loss = self.criterion(y_pred.view(-1), y[i].float())
                    c_params = self.gradient_update_parameters(self.classifier, inner_c_loss, step_size=self.hparams["alpha"], params=c_params)
                    y_pred_out = self.deepfm_forward(z_l[i], z[i], c_params)
                    objective += self.criterion(y_pred_out.view(-1),y[i].float())
                    self.classifier_params[int(d[i][0])] = c_params

                if i == T - 1:
                    c_params,inner_loss = self.inner_loop(z_l[i], z[i], z_tld[i], int(d[i][0]), y[i].float(),
                                                                     test=True)
                    self.classifier_params[T] = c_params
                    objective += inner_loss

            objective.div_(len(minibatches))
            self.step_optim(objective)

        else:  #erm
            for i in range(len(minibatches)):
                y_pred = self.deepfm_forward(z_l[i], z[i])
                # loss = self.criterion(torch.sigmoid(y_pred).view(-1), y[i].float())
                loss = self.criterion(y_pred.view(-1), y[i].float())
                self.step_optim(loss)

        return 0

    def dm_adapt(self, z):
        outputs = []
        for t in range(1, len(z) + 1):  # 如果是预测下一个领域，就是len(z)+1; 如果是预测下两个领域，就是len(z)+2
            input_seq = torch.stack(z[:t], dim=0).transpose(0, 1)  # (batch_size, t, feature_dim)
            output = self.d(input_seq)
            outputs.append(output[:, -1, :])  # 取最后一个时间步的输出
        return torch.stack(outputs, dim=0)

    def dm_linear_adapt(self, z):
        T = len(z)
        n = len(z[0])
        z_cat = torch.cat(z)
        all_masks = []
        attend_to_domain_embs = False
        for i in range(len(z)):
            if attend_to_domain_embs:
                z_kv = torch.stack([torch.mean(z_i, dim=0) for z_i in z])
                msk = torch.tensor(
                    [[False] if cnt <= i else [True] for cnt in range(T)])
            else:
                msk = torch.tensor(
                    [[False] * n if cnt <= i else [True] * n for cnt in range(T)])
                z_kv = z_cat.detach()
            all_masks.append(msk.flatten().repeat(n, 1))
        all_masks = torch.cat(all_masks).to(self.device)
        return self.d_l(z_cat, z_kv, all_masks).view(T, n, -1)

    def inner_loop(self, x, z, z_tld, d, y, test=False):
        eps = 1e-6
        c_params = None
        tau_temp = 0.5

        logit_r = torch.sigmoid(self.deepfm_forward(x, z, self.classifier_params[d])).view(-1) / self.hparams["tau_temp"]
        # logit_r = self.deepfm_forward(x, z, self.classifier_params[d]).view(-1) / self.hparams["tau_temp"]

        if test:
            iteration = self.hparams['test_step']
        else:
            iteration = self.hparams['train_step']
        for _ in range(iteration):
            logit_p = torch.sigmoid(self.deepfm_forward(x, z_tld,  c_params)).view(-1) / self.hparams["tau_temp"]
            # logit_p = self.deepfm_forward(x, z_tld, c_params).view(-1) / self.hparams["tau_temp"]
            inner_loss = self.hparams["lambda"] * self.criterion(self.deepfm_forward(x, z_tld, c_params).view(-1), y) + (
                                     1 - self.hparams["lambda"]) * self.contrastive_loss(logit_p, logit_r)
            c_params = self.gradient_update_parameters(self.classifier, inner_loss,step_size=self.hparams["alpha"],params=c_params)
        return c_params, inner_loss

    def step_optim(self, objective):
        self.zero_grad()
        objective.backward()
        nn.utils.clip_grad_norm(self.parameters(), self.hparams["clip"])
        self.optimizer.step()

    def predict(self, x, d):
        d = torch.unique(d)
        x_emb, dense_value_list = self.input_from_feature_columns(x, self.dnn_feature_columns,
                                                                      self.embedding_dict_stable,
                                                                      self.feature_index)
        x_emb_linear = self.input_from_feature_columns_linear(x)

        x_emb = x_emb.view(x_emb.size(0), -1)  # 这些相当于 z = self.featurizer(x)
        x_emb_linear = x_emb_linear.view(x_emb.size(0), -1)

        preds = self.deepfm_forward(x_emb_linear, x_emb, self.classifier_params[int(d if d < self.T else -1)]).view(-1)
        return preds

    def get_emb(self,x):
        x_emb, dense_value_list = self.input_from_feature_columns(x, self.dnn_feature_columns,
                                                                  self.embedding_dict_stable,
                                                                  self.feature_index)
        x_emb = x_emb.view(x_emb.size(0), -1)
        return x_emb

    def contrastive_loss(self, logit_p, logit_r):

        logit_p = logit_p.clamp(min=1e-7, max=1 - 1e-7)  # 避免数值不稳定
        logit_r = logit_r.clamp(min=1e-7, max=1 - 1e-7)  # 避免数值不稳定
        loss = F.binary_cross_entropy(logit_p, logit_r.detach())  #binary_cross_entropy的输入是经过sigmoid的


        # logit_p_norm = F.normalize(logit_p, p=2, dim=0)
        # logit_r_norm = F.normalize(logit_r.detach(), p=2, dim=0)
        # cosine_similarity = torch.dot(logit_p_norm, logit_r_norm)
        # loss = 1 - cosine_similarity
        return loss

class DCN(Algorithm):
    def __init__(self, input_shape, num_classes, feature_index, num_domains, hparams, dnn_feature_columns, linear_feature_columns,model="deepfm",
                 init_std=0.0001):
        super(DCN, self).__init__(num_classes, num_domains,hparams)
        self.gradient_update_parameters = gradient_update_parameters
        self.dnn_feature_columns,self.linear_feature_columns = dnn_feature_columns,linear_feature_columns
        self.embedding_dict_linear = create_embedding_matrix(self.linear_feature_columns, init_std, linear=True, sparse=False)
        for tensor in self.embedding_dict_linear.values():
            nn.init.normal_(tensor.weight, mean=0, std=init_std)
        self.embedding_dict_stable = create_embedding_matrix(self.dnn_feature_columns, init_std, sparse=False)  #相当于普通推荐模型中的嵌入层
        self.feature_index = feature_index
        self.criterion = nn.BCEWithLogitsLoss()
        a = 8 * len(self.dnn_feature_columns)
        self.classifier = MetaSequential(
            MetaLinear(a, 256),
            nn.ReLU(),
            MetaLinear(256, 128),
            nn.ReLU(),
            MetaLinear(128, 1))
        self.d = networks.ImprovedDomainAdaptor_Second(a, a, hparams, 8, networks.MLP(a, a, hparams))
        self.classifier_params = [None for _ in range(num_domains + 1)]
        self.linear_params = [None for _ in range(num_domains + 1)]
        self.crossnet = CrossNet(in_features=a,layer_num=2, parameterization='vector')
        self.linear = nn.Linear(a, 1, bias=True)
        self.optimizer = torch.optim.Adam(
            [{'params': list(self.embedding_dict_stable.parameters()) + list(self.classifier.parameters())+list(self.embedding_dict_linear.parameters())},
             {'params': self.d.parameters(), 'lr': self.hparams["beta"]}],
            lr=self.hparams["beta"],
            weight_decay=self.hparams['weight_decay'])
        self.hparams = hparams

    def input_from_feature_columns_linear(self, X):
        self.sparse_feature_columns = list(
            filter(lambda x: isinstance(x, SparseFeat), self.linear_feature_columns)) if len(self.linear_feature_columns) else []
        self.dense_feature_columns = list(
            filter(lambda x: isinstance(x, DenseFeat), self.linear_feature_columns)) if len(self.linear_feature_columns) else []
        self.varlen_sparse_feature_columns = list(
            filter(lambda x: isinstance(x, VarLenSparseFeat), self.linear_feature_columns)) if len(self.linear_feature_columns) else []
        sparse_embedding_list = [self.embedding_dict_linear[feat.embedding_name](
            X[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]].long()) for
            feat in self.sparse_feature_columns]

        dense_value_list = [X[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]] for feat in
                            self.dense_feature_columns]

        sequence_embed_dict = varlen_embedding_lookup(X, self.embedding_dict_linear, self.feature_index,
                                                      self.varlen_sparse_feature_columns)
        varlen_embedding_list = get_varlen_pooling_list(sequence_embed_dict, X, self.feature_index,
                                                        self.varlen_sparse_feature_columns, self.device)

        sparse_embedding_list += varlen_embedding_list
        sparse_embedding_cat = torch.cat(sparse_embedding_list, dim=-1)

        return sparse_embedding_cat

    def linear_forward(self, X):
        linear_logit = torch.zeros([X.shape[0], 1]).to(self.device)
        sparse_feat_logit = torch.sum(X, dim=-1, keepdim=False)
        linear_logit += sparse_feat_logit

        return linear_logit

    def input_from_feature_columns(self, X, feature_columns, embedding_dict, feature_index, support_dense=True):
        sparse_feature_columns = list(
            filter(lambda x: isinstance(x, SparseFeat), feature_columns)) if len(feature_columns) else []
        dense_feature_columns = list(
            filter(lambda x: isinstance(x, DenseFeat), feature_columns)) if len(feature_columns) else []
        varlen_sparse_feature_columns = list(
            filter(lambda x: isinstance(x, VarLenSparseFeat), feature_columns)) if feature_columns else []
        if not support_dense and len(dense_feature_columns) > 0:
            raise ValueError(
                "DenseFeat is not supported in dnn_feature_columns")
        sparse_embedding_list = [embedding_dict[feat.embedding_name](
            X[:, feature_index[feat.name][0]:feature_index[feat.name][1]].long()) for
            feat in sparse_feature_columns]  # error

        sequence_embed_dict = varlen_embedding_lookup(X, embedding_dict, feature_index,
                                                      varlen_sparse_feature_columns)

        varlen_sparse_embedding_list = get_varlen_pooling_list(sequence_embed_dict, X, feature_index,
                                                               varlen_sparse_feature_columns, device='cuda')

        dense_value_list = [X[:, feature_index[feat.name][0]:feature_index[feat.name][1]] for feat in
                            dense_feature_columns]
        mlp_input = torch.cat(sparse_embedding_list + varlen_sparse_embedding_list, dim=1)
        return mlp_input, dense_value_list

    def dcn_forward(self,x,z,c_params=None):
        x = x.view(x.size(0), -1, len(self.linear_feature_columns))
        logit = self.linear_forward(x)
        z = z.view(z.size(0), len(self.dnn_feature_columns), -1)
        deep_input = torch.flatten(z, start_dim=1)
        deep_out = self.classifier(deep_input, params=c_params)
        cross_out = self.crossnet(deep_input)
        cross_pred = self.linear(cross_out)
        pred = deep_out + cross_pred + logit
        return pred

    def update(self, minibatches, dnn_feature_columns, feature_index, unlabeled=None, return_z=False, IL=None,model="deepfm"):
        self.device = "cuda" if minibatches[0][0].is_cuda else "cpu"
        T = len(minibatches)
        self.T = T
        n = len(minibatches[0][0])

        d = [torch.tensor([float(d)] * n).to(self.device) for d in range(T)]
        y = [y for _, y, _ in minibatches]
        z = []
        z_l = []
        x = []
        for x_, _, _ in minibatches:
            x_emb, dense_value_list = self.input_from_feature_columns(x_, dnn_feature_columns,
                                                                          self.embedding_dict_stable, feature_index)
            x_emb_linear = self.input_from_feature_columns_linear(x_)
            x_emb = x_emb.view(x_emb.size(0), -1)
            z.append(x_emb)
            z_l.append(x_emb_linear.view(x_emb.size(0), -1))

        if IL == 'irm':
            irm_lambda = 1
            total_loss = 0.0
            irm_penalty = 0.0
            for i in range(len(minibatches)):
                num_envs = len(minibatches)

                y_pred = self.dcn_forward(z_l[i], z[i])
                loss = self.criterion(y_pred.view(-1), y[i].float())
                penalty = compute_irm_penalty(y_pred.view(-1), y[i].float())

                total_loss += loss
                irm_penalty += penalty

            total_loss /= num_envs
            irm_penalty /= num_envs

            final_loss = total_loss + irm_lambda * irm_penalty
            self.step_optim(final_loss)


        elif IL == 'dro':
            alpha = 0.5
            num_envs = len(minibatches)
            env_losses = []
            for i in range(len(minibatches)):
                y_pred = self.dcn_forward(z_l[i], z[i])
                loss = self.criterion(y_pred.view(-1), y[i].float())
                env_losses.append(loss)

            worst_case_loss = torch.stack(env_losses).max()
            total_loss = (1 - alpha) * sum(env_losses) / num_envs + alpha * worst_case_loss
            self.step_optim(total_loss)

        elif IL == "dda":
            z_tld = self.dm_adapt(z)
            # z_tld = z
            objective = 0
            for i in range(len(minibatches)):

                c_params = None
                if i != 0:
                    c_params, inner_loss = self.inner_loop(z_l[i - 1], z[i - 1], z_tld[i - 1], int(d[i][0] - 1), y[
                        i - 1].float())
                    y_pred = self.dcn_forward(z_l[i], z[i], c_params)
                    objective += self.criterion((y_pred).view(-1),
                                                y[i].float())
                    self.classifier_params[int(d[i][0])] = c_params
                elif i == 0:
                    # feedback
                    y_pred = self.dcn_forward(z_l[i], z[i], c_params)
                    inner_c_loss = self.criterion((y_pred).view(-1), y[i].float())
                    c_params = self.gradient_update_parameters(self.classifier, inner_c_loss,
                                                               step_size=self.hparams["alpha"], params=c_params)
                    y_pred_out = self.dcn_forward(z_l[i], z[i], c_params)
                    objective += self.criterion((y_pred_out).view(-1), y[i].float())
                    self.classifier_params[int(d[i][0])] = c_params

                if i == T - 1:
                    c_params, inner_loss = self.inner_loop(z_l[i], z[i], z_tld[i], int(d[i][0]), y[i].float(),
                                                           test=True)
                    self.classifier_params[T] = c_params
                    objective += inner_loss

            objective.div_(len(minibatches))
            self.step_optim(objective)

        else:
            for i in range(len(minibatches)):
                y_pred = self.dcn_forward(z_l[i], z[i])
                # loss = self.criterion(torch.sigmoid(y_pred).view(-1), y[i].float())
                loss = self.criterion(y_pred.view(-1), y[i].float())
                self.step_optim(loss)

        return 0

    def dm_adapt(self, z):
        outputs = []
        for t in range(1, len(z) + 1):
            input_seq = torch.stack(z[:t], dim=0).transpose(0, 1)
            output = self.d(input_seq)
            outputs.append(output[:, -1, :])
        return torch.stack(outputs, dim=0)

    def dm_linear_adapt(self, z):
        T = len(z)
        n = len(z[0])
        z_cat = torch.cat(z)
        all_masks = []
        attend_to_domain_embs = False
        for i in range(len(z)):
            if attend_to_domain_embs:
                z_kv = torch.stack([torch.mean(z_i, dim=0) for z_i in z])
                msk = torch.tensor(
                    [[False] if cnt <= i else [True] for cnt in range(T)])
            else:
                msk = torch.tensor(
                    [[False] * n if cnt <= i else [True] * n for cnt in range(T)])
                z_kv = z_cat.detach()
            all_masks.append(msk.flatten().repeat(n, 1))
        all_masks = torch.cat(all_masks).to(self.device)
        return self.d_l(z_cat, z_kv, all_masks).view(T, n, -1)

    def inner_loop(self, x, z, z_tld, d, y, test=False):
        eps = 1e-6
        c_params = None
        tau_temp = 0.5

        logit_r = torch.sigmoid(self.dcn_forward(x, z, self.classifier_params[d])).view(-1) / self.hparams["tau_temp"]

        if test:
            iteration = self.hparams['test_step']
        else:
            iteration = self.hparams['train_step']
        for _ in range(iteration):
            # logit_p = torch.sigmoid(self.classifier(z_tld, params=c_params) / self.hparams["tau_temp"]).view(-1) + eps
            logit_p = torch.sigmoid(self.dcn_forward(x, z_tld,  c_params)).view(-1) / self.hparams["tau_temp"]
            inner_loss = self.hparams["lambda"] * self.criterion((self.dcn_forward(x, z_tld, c_params)).view(-1), y) + (
                                     1 - self.hparams["lambda"]) * self.contrastive_loss(logit_p, logit_r, tau_temp)
            c_params = self.gradient_update_parameters(self.classifier, inner_loss,step_size=self.hparams["alpha"],params=c_params)
            # l_params = self.gradient_update_parameters(self.linear, inner_loss,step_size=self.hparams["alpha"],params=l_params)
        return c_params, inner_loss

    def step_optim(self, objective):
        self.zero_grad()
        objective.backward()
        nn.utils.clip_grad_norm(self.parameters(), self.hparams["clip"])
        self.optimizer.step()

    def predict(self, x, d):
        d = torch.unique(d)
        x_emb, dense_value_list = self.input_from_feature_columns(x, self.dnn_feature_columns,
                                                                      self.embedding_dict_stable,
                                                                      self.feature_index)
        x_emb_linear = self.input_from_feature_columns_linear(x)

        x_emb = x_emb.view(x_emb.size(0), -1)
        x_emb_linear = x_emb_linear.view(x_emb.size(0), -1)

        preds = self.dcn_forward(x_emb_linear, x_emb, self.classifier_params[int(d if d < self.T else -1)]).view(-1)
        return preds

    def contrastive_loss(self, logit_p, logit_r, temperature=0.5):

        logit_p = logit_p.clamp(min=1e-7, max=1 - 1e-7)
        logit_r = logit_r.clamp(min=1e-7, max=1 - 1e-7)
        loss = F.binary_cross_entropy(logit_p, logit_r.detach())
        return loss

class AFM(Algorithm):
    def __init__(self, input_shape, num_classes, feature_index, num_domains, hparams, dnn_feature_columns, linear_feature_columns,model="deepfm",
                 init_std=0.0001):
        super(AFM, self).__init__(num_classes, num_domains,hparams)
        self.dnn_feature_columns,self.linear_feature_columns = dnn_feature_columns,linear_feature_columns
        self.embedding_dict_linear = create_embedding_matrix(self.linear_feature_columns, init_std, linear=True, sparse=False)
        for tensor in self.embedding_dict_linear.values():
            nn.init.normal_(tensor.weight, mean=0, std=init_std)
        self.embedding_dict_stable = create_embedding_matrix(self.dnn_feature_columns, init_std, sparse=False)
        self.feature_index = feature_index
        self.criterion = nn.BCEWithLogitsLoss()
        embedding_size = 8
        a = embedding_size * len(self.dnn_feature_columns)
        self.d = networks.ImprovedDomainAdaptor_Second(a, a, hparams, 8, networks.MLP(a, a, hparams))
        self.classifier_params = [None for _ in range(num_domains + 1)]
        self.linear_params = [None for _ in range(num_domains + 1)]
        self.classifier = AFMLayer(embedding_size)
        self.linear = nn.Linear(a, 1, bias=True)
        self.optimizer = torch.optim.Adam(
            [{'params': list(self.embedding_dict_stable.parameters()) + list(self.classifier.parameters())+list(self.embedding_dict_linear.parameters())},
             {'params': self.d.parameters(), 'lr': self.hparams["beta"]}],
            lr=self.hparams["beta"],
            weight_decay=self.hparams['weight_decay'])
        self.hparams = hparams

    def input_from_feature_columns_linear(self, X):
        self.sparse_feature_columns = list(
            filter(lambda x: isinstance(x, SparseFeat), self.linear_feature_columns)) if len(self.linear_feature_columns) else []
        self.dense_feature_columns = list(
            filter(lambda x: isinstance(x, DenseFeat), self.linear_feature_columns)) if len(self.linear_feature_columns) else []
        self.varlen_sparse_feature_columns = list(
            filter(lambda x: isinstance(x, VarLenSparseFeat), self.linear_feature_columns)) if len(self.linear_feature_columns) else []
        sparse_embedding_list = [self.embedding_dict_linear[feat.embedding_name](
            X[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]].long()) for
            feat in self.sparse_feature_columns]

        dense_value_list = [X[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]] for feat in
                            self.dense_feature_columns]

        sequence_embed_dict = varlen_embedding_lookup(X, self.embedding_dict_linear, self.feature_index,
                                                      self.varlen_sparse_feature_columns)
        varlen_embedding_list = get_varlen_pooling_list(sequence_embed_dict, X, self.feature_index,
                                                        self.varlen_sparse_feature_columns, self.device)

        sparse_embedding_list += varlen_embedding_list
        sparse_embedding_cat = torch.cat(sparse_embedding_list, dim=-1)

        return sparse_embedding_cat

    def linear_forward(self, X):
        linear_logit = torch.zeros([X.shape[0], 1]).to(self.device)
        sparse_feat_logit = torch.sum(X, dim=-1, keepdim=False)
        linear_logit += sparse_feat_logit

        return linear_logit

    def input_from_feature_columns(self, X, feature_columns, embedding_dict, feature_index, support_dense=True):
        sparse_feature_columns = list(
            filter(lambda x: isinstance(x, SparseFeat), feature_columns)) if len(feature_columns) else []
        dense_feature_columns = list(
            filter(lambda x: isinstance(x, DenseFeat), feature_columns)) if len(feature_columns) else []
        varlen_sparse_feature_columns = list(
            filter(lambda x: isinstance(x, VarLenSparseFeat), feature_columns)) if feature_columns else []
        if not support_dense and len(dense_feature_columns) > 0:
            raise ValueError(
                "DenseFeat is not supported in dnn_feature_columns")
        sparse_embedding_list = [embedding_dict[feat.embedding_name](
            X[:, feature_index[feat.name][0]:feature_index[feat.name][1]].long()) for
            feat in sparse_feature_columns]  # error
        sequence_embed_dict = varlen_embedding_lookup(X, embedding_dict, feature_index,
                                                      varlen_sparse_feature_columns)
        varlen_sparse_embedding_list = get_varlen_pooling_list(sequence_embed_dict, X, feature_index,
                                                               varlen_sparse_feature_columns, device='cuda')
        dense_value_list = [X[:, feature_index[feat.name][0]:feature_index[feat.name][1]] for feat in
                            dense_feature_columns]
        mlp_input = torch.cat(sparse_embedding_list + varlen_sparse_embedding_list, dim=1)
        return mlp_input, dense_value_list

    def afm_forward(self,x,z,c_params=None):
        z = z.view(z.size(0), len(self.dnn_feature_columns), -1)
        z_list = torch.split(z, 1, dim=1)
        x = x.view(x.size(0), -1, len(self.linear_feature_columns))
        logit = self.linear_forward(x)
        if not c_params:
            afmlayer_out = self.classifier(z_list)
        else:
            original_params = {name: param.clone() for name, param in self.classifier.named_parameters()}
            for param_name, param_value in c_params.items():
                if hasattr(self.classifier, param_name):
                    setattr(self.classifier, param_name, nn.Parameter(param_value))
                else:
                    raise ValueError(f"AFMLayer has no attribute named '{param_name}'")
            afmlayer_out = self.classifier(z_list)
            for param_name, original_value in original_params.items():
                setattr(self.classifier, param_name, nn.Parameter(original_value))
        y_pred = logit + afmlayer_out
        return y_pred

    def update(self, minibatches, dnn_feature_columns, feature_index, unlabeled=None, return_z=False, IL=None,model="deepfm"):
        self.device = "cuda" if minibatches[0][0].is_cuda else "cpu"
        T = len(minibatches)
        self.T = T
        n = len(minibatches[0][0])

        d = [torch.tensor([float(d)] * n).to(self.device) for d in range(T)]
        y = [y for _, y, _ in minibatches]
        z = []
        z_l = []
        x = []
        for x_, _, _ in minibatches:
            x_emb, dense_value_list = self.input_from_feature_columns(x_, dnn_feature_columns,
                                                                          self.embedding_dict_stable, feature_index)
            x_emb_linear = self.input_from_feature_columns_linear(x_)
            x_emb = x_emb.view(x_emb.size(0), -1)
            z.append(x_emb)
            z_l.append(x_emb_linear.view(x_emb.size(0), -1))

        if IL == 'irm':
            irm_lambda = 1
            total_loss = 0.0
            irm_penalty = 0.0
            for i in range(len(minibatches)):
                num_envs = len(minibatches)
                y_pred = self.afm_forward(z_l[i], z[i])
                loss = self.criterion(y_pred.view(-1), y[i].float())
                penalty = compute_irm_penalty(y_pred.view(-1), y[i].float())
                total_loss += loss
                irm_penalty += penalty
            total_loss /= num_envs
            irm_penalty /= num_envs
            final_loss = total_loss + irm_lambda * irm_penalty
            self.step_optim(final_loss)

        elif IL == 'dro':
            alpha = 0.5
            num_envs = len(minibatches)
            env_losses = []
            for i in range(len(minibatches)):
                y_pred = self.afm_forward(z_l[i], z[i])
                loss = self.criterion(y_pred.view(-1), y[i].float())
                env_losses.append(loss)
            worst_case_loss = torch.stack(env_losses).max()
            total_loss = (1 - alpha) * sum(env_losses) / num_envs + alpha * worst_case_loss
            self.step_optim(total_loss)

        elif IL == 'dda':
            z_tld = self.dm_adapt(z)
            objective = 0
            for i in range(len(minibatches)):
                c_params = None
                if i != 0:
                    c_params, inner_loss = self.inner_loop(z_l[i - 1], z[i - 1], z_tld[i - 1], int(d[i][0] - 1), y[
                        i - 1].float())
                    y_pred = self.afm_forward(z_l[i], z[i], c_params)
                    objective += self.criterion((y_pred).view(-1),
                                                y[i].float())
                    self.classifier_params[int(d[i][0])] = c_params
                elif i == 0:
                    # feedback
                    y_pred = self.afm_forward(z_l[i], z[i], c_params)
                    inner_c_loss = self.criterion((y_pred).view(-1), y[i].float())
                    c_params = self.gradient_update_parameters(self.classifier, inner_c_loss,
                                                               step_size=self.hparams["alpha"], params=c_params)
                    y_pred_out = self.afm_forward(z_l[i], z[i], c_params)
                    objective += self.criterion((y_pred_out).view(-1), y[i].float())
                    self.classifier_params[int(d[i][0])] = c_params

                if i == T - 1:
                    c_params, inner_loss = self.inner_loop(z_l[i], z[i], z_tld[i], int(d[i][0]), y[i].float(),
                                                           test=True)
                    self.classifier_params[T] = c_params
                    objective += inner_loss
            objective.div_(len(minibatches))
            self.step_optim(objective)

        else:
            for i in range(len(minibatches)):
                y_pred = self.afm_forward(z_l[i], z[i])
                loss = self.criterion((y_pred).view(-1), y[i].float())
                self.step_optim(loss)

        return 0

    def dm_adapt(self, z):
        outputs = []
        for t in range(1, len(z) + 1):
            input_seq = torch.stack(z[:t], dim=0).transpose(0, 1)  # (batch_size, t, feature_dim)
            output = self.d(input_seq)
            outputs.append(output[:, -1, :])
        return torch.stack(outputs, dim=0)

    def dm_linear_adapt(self, z):
        T = len(z)
        n = len(z[0])
        z_cat = torch.cat(z)
        all_masks = []
        attend_to_domain_embs = False
        for i in range(len(z)):
            if attend_to_domain_embs:
                z_kv = torch.stack([torch.mean(z_i, dim=0) for z_i in z])
                msk = torch.tensor(
                    [[False] if cnt <= i else [True] for cnt in range(T)])
            else:
                msk = torch.tensor(
                    [[False] * n if cnt <= i else [True] * n for cnt in range(T)])
                z_kv = z_cat.detach()
            all_masks.append(msk.flatten().repeat(n, 1))
        all_masks = torch.cat(all_masks).to(self.device)
        return self.d_l(z_cat, z_kv, all_masks).view(T, n, -1)

    def inner_loop(self, x, z, z_tld, d, y, test=False):
        eps = 1e-6
        c_params = None
        tau_temp = 0.5

        logit_r = torch.sigmoid(self.afm_forward(x, z, self.classifier_params[d])).view(-1) / self.hparams["tau_temp"]

        if test:
            iteration = self.hparams['test_step']
        else:
            iteration = self.hparams['train_step']
        for _ in range(iteration):
            # logit_p = torch.sigmoid(self.classifier(z_tld, params=c_params) / self.hparams["tau_temp"]).view(-1) + eps
            logit_p = torch.sigmoid(self.afm_forward(x, z_tld,  c_params)).view(-1) / self.hparams["tau_temp"]
            inner_loss = self.hparams["lambda"] * self.criterion((self.afm_forward(x, z_tld, c_params)).view(-1), y) + (
                                     1 - self.hparams["lambda"]) * self.contrastive_loss(logit_p, logit_r, tau_temp)
            c_params = self.gradient_update_parameters(self.classifier, inner_loss,step_size=self.hparams["alpha"],params=c_params)
            # l_params = self.gradient_update_parameters(self.linear, inner_loss,step_size=self.hparams["alpha"],params=l_params)
        return c_params, inner_loss

    def step_optim(self, objective):
        self.zero_grad()
        objective.backward()
        nn.utils.clip_grad_norm(self.parameters(), self.hparams["clip"])
        self.optimizer.step()

    def gradient_update_parameters(self, model, loss, step_size, params=None):
        if params is None:
            params = OrderedDict(model.named_parameters())

        grads = torch.autograd.grad(loss, params.values(), create_graph=True, allow_unused=True)

        updated_params = OrderedDict()
        for (name, param), grad in zip(params.items(), grads):
            if grad is None:
                updated_params[name] = param
            else:
                updated_params[name] = param - step_size * grad

        return updated_params

    def predict(self, x, d):
        '''
        only test on the test environment
        :param x:
        :param d: domain index
        :return:
        '''
        d = torch.unique(d)
        x_emb, dense_value_list = self.input_from_feature_columns(x, self.dnn_feature_columns,
                                                                      self.embedding_dict_stable,
                                                                      self.feature_index)
        x_emb_linear = self.input_from_feature_columns_linear(x)

        x_emb = x_emb.view(x_emb.size(0), -1)
        x_emb_linear = x_emb_linear.view(x_emb.size(0), -1)

        preds = self.afm_forward(x_emb_linear, x_emb, self.classifier_params[int(d if d < self.T else -1)]).view(-1)
        return preds

    def contrastive_loss(self, logit_p, logit_r, temperature=0.5):

        logit_p = logit_p.clamp(min=1e-7, max=1 - 1e-7)
        logit_r = logit_r.clamp(min=1e-7, max=1 - 1e-7)
        loss = F.binary_cross_entropy(logit_p, logit_r.detach())
        return loss

class PNN(Algorithm):
    def __init__(self, input_shape, num_classes, feature_index, num_domains, hparams, dnn_feature_columns, linear_feature_columns,model="deepfm",
                 init_std=0.0001):
        super(PNN, self).__init__(num_classes, num_domains,hparams)
        self.gradient_update_parameters = gradient_update_parameters
        self.dnn_feature_columns,self.linear_feature_columns = dnn_feature_columns,linear_feature_columns
        self.embedding_dict_linear = create_embedding_matrix(self.linear_feature_columns, init_std, linear=True, sparse=False)
        for tensor in self.embedding_dict_linear.values():
            nn.init.normal_(tensor.weight, mean=0, std=init_std)
        self.embedding_dict_stable = create_embedding_matrix(self.dnn_feature_columns, init_std, sparse=False)
        self.feature_index = feature_index
        self.criterion = nn.BCEWithLogitsLoss()
        self.innerproduct = InnerProductLayer()
        # self.criterion = F.binary_cross_entropy_with_logits
        a = 8 * len(self.dnn_feature_columns)
        self.classifier = MetaSequential(
            MetaLinear(162, 256),
            nn.ReLU(),
            MetaLinear(256, 128),
            nn.ReLU(),
            MetaLinear(128, 1))
        self.d = networks.ImprovedDomainAdaptor_Second(a, a, hparams, 8, networks.MLP(a, a, hparams))
        self.classifier_params = [None for _ in range(num_domains + 1)]
        self.linear_params = [None for _ in range(num_domains + 1)]
        if model=='dcn':
            self.crossnet = CrossNet(in_features=a,layer_num=2, parameterization='vector')

        self.optimizer = torch.optim.Adam(
            [{'params': list(self.embedding_dict_stable.parameters()) + list(self.classifier.parameters())+list(self.embedding_dict_linear.parameters())},
             {'params': self.d.parameters(), 'lr': self.hparams["beta"]}],
            lr=self.hparams["beta"],
            weight_decay=self.hparams['weight_decay'])
        self.hparams = hparams

    def input_from_feature_columns_linear(self, X):
        self.sparse_feature_columns = list(
            filter(lambda x: isinstance(x, SparseFeat), self.linear_feature_columns)) if len(self.linear_feature_columns) else []
        self.dense_feature_columns = list(
            filter(lambda x: isinstance(x, DenseFeat), self.linear_feature_columns)) if len(self.linear_feature_columns) else []
        self.varlen_sparse_feature_columns = list(
            filter(lambda x: isinstance(x, VarLenSparseFeat), self.linear_feature_columns)) if len(self.linear_feature_columns) else []
        sparse_embedding_list = [self.embedding_dict_linear[feat.embedding_name](
            X[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]].long()) for
            feat in self.sparse_feature_columns]

        dense_value_list = [X[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]] for feat in
                            self.dense_feature_columns]

        sequence_embed_dict = varlen_embedding_lookup(X, self.embedding_dict_linear, self.feature_index,
                                                      self.varlen_sparse_feature_columns)
        varlen_embedding_list = get_varlen_pooling_list(sequence_embed_dict, X, self.feature_index,
                                                        self.varlen_sparse_feature_columns, self.device)

        sparse_embedding_list += varlen_embedding_list
        sparse_embedding_cat = torch.cat(sparse_embedding_list, dim=-1)

        return sparse_embedding_cat

    def linear_forward(self, X):
        linear_logit = torch.zeros([X.shape[0], 1]).to(self.device)
        sparse_feat_logit = torch.sum(X, dim=-1, keepdim=False)
        linear_logit += sparse_feat_logit

        return linear_logit

    def input_from_feature_columns(self, X, feature_columns, embedding_dict, feature_index, support_dense=True):
        sparse_feature_columns = list(
            filter(lambda x: isinstance(x, SparseFeat), feature_columns)) if len(feature_columns) else []
        dense_feature_columns = list(
            filter(lambda x: isinstance(x, DenseFeat), feature_columns)) if len(feature_columns) else []
        varlen_sparse_feature_columns = list(
            filter(lambda x: isinstance(x, VarLenSparseFeat), feature_columns)) if feature_columns else []
        if not support_dense and len(dense_feature_columns) > 0:
            raise ValueError(
                "DenseFeat is not supported in dnn_feature_columns")
        sparse_embedding_list = [embedding_dict[feat.embedding_name](
            X[:, feature_index[feat.name][0]:feature_index[feat.name][1]].long()) for
            feat in sparse_feature_columns]

        sequence_embed_dict = varlen_embedding_lookup(X, embedding_dict, feature_index,
                                                      varlen_sparse_feature_columns)

        varlen_sparse_embedding_list = get_varlen_pooling_list(sequence_embed_dict, X, feature_index,
                                                               varlen_sparse_feature_columns, device='cuda')

        dense_value_list = [X[:, feature_index[feat.name][0]:feature_index[feat.name][1]] for feat in
                            dense_feature_columns]
        mlp_input = torch.cat(sparse_embedding_list + varlen_sparse_embedding_list, dim=1)
        return mlp_input, dense_value_list

    def combined_dnn_input(self, sparse_embedding_list, dense_value_list):
        if len(sparse_embedding_list) > 0 and len(dense_value_list) > 0:
            sparse_dnn_input = torch.flatten(
                torch.cat(sparse_embedding_list, dim=-1), start_dim=1)
            dense_dnn_input = torch.flatten(
                torch.cat(dense_value_list, dim=-1), start_dim=1)
            return concat_fun([sparse_dnn_input, dense_dnn_input])
        elif len(sparse_embedding_list) > 0:
            return torch.flatten(torch.cat(sparse_embedding_list, dim=-1), start_dim=1)
        elif len(dense_value_list) > 0:
            return torch.flatten(torch.cat(dense_value_list, dim=-1), start_dim=1)
        else:
            raise NotImplementedError

    def forward(self, x,z, c_params=None):
        z_ = z.view(z.size(0), len(self.dnn_feature_columns), -1)
        z_list = list(torch.split(z_, 1, dim=1))
        inner_product = torch.flatten(self.innerproduct(z_list), start_dim=1)
        product_layer = torch.cat([z, inner_product], dim=1)
        dnn_input = product_layer
        y_pred = self.classifier(dnn_input, c_params)
        return y_pred

    def update(self, minibatches, dnn_feature_columns, feature_index, unlabeled=None, return_z=False, IL=None):
        self.device = "cuda" if minibatches[0][0].is_cuda else "cpu"
        T = len(minibatches)
        self.T = T
        n = len(minibatches[0][0])

        d = [torch.tensor([float(d)] * n).to(self.device) for d in range(T)]
        y = [y for _, y, _ in minibatches]
        z = []
        z_l = []
        x = []
        for x_, _, _ in minibatches:
            x_emb, dense_value_list = self.input_from_feature_columns(x_, dnn_feature_columns,
                                                                          self.embedding_dict_stable, feature_index)
            x_emb_linear = self.input_from_feature_columns_linear(x_)
            x_emb = x_emb.view(x_emb.size(0), -1)
            z.append(x_emb)
            z_l.append(x_emb_linear.view(x_emb.size(0), -1))

        if IL == 'irm':
            irm_lambda = 1
            total_loss = 0.0
            irm_penalty = 0.0
            for i in range(len(minibatches)):
                num_envs = len(minibatches)
                y_pred = self.forward(z_l[i], z[i])
                loss = self.criterion(y_pred.view(-1), y[i].float())
                penalty = compute_irm_penalty(y_pred.view(-1), y[i].float())
                total_loss += loss
                irm_penalty += penalty
            total_loss /= num_envs
            irm_penalty /= num_envs
            final_loss = total_loss + irm_lambda * irm_penalty
            self.step_optim(final_loss)

        elif IL == 'dro':
            alpha = 0.5
            num_envs = len(minibatches)
            env_losses = []
            for i in range(len(minibatches)):
                y_pred = self.forward(z_l[i], z[i])
                loss = self.criterion(y_pred.view(-1), y[i].float())
                env_losses.append(loss)
            worst_case_loss = torch.stack(env_losses).max()
            total_loss = (1 - alpha) * sum(env_losses) / num_envs + alpha * worst_case_loss
            self.step_optim(total_loss)

        elif IL == 'dda':
            z_tld = self.dm_adapt(z)
            objective = 0
            for i in range(len(minibatches)):

                c_params = None
                if i != 0:
                    c_params,inner_loss = self.inner_loop(z_l[i - 1], z[i - 1], z_tld[i - 1], int(d[i][0] - 1), y[
                        i - 1].float())
                    y_pred = self.forward(z_l[i], z[i], c_params)
                    objective += self.criterion(y_pred.view(-1), y[i].float())
                    self.classifier_params[int(d[i][0])] = c_params
                elif i == 0:
                    # feedback
                    y_pred = self.forward(z_l[i], z[i], c_params)
                    inner_c_loss = self.criterion(y_pred.view(-1), y[i].float())
                    c_params = self.gradient_update_parameters(self.classifier, inner_c_loss, step_size=self.hparams["alpha"], params=c_params)
                    y_pred_out = self.forward(z_l[i], z[i], c_params)
                    objective += self.criterion(y_pred_out.view(-1),y[i].float())
                    self.classifier_params[int(d[i][0])] = c_params

                if i == T - 1:
                    c_params,inner_loss = self.inner_loop(z_l[i], z[i], z_tld[i], int(d[i][0]), y[i].float(),
                                                                     test=True)
                    self.classifier_params[T] = c_params
                    objective += inner_loss

            objective.div_(len(minibatches))
            self.step_optim(objective)

        else:
            for i in range(len(minibatches)):
                y_pred = self.forward(z_l[i], z[i])
                loss = self.criterion(y_pred.view(-1), y[i].float())
                self.step_optim(loss)

        return 0

    def dm_adapt(self, z):
        outputs = []
        for t in range(1, len(z) + 1):
            input_seq = torch.stack(z[:t], dim=0).transpose(0, 1)
            output = self.d(input_seq)
            outputs.append(output[:, -1, :])
        return torch.stack(outputs, dim=0)

    def dm_linear_adapt(self, z):
        T = len(z)
        n = len(z[0])
        z_cat = torch.cat(z)
        all_masks = []
        attend_to_domain_embs = False
        for i in range(len(z)):
            if attend_to_domain_embs:
                z_kv = torch.stack([torch.mean(z_i, dim=0) for z_i in z])
                msk = torch.tensor(
                    [[False] if cnt <= i else [True] for cnt in range(T)])
            else:
                msk = torch.tensor(
                    [[False] * n if cnt <= i else [True] * n for cnt in range(T)])
                z_kv = z_cat.detach()
            all_masks.append(msk.flatten().repeat(n, 1))
        all_masks = torch.cat(all_masks).to(self.device)
        return self.d_l(z_cat, z_kv, all_masks).view(T, n, -1)

    def inner_loop(self, x, z, z_tld, d, y, test=False):
        eps = 1e-8
        c_params = None
        tau_temp = 0.5
        logit_r = torch.sigmoid(self.forward(x, z, self.classifier_params[d])).view(-1) / self.hparams["tau_temp"]
        if test:
            iteration = self.hparams['test_step']
        else:
            iteration = self.hparams['train_step']
        for _ in range(iteration):
            logit_p = torch.sigmoid(self.forward(x, z_tld,  c_params)).view(-1) / (self.hparams["tau_temp"])
            bce_loss = self.hparams["lambda"] * self.criterion((self.forward(x, z_tld, c_params)).view(-1), y)
            regularization = (1 - self.hparams["lambda"]) * self.contrastive_loss(logit_p, logit_r, tau_temp)
            inner_loss = bce_loss + regularization
            if torch.isnan(inner_loss).any():
                print("logit_r contains NaN values")
            if torch.isinf(inner_loss).any():
                print("logit_r contains Inf values")
            c_params = self.gradient_update_parameters(self.classifier, inner_loss,step_size=self.hparams["alpha"],params=c_params)

        return c_params, inner_loss

    def step_optim(self, objective):
        self.zero_grad()
        objective.backward()
        nn.utils.clip_grad_norm(self.parameters(), self.hparams["clip"])
        self.optimizer.step()

        for name, param in self.named_parameters():
            if torch.isnan(param).any():
                print(f"Parameter {name} contains NaN values")
                print(param[torch.isnan(param)])

    def predict(self, x, d):
        d = torch.unique(d)
        x_emb, dense_value_list = self.input_from_feature_columns(x, self.dnn_feature_columns,
                                                                      self.embedding_dict_stable,
                                                                      self.feature_index)
        x_emb_linear = self.input_from_feature_columns_linear(x)
        x_emb = x_emb.view(x_emb.size(0), -1)
        x_emb_linear = x_emb_linear.view(x_emb.size(0), -1)

        preds = self.forward(x_emb_linear, x_emb, self.classifier_params[int(d if d < self.T else -1)]).view(-1)
        return preds

    def contrastive_loss(self, logit_p, logit_r, temperature=0.5):

        logit_p = logit_p.clamp(min=1e-7, max=1 - 1e-7)
        logit_r = logit_r.clamp(min=1e-7, max=1 - 1e-7)
        loss = F.binary_cross_entropy(logit_p, logit_r.detach())
        return loss
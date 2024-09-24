from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from modeling_t5 import VLT5
class VLT5VQADG(VLT5):
    def __init__(self, config, num_answers=None, label2ans=None):
        super().__init__(config)
        self.answer_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model * 2),
            nn.GELU(),
            nn.LayerNorm(config.d_model * 2),
            nn.Linear(config.d_model * 2, 1),
            nn.Sigmoid()
        )
        #这里把nn.sigmoid去掉再计算一次
        #这里是应该用mse还是bce啊（相似度计算是一个分类还是回归啊）
        self.bce_loss = nn.BCEWithLogitsLoss()
        #self.mse_loss = nn.MSELoss()

    def train_step(self, batch):

        device = next(self.parameters()).device
        vis_feats = batch['vis_feats'].to(device)
        #question
        input_ids = batch['input_ids'].to(device)
        #answer
        input_ids2 = batch['input_ids2'].to(device)
        #distractors
        input_ids3 = batch['input_ids3'].to(device)
        input_ids4 = batch['input_ids4'].to(device)
        input_ids5 = batch['input_ids5'].to(device)
        vis_pos = batch['boxes'].to(device)
        target = batch['targets'].to(device)
        #如果使用分类模型
        B = len(input_ids)
        #如果模型loss降不下来，是不是negative不够多，尝试多采样几次，增加负样本？或者就只能修改模型一个正例很多个负例
        #Q+A
        total_input_ids = torch.cat((input_ids, input_ids2), 1)
        #Q+D
        total_input_ids3 = torch.cat((input_ids, input_ids3), 1)
        total_input_ids4 = torch.cat((input_ids, input_ids4), 1)
        total_input_ids5 = torch.cat((input_ids, input_ids5), 1)



        decoder_input_ids = torch.ones(
            B, 1, dtype=torch.long, device=device) * self.config.decoder_start_token_id

        output = self(
            input_ids=total_input_ids,
            vis_inputs=(vis_feats, vis_pos),
            decoder_input_ids=decoder_input_ids,
            output_hidden_states=True,
            return_dict=True
        )
        last_layer_hidden_state = output.decoder_hidden_states[-1]
        last_hidden_state = last_layer_hidden_state.view(B, -1, self.config.d_model)[:, -1]
        # [B, num_answers]
        logit = self.answer_head(last_hidden_state)

        negative_output3 = self(
            input_ids=total_input_ids3,
            vis_inputs=(vis_feats, vis_pos),
            decoder_input_ids=decoder_input_ids,
            output_hidden_states=True,
            return_dict=True
        )
        neg_last_layer_hidden_state3 = negative_output3.decoder_hidden_states[-1]
        neg_last_hidden_state3 = neg_last_layer_hidden_state3.view(B, -1, self.config.d_model)[:, -1]
        neg_logit3 = self.answer_head(neg_last_hidden_state3)

        negative_output4 = self(
            input_ids=total_input_ids4,
            vis_inputs=(vis_feats, vis_pos),
            decoder_input_ids=decoder_input_ids,
            output_hidden_states=True,
            return_dict=True
        )
        neg_last_layer_hidden_state4 = negative_output4.decoder_hidden_states[-1]
        neg_last_hidden_state4 = neg_last_layer_hidden_state4.view(B, -1, self.config.d_model)[:, -1]
        neg_logit4 = self.answer_head(neg_last_hidden_state4)

        negative_output5 = self(
            input_ids=total_input_ids5,
            vis_inputs=(vis_feats, vis_pos),
            decoder_input_ids=decoder_input_ids,
            output_hidden_states=True,
            return_dict=True
        )
        neg_last_layer_hidden_state5 = negative_output5.decoder_hidden_states[-1]
        neg_last_hidden_state5 = neg_last_layer_hidden_state5.view(B, -1, self.config.d_model)[:, -1]
        neg_logit5 = self.answer_head(neg_last_hidden_state5)

        total_logit = torch.cat((logit, neg_logit3, neg_logit4, neg_logit5), dim=0)
        neg_targets = target.new_zeros((B, 1)).to(device)
        total_targets = torch.cat((target, neg_targets, neg_targets, neg_targets), dim=0)
        # print('total logit', total_logit)
        # print('total targets', total_targets)

        loss = self.bce_loss(total_logit, total_targets)
        #loss = self.mse_loss(total_logit, total_targets)
        result = {
            'loss': loss
        }

        return result

    @torch.no_grad()
    def test_step(self, batch, **kwargs):
        self.eval()
        device = next(self.parameters()).device
        vis_feats = batch['vis_feats'].to(device)
        #text ids
        input_ids = batch['input_ids'].to(device)
        #print('input ids', input_ids.size())
        vis_pos = batch['boxes'].to(device)

        result = {}
        B = len(input_ids)

        decoder_input_ids = torch.ones(
            B, 1, dtype=torch.long, device=device) * self.config.decoder_start_token_id

        output = self(
            input_ids=input_ids,
            vis_inputs=(vis_feats, vis_pos),
            decoder_input_ids=decoder_input_ids,
            output_hidden_states=True,
            return_dict=True
        )

        last_layer_hidden_state = output.decoder_hidden_states[-1]
        last_hidden_state = last_layer_hidden_state.view(B, -1, self.config.d_model)[:, -1]

        # [B, num_answers]
        logit = self.answer_head(last_hidden_state)
        logit_numpy = logit.cpu().numpy()[0][0]
        result['logit'] = logit
        result['pred_ans'] = logit_numpy
        return result

from modeling_bart import VLBart
class VLBartVQADG(VLBart):
    def __init__(self, config, num_answers=None, label2ans=None):
        super().__init__(config)

        if config.classifier:
            self.answer_head = nn.Sequential(
                nn.Linear(config.d_model, config.d_model * 2),
                nn.GELU(),
                nn.LayerNorm(config.d_model * 2),
                nn.Linear(config.d_model * 2, num_answers)
            )

        self.num_answers = num_answers
        self.label2ans = label2ans
        self.bce_loss = nn.BCEWithLogitsLoss()

    def train_step(self, batch):

        device = next(self.parameters()).device
        vis_feats = batch['vis_feats'].to(device)
        input_ids = batch['input_ids'].to(device)
        vis_pos = batch['boxes'].to(device)

        if self.config.classifier:
            B = len(input_ids)

            decoder_input_ids = torch.tensor(
                [self.config.decoder_start_token_id, self.config.bos_token_id],
                dtype=torch.long, device=device).unsqueeze(0).expand(B, 2)

            output = self(
                input_ids=input_ids,
                vis_inputs=(vis_feats, vis_pos),
                decoder_input_ids=decoder_input_ids,
                output_hidden_states=True,
                return_dict=True
            )

            target = batch['targets'].to(device)

            last_layer_hidden_state = output.decoder_hidden_states[-1]
            last_hidden_state = last_layer_hidden_state.view(B, -1, self.config.d_model)[:, -1]

            # [B, num_answers]
            logit = self.answer_head(last_hidden_state)

            loss = self.bce_loss(logit, target)

        else:
            lm_labels = batch["target_ids"].to(device)

            output = self(
                input_ids=input_ids,
                vis_inputs=(vis_feats, vis_pos),
                labels=lm_labels,
                return_dict=True
            )
            assert 'loss' in output

            lm_mask = (lm_labels != -100).float()
            B, L = lm_labels.size()

            loss = output['loss']

            loss = loss.view(B, L) * lm_mask

            loss = loss.sum(dim=1) / lm_mask.sum(dim=1).clamp(min=1)  # B

            loss = loss * batch['scores'].to(device=device)

            loss = loss.mean()

        result = {
            'loss': loss
        }

        return result

    @torch.no_grad()
    def test_step(self, batch, **kwargs):
        self.eval()
        device = next(self.parameters()).device
        vis_feats = batch['vis_feats'].to(device)
        input_ids = batch['input_ids'].to(device)
        vis_pos = batch['boxes'].to(device)

        result = {}
        if self.config.classifier:
            B = len(input_ids)

            decoder_input_ids = torch.tensor(
                [self.config.decoder_start_token_id, self.config.bos_token_id],
                dtype=torch.long, device=device).unsqueeze(0).expand(B, 2)

            output = self(
                input_ids=input_ids,
                vis_inputs=(vis_feats, vis_pos),
                decoder_input_ids=decoder_input_ids,
                output_hidden_states=True,
                return_dict=True
            )

            last_layer_hidden_state = output.decoder_hidden_states[-1]
            last_hidden_state = last_layer_hidden_state.view(B, -1, self.config.d_model)[:, -1]

            # [B, num_answers]
            logit = self.answer_head(last_hidden_state)

            score, pred_ans_id = logit.max(1)
            pred_ans_id = pred_ans_id.cpu().numpy()
            pred_ans = [self.label2ans[ans_id] for ans_id in pred_ans_id]

            result['pred_ans'] = pred_ans

        else:

            output = self.generate(
                input_ids=input_ids,
                vis_inputs=(vis_feats, vis_pos),
                **kwargs
            )
            generated_sents = self.tokenizer.batch_decode(output, skip_special_tokens=True)
            result['token_ids'] = output
            result['pred_ans'] = generated_sents

        return result
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from modeling_t5 import VLT5
from tokenization import VLT5TokenizerFast
tokenizer = VLT5TokenizerFast.from_pretrained('t5-base')
class VLT5VQADG(VLT5):
    def __init__(self, config, num_answers=None, label2ans=None):
        super().__init__(config)
        #这里需要检验为什么有classifier，论文中说是生成式模型，是否有2种模式
        if config.classifier:
            print('vqadg_model.py classifier')
            self.answer_head = nn.Sequential(
                nn.Linear(config.d_model, config.d_model * 2),
                nn.GELU(),
                nn.LayerNorm(config.d_model * 2),
                nn.Linear(config.d_model * 2, num_answers)
            )
        else:
            print('vqadg_model.py not classifier')

        self.num_answers = num_answers
        self.label2ans = label2ans
        self.bce_loss = nn.BCEWithLogitsLoss()

    def train_step(self, batch, trainer1):

        device = next(self.parameters()).device
        vis_feats = batch['vis_feats'].to(device)
        #qa token id
        input_ids = batch['input_ids'].to(device)
        #only question token id
        input_ids2 = batch['input_ids2'].to(device)
        vis_pos = batch['boxes'].to(device)
        #如果使用分类模型
        if self.config.classifier:
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
            target = batch['targets'].to(device)

            last_layer_hidden_state = output.decoder_hidden_states[-1]
            last_hidden_state = last_layer_hidden_state.view(B, -1, self.config.d_model)[:, -1]

            # [B, num_answers]
            logit = self.answer_head(last_hidden_state)

            loss = self.bce_loss(logit, target)
        #如果使用语言模型
        else:
            lm_labels = batch["target_ids"].to(device)
            output = self(
                input_ids=input_ids,
                vis_inputs=(vis_feats, vis_pos),
                labels=lm_labels,
                return_dict=True
            )
            assert 'loss' in output
            #get distractors token id
            result = self.test_step(batch)
            #distractors1 = result['pred_ans']
            #print('distractors1', distractors1)
            try:
                d1_input = torch.ones(16, 8, dtype=torch.long).to(device)
                d2_input = torch.ones(16, 8, dtype=torch.long).to(device)
                d3_input = torch.ones(16, 8, dtype=torch.long).to(device)
                #必须生成的16个全满足格式才会跳出来继续计算！
                for i in range(16):
                    distractors = result['pred_ans'][i][3:][1:-1]
                    distractors_list = distractors.split(',')
                    d1 = distractors_list[0][1:-1]
                    d2 = distractors_list[1][2:-1]
                    d3 = distractors_list[2][2:-1]
                    # print('distractors', distractors_list)
                    # print('distractors 1', d1)
                    # print('distractors 2', d2)
                    # print('distractors 3', d3)
                    # print('*****************')
                    d1_token_tensor = tokenizer(d1, return_tensors='pt', padding=True).input_ids
                    d2_token_tensor = tokenizer(d2, return_tensors='pt', padding=True).input_ids
                    d3_token_tensor = tokenizer(d3, return_tensors='pt', padding=True).input_ids
                    # print('d1 token', d1_token_tensor)
                    # print('d2 token', d2_token_tensor)
                    # print('d3 token', d3_token_tensor)
                    # print('*****************')
                    d1_length = len(d1_token_tensor[0])
                    d2_length = len(d2_token_tensor[0])
                    d3_length = len(d3_token_tensor[0])
                    # print('d1 length', d1_length)
                    # print('d2 length', d2_length)
                    # print('d3 length', d3_length)
                    # print('*****************')
                    d1_input[i, :d1_length] = d1_token_tensor[0]
                    d2_input[i, :d2_length] = d2_token_tensor[0]
                    d3_input[i, :d3_length] = d3_token_tensor[0]
                    # print('d1 input', d1_input)
                    # print('d2 input', d2_input)
                    # print('d3 input', d3_input)
                    # print('*****************')

                #get question token id
                # print('question shape', input_ids2.size())
                # print('d1 shape', d1_input.size())
                # print('d2 shape', d2_input.size())
                # print('d3 shape', d3_input.size())
                total_token_tensor1 = torch.cat((input_ids2, d1_input), 1)
                total_token_tensor2 = torch.cat((input_ids2, d2_input), 1)
                total_token_tensor3 = torch.cat((input_ids2, d3_input), 1)
                # print('total token tensor1 shape', total_token_tensor1.size())
                # print('total token tensor2 shape', total_token_tensor2.size())
                # print('total token tensor3 shape', total_token_tensor3.size())
                new_batch1 = {}
                new_batch1['input_ids'] = total_token_tensor1.to(device)
                new_batch1['vis_feats'] = vis_feats.to(device)
                new_batch1['boxes'] = vis_pos.to(device)
                score1_predicted = trainer1.model.module.test_step(new_batch1)['logit'].to(device=device)
                new_batch2 = {}
                new_batch2['input_ids'] = total_token_tensor2.to(device)
                new_batch2['vis_feats'] = vis_feats.to(device)
                new_batch2['boxes'] = vis_pos.to(device)
                score2_predicted = trainer1.model.module.test_step(new_batch2)['logit'].to(device=device)
                new_batch3 = {}
                new_batch3['input_ids'] = total_token_tensor3.to(device)
                new_batch3['vis_feats'] = vis_feats.to(device)
                new_batch3['boxes'] = vis_pos.to(device)
                score3_predicted = trainer1.model.module.test_step(new_batch3)['logit'].to(device=device)
                # print('score1 predicted', score1_predicted)
                # print('score2 predicted', score2_predicted)
                # print('score3 predicted', score3_predicted)
                loss1_ori = score1_predicted.mean()
                loss2_ori = score2_predicted.mean()
                loss3_ori = score3_predicted.mean()
                #这里阈值如果改为0.4，0.5，0.6是不是就能生成不一样的d1,d2,d3了
                loss1 = abs(loss1_ori - 0.5)
                loss2 = abs(loss2_ori - 0.5)
                loss3 = abs(loss3_ori - 0.5)
                # print('score loss1', loss1)
                # print('score loss2', loss2)
                # print('score loss3', loss3)
                loss_d = loss1+loss2+loss3
                #print('score lossd', loss_d)
            except:
                loss1 = 0
                loss2 = 0
                loss3 = 0
                loss_d = loss1+loss2+loss3

            lm_mask = (lm_labels != -100).float()
            B, L = lm_labels.size()
            #print('B', B)
            #print('L', L)
            loss = output['loss']

            loss = loss.view(B, L) * lm_mask

            loss = loss.sum(dim=1) / lm_mask.sum(dim=1).clamp(min=1)  # B

            loss = loss.to(device=device)
            #loss = loss * batch['scores'].to(device=device)

            loss = loss.mean()
            #这里loss_d可以选择除以3或者不，试一下。同时，这种方法会让D完全和QA无关，起不到迷惑的作用了！
            total_loss = loss + loss_d
            print('lossd', loss_d)

        result = {
            'loss': total_loss
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
        if self.config.classifier:
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

            score, pred_ans_id = logit.max(1)
            pred_ans_id = pred_ans_id.cpu().numpy()
            pred_ans = [self.label2ans[ans_id] for ans_id in pred_ans_id]

            result['pred_ans'] = pred_ans

        else:
            output = self.generate(
                input_ids=input_ids,
                vis_inputs=(vis_feats, vis_pos),
                max_length=60,
                **kwargs
            )
            #print('gen_max_length', self.config.gen_max_length)
            #print('max_text_length', self.config.max_text_length)
            generated_sents = self.tokenizer.batch_decode(output, skip_special_tokens=True)
            result['token_ids'] = output
            result['pred_ans'] = generated_sents
            #print(result)

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
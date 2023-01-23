from builtins import ValueError
import json
import math
import os
from time import time
import torch
from torch.nn import functional as F
import torch.distributed as dist
from utils.logger import LOGGER
from utils.distributed import ddp_allgather, all_gather_list

from utils.misc import NoOp
from cococaption.pycocoevalcap.eval import COCOEvalCap
from cococaption.pycocotools.coco import COCO
from tqdm import tqdm 


def validate(model, val_dataloaders, opts, global_step, total_step):

    eval_log = {}
    model.eval()
    for task, loader in val_dataloaders.items():
        LOGGER.info(f"validate on {task} task")
        val_log = validate_single(model, loader, task.split('--')[0], opts, global_step, total_step,task.split('--')[1])
        eval_log[task] = val_log
    model.train()
    return eval_log


@torch.no_grad()
def validate_single(model, val_loader, task, opts, global_step, total_step,dset_name):
    LOGGER.info("start running {} validation...".format(task))
    if task.startswith('pt'):
        return validate_pt(model, val_loader, task, opts, global_step)
    elif task.startswith('ret'):
        return validate_ret(model, val_loader, task, opts, global_step)
    elif task.startswith('cap'):
        return validate_cap(model, val_loader, task, opts, global_step, dset_name)
    elif task.startswith('qa'):
        return validate_qa(model, val_loader, task, opts, global_step,dset_name)
    

@torch.no_grad()
def validate_qa(model, eval_loader, task_str, opts, global_step,dset_name):
    st = time()
    val_log = {}

    task = task_str.split('%')[1:]
    output_dir = os.path.join(opts.output_dir,f'predict_answers')
    os.makedirs(output_dir,exist_ok=True)
    

    groundtruth_answers=[]
    generated_answers_t_v = []
    generated_answers_t_va = []
    generated_answers_t_a = []

    
    if dist.get_rank() == 0:
        pbar = tqdm(total=len(eval_loader))
    else:
        pbar = NoOp()


    answer_tokens = None
    
    submit_file_tv = []
        
    for batch in eval_loader:
        ids = batch['ids']
        groundtruth_answers += batch['txt_tokens']
        # if global_step !=0:
        
        evaluation_dict = model(batch, task_str, compute_loss=False)
        if 'tv' in task:
            answers = evaluation_dict['generated_answers_t_v']
            answers = get_model_attr(model, 'decode_sequence')(answers.data)
            generated_answers_t_v += answers
            if batch['question_ids'] is not None:
                for i in range(len(batch['question_ids'])):
                    submit_file_tv.append({'question_id':batch['question_ids'][i],'answer':answers[i]})

        if 'tva' in task:
            answers = evaluation_dict['generated_answers_t_va']
            answers = get_model_attr(model, 'decode_sequence')(answers.data)
            generated_answers_t_va += answers

        if 'ta' in task:
            answers = evaluation_dict['generated_answers_t_a']
            answers = get_model_attr(model, 'decode_sequence')(answers.data)
            generated_answers_t_a += answers
  
       


        pbar.update(1)
        

    pbar.close()

    groundtruth_answers = [i for j in all_gather_list(groundtruth_answers)  for i in j]
    if dist.get_rank()==0:
        json.dump(groundtruth_answers,open(os.path.join(output_dir,f'step{global_step}_gt.json'),'w'))
    total_num = len(groundtruth_answers)
    #assert len(all_groundtruth_answers) == total_num
    LOGGER.info('total {} questions has been tested'.format(total_num))

    if 'tv' in task:
        generated_answers_t_v = [i for j in all_gather_list(generated_answers_t_v)  for i in j]
        if dist.get_rank()==0:
            json.dump(generated_answers_t_v,open(os.path.join(output_dir,f'step{global_step}_tv_pred.json'),'w'))
        submit_files_t_v = [i for j in all_gather_list(submit_file_tv)  for i in j]
        if dist.get_rank()==0:
            json.dump(submit_files_t_v,open(os.path.join(output_dir,f'step{global_step}_tv_pred_submited_{dset_name}.json'),'w'))
        accurate_num = sum([generated_answers_t_v[i] == groundtruth_answers[i] for i in range(total_num)])
        accuracy = accurate_num / total_num
        val_log['tv'] = {'accuracy':round(accuracy*100,2)} 
    if 'tva' in task:
        generated_answers_t_va = [i for j in all_gather_list(generated_answers_t_va)  for i in j]
        accurate_num = sum([generated_answers_t_va[i] == groundtruth_answers[i] for i in range(total_num)])
        accuracy = accurate_num / total_num
        val_log['tva'] = {'accuracy':round(accuracy*100,2)} 
 
    if 'ta' in task:
        generated_answers_t_a = [i for j in all_gather_list(generated_answers_t_a)  for i in j]
        accurate_num = sum([generated_answers_t_a[i] == groundtruth_answers[i] for i in range(total_num)])
        accuracy = accurate_num / total_num
        val_log['ta'] = {'accuracy':round(accuracy*100,2)} 
 
    return val_log




@torch.no_grad()
def validate_cap(model, eval_loader, task_str, opts, global_step,dset_name):
    st = time()
    coco_submit = opts.coco_submit
    vatex_submit = opts.vatex_submit
    nocaps_submit = opts.nocaps_submit
    val_log = {}
    task = task_str.split('%')[1:]
    annfile_path = eval_loader.dataset.annfile
    

    generated_sequences_t_v = []
    generated_sequences_t_va = []
    generated_sequences_t_a = []


    if dist.get_rank() == 0:
        pbar = tqdm(total=len(eval_loader))
    else:
        pbar = NoOp()

    
    for batch in eval_loader:
        ids = batch['ids']
        evaluation_dict = model(batch, task_str, compute_loss=False)
        if 'tv' in task:
            sents = evaluation_dict['generated_sequences_t_v']
            
            sents = get_model_attr(model, 'decode_sequence')(sents.data)
            for  i in range(len(sents)):
                if coco_submit:
                    generated_sequences_t_v.append({'image_id':int(ids[i].split('_')[-1]), 'caption': sents[i]})
                elif nocaps_submit:
                    generated_sequences_t_v.append({'image_id':int(ids[i]), 'caption': sents[i]})
                else:
                    generated_sequences_t_v.append({'video_id':ids[i], 'caption': sents[i]})



        if 'tva' in task:
            sents = evaluation_dict['generated_sequences_t_va']
            sents = get_model_attr(model, 'decode_sequence')(sents.data)
            for  i in range(len(sents)):
                if vatex_submit:
                    generated_sequences_t_va.append({ids[i]:  sents[i]})
                else:

                    generated_sequences_t_va.append({'video_id':ids[i], 'caption': sents[i]})



        if 'ta' in task:
            sents = evaluation_dict['generated_sequences_t_a']
            sents = get_model_attr(model, 'decode_sequence')(sents.data)
            
            for  i in range(len(sents)):
                generated_sequences_t_a.append({'video_id':ids[i], 'caption': sents[i]})


        pbar.update(1)
        

    pbar.close()


    result_folder = os.path.join(opts.output_dir, f'results_test_{dset_name}')
    os.makedirs(result_folder, exist_ok=True)
    if 'tva' in task:
        results = [i for j in all_gather_list(generated_sequences_t_va)  for i in j]
        
        if  vatex_submit:
            result = {}
            for i in results:
                result[list(i.keys())[0]]=i[list(i.keys())[0]]
            json.dump(result,open(os.path.join(result_folder, 'submission.json'), 'w'))
        else:

        
            val_log['tva'] = compute_metric_cap(results, annfile_path) 
            json.dump(results,open(os.path.join(result_folder, 'step_{}_tva.json'.format(global_step)), 'w'))
            
    if 'tv' in task:
        results = [i for j in all_gather_list(generated_sequences_t_v)  for i in j]
       
    
        if  coco_submit or nocaps_submit:
            json.dump(results,open(os.path.join(result_folder, 'submission.json'), 'w'))
        else:
            val_log['tv'] = compute_metric_cap(results, annfile_path) 
            json.dump(results,open(os.path.join(result_folder, 'step_{}_tv.json'.format(global_step)), 'w'))
        
            


    if 'ta' in task:
        results = [i for j in all_gather_list(generated_sequences_t_a)  for i in j]
        if dist.get_rank() == 0:
            val_log['ta'] = compute_metric_cap(results, annfile_path) 
            json.dump(results,open(os.path.join(result_folder, 'step_{}_ta.json'.format(global_step)), 'w'))

   
        
    return val_log










@torch.no_grad()
def validate_ret(model, val_loader, task_str, opts, global_step):
    val_log = {}
    feat_t = []
    feat_v = []
    feat_a=[]
    feat_va = []

    ids = []
    ids_txt = []
    txt_tokens = []
    task = task_str.split('%')[1:]
    

    for i, batch in enumerate(val_loader):
        evaluation_dict= model(batch, task_str, compute_loss=False)
        feat_t.append(evaluation_dict['feat_t'])
        feat_v.append(evaluation_dict['feat_v'])
        feat_a.append(evaluation_dict['feat_a'])
        txt_tokens.append(evaluation_dict['txt_tokens'])
        ids += batch['ids']
        if batch['ids_txt'] is  None:
            ids_txt  += batch['ids']
        else:
            ids_txt  += batch['ids_txt']

    
    ids = [j for i in all_gather_list(ids) for j in i]
    ids_txt = [j for i in all_gather_list(ids_txt) for j in i]
    

    if feat_t[0] is not None:
        feat_t = torch.cat(feat_t, dim = 0)
        feat_t = ddp_allgather(feat_t).half()
    if feat_v[0] is not None:
        feat_v = torch.cat(feat_v, dim = 0)
        feat_v = ddp_allgather(feat_v).half()
    if feat_a[0] is not None:  
        feat_a = torch.cat(feat_a, dim = 0)
        feat_a = ddp_allgather(feat_a).half()
    if txt_tokens[0] is not None:  
        txt_tokens = torch.cat(txt_tokens, dim = 0)
        txt_tokens = ddp_allgather(txt_tokens)
    
    
    if dist.get_rank() == 0:
    
        if opts.contra_type == 'fine':
            if 'tv' in task:
                maskA = (txt_tokens !=0).long().cuda()
                maskB = torch.ones(*feat_v.shape[:2]).long().cuda()

                weightA = get_model_attr(model,'fine_weight_mapper')['text'](feat_t).squeeze(2)
                weightB = get_model_attr(model,'fine_weight_mapper')['video'](feat_v).squeeze(2)
                score_matrix_t_v = get_model_attr(model,'compute_fine_matrix')(feat_t, feat_v, maskA, maskB, weightA, weightB)
                log = compute_metric_ret(score_matrix_t_v, ids, ids_txt, model)
                log = {k.replace('forward','video').replace('backward','txt') : v for k,v in log.items()}
                val_log['t_v'] = log

            if 'tva' in task:
                if opts.late_fusion:
                    maskt = (txt_tokens !=0).long().cuda()
                    maskv = torch.ones(*feat_v.shape[:2]).long().cuda()
                    maska = torch.ones(*feat_a.shape[:2]).long().cuda()
                    weightt = torch.ones_like(feat_t[:,:,0])
                    weightv = torch.ones_like(feat_v[:,:,0])
                    weighta = torch.ones_like(feat_a[:,:,0])
                    score_matrix_t_va = get_model_attr(model,'compute_fine_matrix')(feat_t, feat_v, maskt, maskv, weightt, weightv) + \
                                        get_model_attr(model,'compute_fine_matrix')(feat_t, feat_a, maskt, maska, weightt, weighta)

                else:
                   
                    feat_va = torch.cat((feat_v,feat_a),dim=1)
                    maskA = (txt_tokens !=0).long().cuda()
                    maskB = torch.ones(*feat_va.shape[:2]).long().cuda()
                    if opts.fineweight_type == 'none':
                        weightA = torch.ones_like(feat_t[:,:,0])
                        weightB = torch.cat((torch.ones_like(feat_v[:,:,0]),torch.ones_like(feat_a[:,:,0])),dim=1)
                    else:
                        weightA = get_model_attr(model,'fine_weight_mapper')['text'](feat_t).squeeze(2)
                        weightB = torch.cat((get_model_attr(model,'fine_weight_mapper')['video'](feat_v).squeeze(2), get_model_attr(model,'fine_weight_mapper')['audio'](feat_a).squeeze(2)),dim=1)
                    
                            
                    score_matrix_t_va = get_model_attr(model,'compute_fine_matrix')(feat_t, feat_va, maskA, maskB, weightA, weightB)
                log = compute_metric_ret(score_matrix_t_va, ids, ids_txt, model)
                log = {k.replace('forward','video').replace('backward','txt') : v for k,v in log.items()}
                val_log['t_va'] = log

            

            if 'ta' in task:
                maskA = (txt_tokens !=0).long().cuda()
                maskB = torch.ones(*feat_a.shape[:2]).long().cuda()
                weightA = get_model_attr(model,'fine_weight_mapper')['text'](feat_t).squeeze(2)
                weightB = get_model_attr(model,'fine_weight_mapper')['audio'](feat_a).squeeze(2)
                score_matrix_t_a = get_model_attr(model,'compute_fine_matrix')(feat_t, feat_a, maskA, maskB, weightA, weightB)
                log = compute_metric_ret(score_matrix_t_a, ids, ids_txt, model)
                log = {k.replace('forward','audio').replace('backward','txt') : v for k,v in log.items()}
                val_log['t_a'] = log



            if 'va' in task:
                maskA = torch.ones_like(feat_v[:,:,0])
                maskB = torch.ones_like(feat_a[:,:,0])
                weightA = get_model_attr(model,'fine_weight_mapper')['video'](feat_v).squeeze(2)
                weightB = get_model_attr(model,'fine_weight_mapper')['audio'](feat_a).squeeze(2)
                score_matrix_va = get_model_attr(model,'compute_fine_matrix')(feat_v, feat_a, maskA, maskB, weightA, weightB)
                log = compute_metric_ret(score_matrix_va, ids, ids_txt, model)
                log = {k.replace('forward','audio').replace('backward','video') : v for k,v in log.items()}
                val_log['v_a'] = log
                

            if 'vta' in task:
                maskA = torch.ones_like(feat_v[:,:,0])
                maskB = torch.cat(((txt_tokens !=0).long().cuda(), torch.ones_like(feat_a[:,:,0])),dim=1)
                feat_ta = torch.cat((feat_t,feat_a),dim=1)
                weightA = get_model_attr(model,'fine_weight_mapper')['video'](feat_v).squeeze(2)
                weightB = torch.cat((get_model_attr(model,'fine_weight_mapper')['text'](feat_t).squeeze(2),get_model_attr(model,'fine_weight_mapper')['audio'](feat_a).squeeze(2)),dim=1)
                score_matrix_vta = get_model_attr(model,'compute_fine_matrix')(feat_v, feat_ta, maskA, maskB, weightA, weightB)
                log = compute_metric_ret(score_matrix_vta, ids, ids_txt, model)
                log = {k.replace('forward','ta').replace('backward','video') : v for k,v in log.items()}
                val_log['v_ta'] = log

            if 'atv' in task:
                maskA = torch.ones_like(feat_a[:,:,0])
                maskB = torch.cat(((txt_tokens !=0).long().cuda(), torch.ones_like(feat_v[:,:,0])),dim=1)
                feat_tv = torch.cat((feat_t,feat_v),dim=1)
                weightA = get_model_attr(model,'fine_weight_mapper')['audio'](feat_a).squeeze(2)
                weightB = torch.cat((get_model_attr(model,'fine_weight_mapper')['text'](feat_t).squeeze(2),get_model_attr(model,'fine_weight_mapper')['video'](feat_v).squeeze(2)),dim=1)
                score_matrix_atv = get_model_attr(model,'compute_fine_matrix')(feat_a, feat_tv, maskA, maskB, weightA, weightB)
                log = compute_metric_ret(score_matrix_atv, ids, ids_txt, model)
                log = {k.replace('forward','tv').replace('backward','audio') : v for k,v in log.items()}
                val_log['a_tv'] = log

        elif opts.contra_type == 'coarse':
            if 'tv' in task:
                score_matrix_t_v = torch.matmul(feat_t,feat_v.permute(1,0))
                log = compute_metric_ret(score_matrix_t_v, ids, ids_txt, model)
                log = {k.replace('forward','video').replace('backward','txt') : v for k,v in log.items()}
                val_log['t_v'] = log

            if 'tva' in task:
                if opts.late_fusion:
                    score_matrix_t_va = torch.matmul(feat_t,feat_v.permute(1,0)) + torch.matmul(feat_t,feat_a.permute(1,0))
                else:
                    feat_va = F.normalize(get_model_attr(model,'va_fusion')(torch.cat((feat_v,feat_a),dim=-1)),dim=-1)
                    score_matrix_t_va = torch.matmul(feat_t,feat_va.permute(1,0))
                log = compute_metric_ret(score_matrix_t_va, ids, ids_txt, model)
                log = {k.replace('forward','video').replace('backward','txt') : v for k,v in log.items()}
                val_log['t_va'] = log

            
            if 'ta' in task:

                score_matrix_t_a = torch.matmul(feat_t,feat_a.permute(1,0))
                log = compute_metric_ret(score_matrix_t_a, ids, ids_txt, model)
                log = {k.replace('forward','audio').replace('backward','txt') : v for k,v in log.items()}
                val_log['t_a'] = log
        



    return val_log


@torch.no_grad()
def validate_pt(model, val_loader, task, opts, global_step):
    n_word_mlm = 0
    n_correct_mlm_tva = 0
    n_correct_mlm_tv = 0
    n_correct_mlm_ta = 0


    n_word_caption = 0
    n_correct_caption_tva = 0
    n_correct_caption_tv = 0
    n_correct_caption_ta = 0


    feat_t = []
    feat_v = []
    feat_a =[]
    feat_va = []
    ids = []
    ids_txt = []
    txt_tokens = []
    val_log = {}


    for i, batch in enumerate(val_loader):
        evaluation_dict= model(batch, task=task, compute_loss=False)
        task_ls = task.split('_')
        mlm_task = [] 
        caption_task = [] 
        contra_task = [] 
        for i in task_ls:
            if 'mlm' in i:
                mlm_task = i.split('%')[1:]
            elif 'caption' in i:
                caption_task = i.split('%')[1:]
            elif 'contra' in i:
                contra_task = i.split('%')[1:]

        if contra_task !=[]:
            
            feat_t.append(evaluation_dict['feat_t'])
            feat_v.append(evaluation_dict['feat_v'])
            feat_a.append(evaluation_dict['feat_a'])
            ids += batch['ids']
            ids_txt += batch['ids_txt']
            txt_tokens.append(evaluation_dict['txt_tokens'])


        if caption_task !=[]:
            txt_labels_caption = evaluation_dict['txt_labels_caption'] 
            txt_labels_caption = txt_labels_caption[txt_labels_caption != -1]
            n_word_caption += txt_labels_caption.numel()
            if 'tva' in caption_task:
                caption_scores_tva = evaluation_dict['caption_scores_tva'] 
                n_correct_caption_tva += (caption_scores_tva.max(dim=-1)[1] == txt_labels_caption).sum().item()
          
            if 'tv' in caption_task:
                caption_scores_tv = evaluation_dict['caption_scores_tv'] 
                n_correct_caption_tv += (caption_scores_tv.max(dim=-1)[1] == txt_labels_caption).sum().item()
            if 'ta' in caption_task:
                caption_scores_ta = evaluation_dict['caption_scores_ta'] 
                n_correct_caption_ta += (caption_scores_ta.max(dim=-1)[1] == txt_labels_caption).sum().item()
            
            

        if mlm_task !=[]:
            txt_labels_mlm = evaluation_dict['txt_labels_mlm'] 
            txt_labels_mlm = txt_labels_mlm[txt_labels_mlm != -1]
            n_word_mlm += txt_labels_mlm.numel() 
            if 'tva' in caption_task:
                mlm_scores_tva = evaluation_dict['mlm_scores_tva'] 
                n_correct_mlm_tva += (mlm_scores_tva.max(dim=-1)[1] == txt_labels_mlm).sum().item()
          
            if 'tv' in caption_task:
                mlm_scores_tv = evaluation_dict['mlm_scores_tv'] 
                n_correct_mlm_tv += (mlm_scores_tv.max(dim=-1)[1] == txt_labels_mlm).sum().item()
            if 'ta' in caption_task:
                mlm_scores_ta = evaluation_dict['mlm_scores_ta'] 
                n_correct_mlm_ta += (mlm_scores_ta.max(dim=-1)[1] == txt_labels_mlm).sum().item()

        
    if caption_task !=[]:
        n_word_caption = sum(all_gather_list(n_word_caption))
        if 'tva' in caption_task:
            n_correct_caption_tva = sum(all_gather_list(n_correct_caption_tva))
            caption_acc_tva = n_correct_caption_tva / n_word_caption
            val_log['caption_acc_tva'] = round(caption_acc_tva,2)
        
        if 'tv' in caption_task:    
            n_correct_caption_tv = sum(all_gather_list(n_correct_caption_tv))
            caption_acc_tv = n_correct_caption_tv / n_word_caption   
            val_log['caption_acc_tv'] = round(caption_acc_tv,2)
        if 'ta' in caption_task:
            n_correct_caption_ta = sum(all_gather_list(n_correct_caption_ta))
            caption_acc_ta = n_correct_caption_ta / n_word_caption  
            val_log['caption_acc_ta'] = round(caption_acc_ta,2)       

        
        
        

        

    if mlm_task !=[]:
        n_word_mlm = sum(all_gather_list(n_word_mlm))
        if 'tva' in mlm_task:
            n_correct_mlm_tva = sum(all_gather_list(n_correct_mlm_tva))
            mlm_acc_tva = n_correct_mlm_tva / n_word_mlm     
            val_log['mlm_acc_tva'] = round(mlm_acc_tva,2)
     
        if 'tv' in mlm_task:
            n_correct_mlm_tv = sum(all_gather_list(n_correct_mlm_tv))
            mlm_acc_tv = n_correct_mlm_tv / n_word_mlm    
            val_log['mlm_acc_tv'] = round(mlm_acc_tv,2)
        if 'ta' in mlm_task:
            n_correct_mlm_ta = sum(all_gather_list(n_correct_mlm_ta))
            mlm_acc_ta = n_correct_mlm_ta / n_word_mlm     
            val_log['mlm_acc_ta'] = round(mlm_acc_ta,2)  

        
    if contra_task !=[]:
        
            
            
        
        ids = [j for i in all_gather_list(ids) for j in i]
        ids_txt = [j for i in all_gather_list(ids_txt) for j in i]
        

        if feat_t[0] is not None:
            feat_t = torch.cat(feat_t, dim = 0)
            feat_t = ddp_allgather(feat_t).half()
        if feat_v[0] is not None:
            feat_v = torch.cat(feat_v, dim = 0)
            feat_v = ddp_allgather(feat_v).half()
        if feat_a[0] is not None:  
            feat_a = torch.cat(feat_a, dim = 0)
            feat_a = ddp_allgather(feat_a).half()

        if txt_tokens[0] is not None:  
            txt_tokens = torch.cat(txt_tokens, dim = 0)
            txt_tokens = ddp_allgather(txt_tokens)
        
        

        if dist.get_rank() == 0:
            if opts.contra_type == 'fine':
                if 'tv' in contra_task:
                    maskA = (txt_tokens !=0).long().cuda()
                    maskB = torch.ones(*feat_v.shape[:2]).long().cuda()
                    weightA = get_model_attr(model,'fine_weight_mapper')['text'](feat_t).squeeze(2)
                    weightB = get_model_attr(model,'fine_weight_mapper')['video'](feat_v).squeeze(2)
                    score_matrix_t_v = get_model_attr(model,'compute_fine_matrix')(feat_t, feat_v, maskA, maskB, weightA, weightB)
                    t2v_recall = compute_metric_ret(score_matrix_t_v, ids, ids_txt, model)['forward_recall']
                    val_log['t2v_recall'] = t2v_recall

                if 'tva' in contra_task:
                    if opts.late_fusion:
                        maskt = (txt_tokens !=0).long().cuda()
                        maskv = torch.ones(*feat_v.shape[:2]).long().cuda()
                        maska = torch.ones(*feat_a.shape[:2]).long().cuda()
                        weightt = torch.ones_like(feat_t[:,:,0])
                        weightv = torch.ones_like(feat_v[:,:,0])
                        weighta = torch.ones_like(feat_a[:,:,0])
                        score_matrix_t_va = get_model_attr(model,'compute_fine_matrix')(feat_t, feat_v, maskt, maskv, weightt, weightv) + \
                                            get_model_attr(model,'compute_fine_matrix')(feat_t, feat_a, maskt, maska, weightt, weighta)

                    else:
                        feat_va = torch.cat((feat_v,feat_a),dim=1)
                        maskA = (txt_tokens !=0).long().cuda()
                        maskB = torch.ones(*feat_va.shape[:2]).long().cuda()
                        if opts.fineweight_type == 'none':
                            weightA = torch.ones_like(feat_t[:,:,0])
                            weightB = torch.cat((torch.ones_like(feat_v[:,:,0]),torch.ones_like(feat_a[:,:,0])),dim=1)
                        else:
                            weightA = get_model_attr(model,'fine_weight_mapper')['text'](feat_t).squeeze(2)
                            weightB = torch.cat((get_model_attr(model,'fine_weight_mapper')['video'](feat_v).squeeze(2), get_model_attr(model,'fine_weight_mapper')['audio'](feat_a).squeeze(2)),dim=1)
                        score_matrix_t_va = get_model_attr(model,'compute_fine_matrix')(feat_t, feat_va, maskA, maskB, weightA, weightB)
                    t2va_recall = compute_metric_ret(score_matrix_t_va, ids, ids_txt, model)['forward_recall']
                    val_log['t2va_recall'] = t2va_recall

            

                
                if 'ta' in contra_task:
                    maskA = (txt_tokens !=0).long().cuda()
                    maskB = torch.ones(*feat_a.shape[:2]).long().cuda()
                    weightA = get_model_attr(model,'fine_weight_mapper')['text'](feat_t).squeeze(2)
                    weightB = get_model_attr(model,'fine_weight_mapper')['audio'](feat_a).squeeze(2)
                    score_matrix_t_a = get_model_attr(model,'compute_fine_matrix')(feat_t, feat_a, maskA, maskB, weightA, weightB)
                    t2a_recall = compute_metric_ret(score_matrix_t_a, ids, ids_txt, model)['forward_recall']
                    val_log['t2a_recall'] = t2a_recall


                if 'va' in task:
                    maskA = torch.ones_like(feat_v[:,:,0])
                    maskB = torch.ones_like(feat_a[:,:,0])
                    weightA = get_model_attr(model,'fine_weight_mapper')['video'](feat_v).squeeze(2)
                    weightB = get_model_attr(model,'fine_weight_mapper')['audio'](feat_a).squeeze(2)
                    score_matrix_va = get_model_attr(model,'compute_fine_matrix')(feat_v, feat_a, maskA, maskB, weightA, weightB)
                    v2a_recall = compute_metric_ret(score_matrix_va, ids, ids_txt, model)['forward_recall']
                    val_log['v2a_recall'] = v2a_recall


                if 'vta' in task:
                    maskA = torch.ones_like(feat_v[:,:,0])
                    maskB = torch.cat(((txt_tokens !=0).long().cuda(), torch.ones_like(feat_a[:,:,0])),dim=1)
                    feat_ta = torch.cat((feat_t,feat_a),dim=1)
                    weightA = get_model_attr(model,'fine_weight_mapper')['video'](feat_v).squeeze(2)
                    weightB = torch.cat((get_model_attr(model,'fine_weight_mapper')['text'](feat_t).squeeze(2),get_model_attr(model,'fine_weight_mapper')['audio'](feat_a).squeeze(2)),dim=1)
                    score_matrix_vta = get_model_attr(model,'compute_fine_matrix')(feat_v, feat_ta, maskA, maskB, weightA, weightB)
                    v2ta_recall = compute_metric_ret(score_matrix_vta, ids, ids_txt, model)['forward_recall']
                    val_log['v2ta_recall'] = v2ta_recall


                if 'atv' in task:
                    maskA = torch.ones_like(feat_a[:,:,0])
                    maskB = torch.cat(((txt_tokens !=0).long().cuda(), torch.ones_like(feat_v[:,:,0])),dim=1)
                    feat_tv = torch.cat((feat_t,feat_v),dim=1)
                    weightA = get_model_attr(model,'fine_weight_mapper')['audio'](feat_a).squeeze(2)
                    weightB = torch.cat((get_model_attr(model,'fine_weight_mapper')['text'](feat_t).squeeze(2),get_model_attr(model,'fine_weight_mapper')['video'](feat_v).squeeze(2)),dim=1)
                    score_matrix_atv = get_model_attr(model,'compute_fine_matrix')(feat_a, feat_tv, maskA, maskB, weightA, weightB)
                    a2tv_recall = compute_metric_ret(score_matrix_atv, ids, ids_txt, model)['forward_recall']
                    val_log['a2tv_recall'] = a2tv_recall


            elif opts.contra_type == 'coarse':
                if 'tv' in contra_task:
                    score_matrix_t_v = torch.matmul(feat_t,feat_v.permute(1,0))
                    t2v_recall = compute_metric_ret(score_matrix_t_v, ids, ids_txt, model)['forward_recall']
                    val_log['t2v_recall'] = t2v_recall

                if 'tva' in contra_task:
                    if opts.late_fusion:
                        score_matrix_t_va = torch.matmul(feat_t,feat_v.permute(1,0)) + torch.matmul(feat_t,feat_a.permute(1,0))
                    else:
                        feat_va = F.normalize(get_model_attr(model,'va_fusion')(torch.cat((feat_v,feat_a),dim=-1)),dim=-1)
                        score_matrix_t_va = torch.matmul(feat_t,feat_va.permute(1,0))
                    t2va_recall = compute_metric_ret(score_matrix_t_va, ids, ids_txt, model)['forward_recall']
                    val_log['t2va_recall'] = t2va_recall

                
                if 'ta' in contra_task:

                    score_matrix_t_a = torch.matmul(feat_t,feat_a.permute(1,0))
                    t2a_recall = compute_metric_ret(score_matrix_t_a, ids, ids_txt, model)['forward_recall']
                    val_log['t2a_recall'] = t2a_recall
            

    val_log = {'placeholder':val_log}

    return val_log



def get_model_attr(model, attr_name):

    
    if hasattr(model,'module') and hasattr(model.module,attr_name):
        return getattr(model.module, attr_name)

    elif hasattr(model,attr_name):
        return getattr(model, attr_name)
    
    else:
        return ValueError

    



def compute_dualsoftmax_forward(score_matrix,model,ids_txt):
    
    txt_per_video = math.ceil(score_matrix.shape[0] /  score_matrix.shape[1])    
    if  get_model_attr(model,'video_encoder_type').startswith('clip'):
        temp = 1./  get_model_attr(model,'clip_model').logit_scale.exp()
    else:
        temp =  get_model_attr(model,'contra_temp')

    if get_model_attr(model,'dual_softmax'):
        score_matrix = score_matrix * F.softmax(score_matrix / temp , dim=0)*len(score_matrix)
    LOGGER.info(f'score_matrix_shape: {score_matrix.shape}')
    return score_matrix,ids_txt


def compute_dualsoftmax_backward(score_matrix,model):
    
    
    if  get_model_attr(model,'video_encoder_type').startswith('clip'):
        temp = 1./  get_model_attr(model,'clip_model').logit_scale.exp()
           
            
    else:
        temp =  get_model_attr(model,'contra_temp')

    if get_model_attr(model,'dual_softmax'):
        score_matrix = score_matrix * F.softmax(score_matrix / temp , dim=1)*len(score_matrix[0])

    return score_matrix

def compute_metric_ret(score_matrix, ids, ids_txt, model):

    score_matrix_forward, ids_txt_forward = compute_dualsoftmax_forward(score_matrix,model,ids_txt)

    
 
    assert score_matrix_forward.shape == (len(ids_txt),len(ids))
   
    
    indice_matrix_1 = score_matrix_forward.sort(dim=-1,descending=True)[1].tolist()


    rank = []
    for i in range(len(ids_txt_forward)):
        gt_indice = ids.index(ids_txt_forward[i])
        rank.append(indice_matrix_1[i].index(gt_indice))
    
    rank = torch.tensor(rank).to(score_matrix_forward)
    
    vr_r1 = (rank < 1).sum().item() / len(ids_txt_forward)
    vr_r5 = (rank < 5).sum().item() / len(ids_txt_forward)
    vr_r10 = (rank < 10).sum().item() / len(ids_txt_forward)
    v_medianR = torch.median(rank).item() +1
    v_meanR = torch.mean(rank).item() +1

   
    if get_model_attr(model,'evaluate_ret_text'):
        score_matrix_backward  = compute_dualsoftmax_backward(score_matrix,model)
        indice_matrix_2 = score_matrix_backward.sort(dim=0,descending=True)[1].permute(1,0).tolist()
        rank = []
        for i in range(len(ids)):
            gt_indices=[]
            for idx, id in enumerate(ids_txt):
                if id == ids[i]:
                    gt_indices.append(idx)

            rank.append(min([indice_matrix_2[i].index(idx) for idx in gt_indices]))
        
        rank = torch.tensor(rank).to(score_matrix_backward)
        
        tr_r1 = (rank < 1).sum().item() / len(ids)
        tr_r5 = (rank < 5).sum().item() / len(ids)
        tr_r10 = (rank < 10).sum().item() / len(ids)
        t_medianR = torch.median(rank).item() +1
        t_meanR = torch.mean(rank).item() +1

        eval_log = {'forward_recall': f'{round(vr_r1*100,1)}/{round(vr_r5*100,1)}/{round(vr_r10*100,1)}',
                    'forward_ravg': round((vr_r1 + vr_r5 + vr_r10)/3 *100,1), 
                    'forward_medianR': v_medianR,
                    'forward_meanR': v_meanR,

                    'backward_recall': f'{round(tr_r1*100,1)}/{round(tr_r5*100,1)}/{round(tr_r10*100,1)}',
                    'backward_ravg': round((tr_r1 + tr_r5 + tr_r10)/3 *100,1), 
                    'backward_medianR': t_medianR,
                    'backward_meanR':t_meanR}
    
    else:
        eval_log = {'forward_recall': f'{round(vr_r1*100,1)}/{round(vr_r5*100,1)}/{round(vr_r10*100,1)}',
                    'forward_ravg': round((vr_r1 + vr_r5 + vr_r10)/3 *100,1), 
                    'forward_medianR': v_medianR,
                    'forward_meanR': v_meanR,}
    return eval_log





def compute_metric_cap(results, annfile_path):
    coco = COCO(annfile_path)
    cocoRes = coco.loadRes(results)
    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.evaluate()
    metric = cocoEval.eval
    metric = {k: round(v*100,2)  for k,v in metric.items()}
    return metric

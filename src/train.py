import os
import cv2
import json
import wandb
import argparse
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import torch
import torch.optim as optim
from torch.optim import Adam, Adadelta, Adamax, Adagrad, RMSprop, Rprop, SGD
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoConfig, BertTokenizer, VisualBertModel, \
        VisualBertForVisualReasoning, LxmertForPreTraining, LxmertTokenizer
from data import ImageTextClassificationDataset
from eval import evaluate
from lxmert_for_classification import LxmertForBinaryClassification

wandb.init(project="visual-spatial-reasoning", entity="hardyqr")


def train(args, train_loader, val_loader, model, scaler=None, step_global=0, epoch=-1, \
        val_best_score=0, processor=None):
    model_type = args.model_type
    train_loss = 0
    train_steps = 0

    model.cuda()
    model.train()
    for i, data in tqdm(enumerate(train_loader), total=len(train_loader)):
        optimizer.zero_grad()

        if model_type == "visualbert":
            batch_cap, batch_img, y = data
            batch_inputs = {}
            for k,v in batch_cap.items():
                batch_inputs[k] = v.cuda()
            img_attention_mask = torch.ones(batch_img.shape[:-1], dtype=torch.long)
            img_token_type_ids = torch.ones(batch_img.shape[:-1], dtype=torch.long)
            batch_inputs.update({
                "visual_embeds": batch_img.cuda(),
                "visual_token_type_ids": img_token_type_ids.cuda(),
                "visual_attention_mask": img_attention_mask.cuda(),
                })
        elif model_type == "lxmert":
            batch_cap, batch_box, batch_img, y = data
            batch_inputs = {}
            for k,v in batch_cap.items():
                batch_inputs[k] = v.cuda()
            batch_inputs.update({
                "visual_feats": batch_img.cuda(),
                "visual_pos": batch_box.cuda(),
                })
        elif model_type == "vilt":
            input_ids, pixel_values, y = data
        y = y.cuda()

        if args.amp:
            with autocast():
                if model_type in ["visualbert", "lxmert"]:
                    outputs = model(**batch_inputs, labels=y)
                elif model_type == "vilt":
                    outputs = model(input_ids=input_ids.cuda(), 
                        pixel_values=pixel_values.cuda(), labels=y)
        else:
            if model_type in ["visualbert", "lxmert"]:
                outputs = model(**batch_inputs, labels=y)
            elif model_type == "vilt":
                outputs = model(input_ids=input_ids.cuda(), 
                        pixel_values=pixel_values.cuda(), labels=y)
                #logits = outputs.logits
                #idx = logits.argmax(-1).item()
                #model.config.id2label[idx]

        loss = outputs.loss
        scores = outputs.logits
        wandb.log({"loss": loss})

        if args.amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        # log lr
        lr = optimizer.param_groups[0]['lr']
        wandb.log({"lr": lr})

        train_loss += loss.item()
        #wandb.log({"Loss": loss.item()})
        train_steps += 1
        step_global += 1

        # save model every K iterations
        if step_global % args.checkpoint_step == 0:
            checkpoint_dir = os.path.join(args.output_dir, f"checkpoint_iter_{step_global}")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            if model_type == "visualbert":
                model.save_pretrained(checkpoint_dir)
            elif model_type == "lxmert":
                model.lxmert.save_pretrained(checkpoint_dir)
            elif model_type == "vilt":
                processor.save_pretrained(checkpoint_dir)
                model.save_pretrained(checkpoint_dir)

        # evaluate and save
        if step_global % args.eval_step == 0:
            # evaluate
            acc, _, _, _ = evaluate(val_loader, model, model_type=model_type)
            print (f"====== evaliuate ======")
            print (f"epoch: {epoch}, global step: {step_global}, val performance: {acc}")
            print (f"=======================")
            wandb.log({"eval_acc": acc})
            if val_best_score < acc:
                val_best_score = acc
            else:
                continue
            checkpoint_dir = os.path.join(args.output_dir, f"best_checkpoint")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            if model_type == "visualbert":
                model.save_pretrained(checkpoint_dir)
            elif model_type == "lxmert":
                model.lxmert.save_pretrained(checkpoint_dir)
            elif model_type == "vilt":
                processor.save_pretrained(checkpoint_dir)
                model.save_pretrained(checkpoint_dir)
            print (f"===== best model saved! =======")
                
    train_loss /= (train_steps + 1e-9)
    return train_loss, step_global, val_best_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train')
    parser.add_argument('--img_feature_path', type=str, required=True)
    parser.add_argument('--train_json_path', type=str, required=True)
    parser.add_argument('--val_json_path', type=str, required=True)
    parser.add_argument('--model_type', type=str, default="visualbert", help="visualbert or lxmert or vilt")
    parser.add_argument('--model_path', type=str, default="uclanlp/visualbert-nlvr2-coco-pre")
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--eval_step', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--amp', action="store_true", \
                help="automatic mixed precision training")
    parser.add_argument('--output_dir', type=str, default="./tmp")
    parser.add_argument('--checkpoint_step', type=int, default=100)
    parser.add_argument('--random_seed', type=int, default=42)
    
    args = parser.parse_args()
    
    torch.manual_seed(args.random_seed)

    model_type = args.model_type
    # load model
    if model_type == "visualbert":
        model = VisualBertForVisualReasoning.from_pretrained(args.model_path)
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        processor = None
    elif model_type == "lxmert":
        model = LxmertForPreTraining.from_pretrained(args.model_path)
        model = LxmertForBinaryClassification(model)
        tokenizer = LxmertTokenizer.from_pretrained("unc-nlp/lxmert-base-uncased") 
        processor = None
    elif model_type == "vilt":
        from transformers import ViltProcessor, ViltModel, ViltForImagesAndTextClassification
        config = AutoConfig.from_pretrained("dandelin/vilt-b32-mlm")
        config.num_images = 1
        model = ViltForImagesAndTextClassification(config)
        model.vilt = ViltModel.from_pretrained(args.model_path)
        processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")
        tokenizer = None

        
    # load data
    def collate_fn_batch_visualbert(batch):
        captions, img_features, labels = zip(*batch)
        toks = tokenizer.batch_encode_plus(
            list(captions), 
            max_length=32, 
            padding="max_length", 
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt")
        img_features = torch.stack(img_features, dim=0)
        labels = torch.tensor(labels)
        return toks, img_features, labels
    
    def collate_fn_batch_lxmert(batch):
        captions, boxes, img_features, labels = zip(*batch)
        toks = tokenizer.batch_encode_plus(
            list(captions), 
            max_length=32, 
            padding="max_length", 
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt")
        img_features = torch.stack(img_features, dim=0)
        boxes = torch.stack(boxes)
        labels = torch.tensor(labels)
        return toks, boxes, img_features, labels

    def collate_fn_batch_vilt(batch):
        #"""
        imgs, captions, labels = zip(*batch)
        inputs = processor(images=list(imgs), text=list(captions), return_tensors="pt", 
                padding='max_length', truncation=True, add_special_tokens=True)
        #"""
        #print (inputs.input_ids.shape, inputs.pixel_values.shape)
        """
        inputs, labels = zip(*batch)
        inputs_ids = [i.input_ids for i in inputs]
        pixel_values = [i.pixel_values for i in inputs]
        for i in pixel_values:
            print (i.shape)
        """
        labels = torch.tensor(labels)
        return inputs.input_ids, inputs.pixel_values, labels
        #return torch.cat(inputs_ids, dim=0), torch.cat(pixel_values, dim=0), labels
    
    img_feature_path = args.img_feature_path
    dataset_train = ImageTextClassificationDataset(img_feature_path, args.train_json_path, model_type=model_type, vilt_processor=processor)
    dataset_val = ImageTextClassificationDataset(img_feature_path, args.val_json_path, model_type=model_type)

    if model_type == "visualbert":
        collate_fn_batch = collate_fn_batch_visualbert
    elif model_type == "lxmert":
        collate_fn_batch = collate_fn_batch_lxmert
    elif model_type == "vilt":
        collate_fn_batch = collate_fn_batch_vilt

    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        collate_fn = collate_fn_batch,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=16,)
    val_loader = torch.utils.data.DataLoader(
        dataset_val,
        collate_fn = collate_fn_batch,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=16,)
    
    # mixed precision training 
    if args.amp:
        scaler = GradScaler()
    else:
        scaler = None
    
    optimizer = optim.AdamW(
        [{'params': model.parameters()},], 
        lr=args.learning_rate)
   
    global_step, val_best_score = 0, 0
    for epoch in range(args.epoch):
        loss, global_step, val_best_score = train(args, train_loader, val_loader, model, scaler=scaler, \
                step_global=global_step, epoch=epoch, val_best_score=val_best_score, processor=processor)
        print (f"epoch: {epoch}, global step: {global_step}, loss: {loss}")


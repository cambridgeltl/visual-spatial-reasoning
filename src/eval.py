
import os
import argparse
from tqdm.auto import tqdm
import torch
from transformers import BertTokenizer, VisualBertModel, \
        VisualBertForVisualReasoning, LxmertForPreTraining, LxmertTokenizer
from lxmert_for_classification import LxmertForBinaryClassification
from data import ImageTextClassificationDataset

def evaluate(data_loader, model, model_type="visualbert"):
    model.cuda()
    model.eval()

    correct, total, all_true = 0, 0, 0
    preds = []
    
    for i, data in tqdm(enumerate(data_loader), total=len(data_loader)):
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

        elif  model_type == "lxmert":
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
        with torch.no_grad():
            if model_type in ["visualbert", "lxmert"]:
                outputs = model(**batch_inputs, labels=y)
            elif model_type == "vilt":
                batch_cap = input_ids.cuda()
                batch_img = pixel_values.cuda()
                outputs = model(input_ids=batch_cap, 
                        pixel_values=batch_img)
                #logits = outputs.logits
                #idx = logits.argmax(-1).item()
                #model.config.id2label[idx]

        scores = outputs.logits
        preds_current = torch.argmax(scores, dim=1)
        correct += sum(y == preds_current)
        preds += preds_current.cpu().numpy().tolist()
        total+=batch_img.shape[0]
        all_true += sum(y)

        # print errors
        #print (y != torch.argmax(scores, dim=1))

    # TODO: save also predictions
    return correct / total, total, all_true, preds
            

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='eval')
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--model_type', type=str, default='visualbert')
    parser.add_argument('--img_feature_path', type=str, required=True)
    parser.add_argument('--test_json_path', type=str, required=True)
    parser.add_argument('--output_preds', action='store_true')

    args = parser.parse_args()

    model_type = args.model_type
    # load model
    if model_type == "visualbert":
        model = VisualBertForVisualReasoning.from_pretrained(args.checkpoint_path)
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    elif model_type == "lxmert":
        model = LxmertForPreTraining.from_pretrained(args.checkpoint_path)
        model = LxmertForBinaryClassification(model)
        tokenizer = LxmertTokenizer.from_pretrained("unc-nlp/lxmert-base-uncased") 

    elif model_type == "vilt":
        from transformers import ViltProcessor, ViltForImagesAndTextClassification
        processor = ViltProcessor.from_pretrained(args.checkpoint_path)
        model = ViltForImagesAndTextClassification.from_pretrained(args.checkpoint_path)
    
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
        imgs, captions, labels = zip(*batch)
        inputs = processor(list(imgs), list(captions), return_tensors="pt", padding=True, truncation=True)
        labels = torch.tensor(labels)
        return inputs.input_ids, inputs.pixel_values.unsqueeze(1), labels
        

    img_feature_path = args.img_feature_path
    json_path = args.test_json_path
    if model_type in ["visualbert", "lxmert"]:
        dataset = ImageTextClassificationDataset(img_feature_path, json_path, model_type=model_type)
    elif model_type == "vilt":
        dataset = ImageTextClassificationDataset(img_feature_path, json_path, model_type=model_type, vilt_processor=processor)
    if model_type == "visualbert":
        collate_fn_batch = collate_fn_batch_visualbert
    elif model_type == "lxmert":
        collate_fn_batch = collate_fn_batch_lxmert
    elif model_type == "vilt":
        collate_fn_batch = collate_fn_batch_vilt

    test_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn = collate_fn_batch,
        batch_size=16,
        shuffle=False,
        num_workers=16,)
    acc, total, all_true, preds = evaluate(test_loader, model, model_type=model_type)
    print (f"total example: {total}, # true example: {all_true}, acccuracy: {acc}")

    # save preds
    if args.output_preds:
        with open(os.path.join(args.checkpoint_path, "preds.txt"), "w") as f:
            for i in range(len(preds)):
                f.write(str(preds[i])+"\n")
        



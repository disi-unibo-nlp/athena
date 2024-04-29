import math
import torch
import os
import shutil
import time
from tqdm import tqdm
from transformers import get_scheduler
from testing import test


def save_model(model, tokenizer, seg_model, seg_tokenizer, path):
    print(f"\nSave checkpoint to: {path}")
    if os.path.exists(path):
        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")
    model.save_pretrained(os.path.join(path, "model"))
    tokenizer.save_pretrained(os.path.join(path, "tokenizer"))
    if seg_model is not None and seg_tokenizer is not None:
        seg_model.save_pretrained(os.path.join(path, "seg_model"))
        seg_tokenizer.save_pretrained(os.path.join(path, "seg_tokenizer"))


def train(args, model, tokenizer, train_dataset, eval_dataset, accelerator, text_column, summary_column, run_name,
          min_target_length, max_target_length, stop_ngrams, segmenter):
    num_update_steps_per_epoch = math.ceil(len(train_dataset) / args.gradient_accumulation_steps)
    max_train_steps = args.num_epochs * num_update_steps_per_epoch

    if args.no_seg or args.single_model or not args.alignment_loss:
        parameters = model.parameters()
    else:
        parameters = list(segmenter.get_model().parameters()) + list(model.parameters())

    optimizer = torch.optim.AdamW(params=parameters, lr=args.lr)

    if args.num_warmup_steps is None:
        num_warmup_steps = math.ceil(max_train_steps / 10) * args.gradient_accumulation_steps
    else:
        num_warmup_steps = args.num_warmup_steps * args.gradient_accumulation_steps

    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=max_train_steps * args.gradient_accumulation_steps,
    )

    model, optimizer, lr_scheduler, train_dataset = accelerator.prepare(
        model, optimizer, lr_scheduler, train_dataset
    )

    num_update_steps_per_epoch = math.ceil(len(train_dataset) / args.gradient_accumulation_steps)
    max_train_steps = args.num_epochs * num_update_steps_per_epoch
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    total_batch_size = args.train_batch_size * args.gradient_accumulation_steps

    progress_bar = tqdm(range(max_train_steps))
    best_rouge_scores = 0

    print("\n***** Running training *****")
    print(f"Run name: {run_name}")
    print(f"Num examples = {len(train_dataset)}")
    print(f"Num epochs = {args.num_epochs}")
    print(f"Total train batch size = {total_batch_size}")
    print(f"Gradient accumulation steps = {args.gradient_accumulation_steps}")
    print(f"Total optimization steps = {max_train_steps}\n")

    s_chunks = []
    s_targets = []

    start_time = time.time()

    for epoch in range(num_train_epochs):
        model.train()
        if not args.no_seg:
            segmenter.set_to_train()
        for step, batch in enumerate(train_dataset):

            document = batch[text_column]
            summary = batch[summary_column]

            if args.no_seg:
                salient_chunks = [document]
                salient_targets = [summary]
                loss_alignment = None
            else:
                # Segmentation
                _, _, salient_chunks, salient_targets, loss_alignment = segmenter(
                    document=document,
                    summary=summary,
                    train=args.alignment_loss,
                )

            s_chunks = salient_chunks
            s_targets = salient_targets

            input_dict = tokenizer(salient_chunks, return_tensors="pt", truncation=True, padding=True,
                                   max_length=args.max_source_length)
            input_ids = input_dict["input_ids"].to(model.device)
            attention_mask = input_dict["attention_mask"].to(model.device)

            max_length = int(max_target_length / len(salient_chunks))
            with tokenizer.as_target_tokenizer():
                output_dict = tokenizer(salient_targets, return_tensors="pt", truncation=True, padding=True,
                                        max_length=max_length)
            labels = output_dict["input_ids"].to(model.device)
            labels = labels.where(labels != tokenizer.pad_token_id,
                                  torch.tensor(-100, device=model.device)).to(model.device)
            decoder_attention_mask = output_dict["attention_mask"].to(model.device)

            global_attention_mask = None
            if args.model_checkpoint.partition("/")[-1].startswith("led"):
                global_attention_mask = torch.zeros_like(attention_mask)
                global_attention_mask[:, 0] = 1

            with accelerator.accumulate(model):
                if global_attention_mask is None:
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                        decoder_attention_mask=decoder_attention_mask
                    )
                else:
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        global_attention_mask=global_attention_mask,
                        labels=labels,
                        decoder_attention_mask=decoder_attention_mask
                    )
                loss_summarization = outputs.loss

                if args.no_lgen:
                    if loss_alignment is not None:
                        loss = loss_alignment
                    else:
                        loss = None
                else:
                    if loss_alignment is not None:
                        loss = loss_summarization + loss_alignment
                    else:
                        loss = loss_summarization

                if loss is not None:
                    accelerator.backward(loss)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

            if accelerator.sync_gradients:
                if loss is not None:
                    progress_bar.set_postfix_str(f"Loss: {str(loss.item())}")
                progress_bar.update(1)

        if args.do_eval:
            predictions, rouge_scores, avg_rouge = test(
                args, model, tokenizer, eval_dataset, accelerator, text_column, summary_column, run_name,
                min_target_length, max_target_length, stop_ngrams, segmenter, epoch=epoch, is_eval=True
            )

            current_rouge_scores = avg_rouge

            print(f"\n\nROUGE scores: {rouge_scores}\nAVG ROUGE: {round(current_rouge_scores, 2)}\n")
            if current_rouge_scores > best_rouge_scores:
                best_rouge_scores = current_rouge_scores
                if not args.no_save:
                    if args.no_seg:
                        save_model(model, tokenizer, None, None, path=args.save_to + run_name)
                    else:
                        save_model(model, tokenizer, segmenter.get_model(), segmenter.get_tokenizer(),
                                   path=args.save_to + run_name)

    total_time = time.time() - start_time

    return s_chunks, s_targets, total_time

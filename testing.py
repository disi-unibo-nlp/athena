import torch
import nltk
from datasets import load_metric
from tqdm import tqdm


def get_rouge_scores(predictions, references):
    rouge_metric = load_metric("rouge")
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in predictions]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in references]
    rouge_scores = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    rouge_scores = {key: value.mid.fmeasure * 100 for key, value in rouge_scores.items()}
    rouge_scores = {k: round(v, 2) for k, v in rouge_scores.items()}
    return rouge_scores


def test(args, model, tokenizer, dataset, accelerator, text_column, summary_column, run_name, min_target_length,
         max_target_length, stop_ngrams, segmenter, epoch, is_eval):
    if is_eval and epoch is not None:
        print("\n\n***** Running validation *****")
        print(f"Run name: {run_name} - Epoch: {epoch + 1}/{args.num_epochs}")
    else:
        print(f"\n***** Running test *****")
        print(f"Run name: {run_name}")
    print(f"Num examples = {len(dataset)}")
    print(f"Batch size = {args.eval_batch_size}\n")

    progress_bar = tqdm(range(len(dataset)))

    predictions = []
    model.eval()

    if not args.no_seg:
        segmenter.set_to_eval()

    for step, batch in enumerate(dataset):
        with torch.no_grad():
            document = batch[text_column]

            if args.no_seg:
                chunks = [document]
            else:
                # Segmentation
                chunks, _, _, _, _ = segmenter(document=document, summary=None, train=False)

            min_length = int(min_target_length / len(chunks))
            max_length = int(max_target_length / len(chunks))
            if max_length < args.min_max_len:
                max_length = args.min_max_len

            chunks_batch = [chunks[x:x + args.eval_batch_size] for x in range(0, len(chunks), args.eval_batch_size)]

            document_prediction = []
            for b in chunks_batch:

                input_dict = tokenizer(b, return_tensors="pt", truncation=True, padding=True,
                                       max_length=args.max_source_length)
                input_ids = input_dict["input_ids"].to(model.device)
                attention_mask = input_dict["attention_mask"].to(model.device)

                global_attention_mask = None
                if args.model_checkpoint.partition("/")[-1].startswith("led"):
                    global_attention_mask = torch.zeros_like(attention_mask)
                    global_attention_mask[:, 0] = 1

                if is_eval:
                    model = accelerator.unwrap_model(model)

                if global_attention_mask is None:
                    output_model = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_length=max_length,
                        min_length=min_length,
                        early_stopping=True,
                        num_beams=args.num_beams_eval if is_eval else args.num_beams,
                        length_penalty=args.length_penalty,
                        repetition_penalty=args.repetition_penalty,
                        no_repeat_ngram_size=stop_ngrams,
                    )
                else:
                    output_model = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        global_attention_mask=global_attention_mask,
                        max_length=max_length,
                        min_length=min_length,
                        early_stopping=True,
                        num_beams=args.num_beams_eval if is_eval else args.num_beams,
                        length_penalty=args.length_penalty,
                        repetition_penalty=args.repetition_penalty,
                        no_repeat_ngram_size=stop_ngrams,
                    )

                prediction = " ".join(tokenizer.batch_decode(output_model, skip_special_tokens=True))
                document_prediction.append(prediction)

            predictions.append(" ".join(document_prediction))
            progress_bar.update(1)

    rouge_scores = get_rouge_scores(predictions=predictions, references=dataset[summary_column])
    rouge_scores_list = [rouge_scores["rouge1"], rouge_scores["rouge2"], rouge_scores["rougeLsum"]]
    avg_rouge = round(sum(rouge_scores_list) / len(rouge_scores_list), 2)

    return predictions, rouge_scores, avg_rouge

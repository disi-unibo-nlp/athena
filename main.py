import argparse
import os
import nltk
import torch
import csv
from accelerate import Accelerator
from datasets import load_metric
from transformers import (
    set_seed,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModel
)
from training import train
from testing import test
from process_data import setup_dataset
from segmentation import Segmenter

try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    nltk.download("punkt")


def get_run_name(args):
    model = args.model_checkpoint.partition("/")[-1]
    dataset = args.dataset
    num_train_samples = str(args.max_train_samples)
    data_subset = str(args.data_subset)
    name = "_".join([model, dataset, num_train_samples, data_subset])
    if args.no_seg:
        name += "_no-seg"
    else:
        # name += "_" + args.target_rouge
        if args.single_model:
            name += "_single"
        if args.alignment_loss:
            name += "_align"
        if args.no_lgen:
            name += "_no_lgen"
    return name


def main(args):
    set_seed(args.seed)

    dataset, text_column, summary_column, min_length, max_length, no_repeat_ngram_size = setup_dataset(args)

    train_dataset = dataset["train"]
    eval_dataset = dataset["validation"]
    test_dataset = dataset["test"]

    if args.max_train_samples is not None:
        train_dataset = train_dataset.select(range(args.data_subset * args.max_train_samples,
                                                   args.data_subset * args.max_train_samples + args.max_train_samples))

    if args.max_eval_samples is not None:
        eval_dataset = eval_dataset.select(range(min(len(eval_dataset), args.max_eval_samples)))
    if args.max_test_samples is not None:
        test_dataset = test_dataset.select(range(min(len(test_dataset), args.max_test_samples)))

    accelerator = None
    if args.do_train:
        accelerator = Accelerator(mixed_precision="fp16", gradient_accumulation_steps=args.gradient_accumulation_steps,
                                  cpu=args.cpu)

    run_name = get_run_name(args)

    # if args.bertscore or args.bleurt:
    #     refs = []
    #     preds = []
    #     with open(args.save_predictions_to + run_name + ".csv", "r") as file:
    #         predictions = csv.reader(file, delimiter="|")
    #         for row in predictions:
    #             refs.append(row[0])
    #             preds.append(row[1])
    #     if args.bertscore:
    #         scores = get_bertscore_scores(preds, refs)
    #     else:
    #         scores = get_bleurt_scores(preds, refs)
    #     print(scores)
    #     exit()

    if args.do_train:
        print("Do train")
        tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_checkpoint).to(accelerator.device)

        model.gradient_checkpointing_enable()
        model.config.use_cache = False

        segmenter = None
        if not args.no_seg:
            print("Segmentation")
            if args.single_model:
                print("Single Model")
                seg_tokenizer = tokenizer
                seg_model = model.get_encoder()
            else:
                print("Dual Model")
                seg_tokenizer = AutoTokenizer.from_pretrained(args.segmenter_checkpoint)
                seg_model = AutoModel.from_pretrained(args.segmenter_checkpoint).to(accelerator.device)
                seg_model.gradient_checkpointing_enable()
                seg_model.config.use_cache = False

            seg_model = accelerator.prepare(seg_model)
            segmenter = Segmenter(seg_model, seg_tokenizer, tokenizer, args)
        else:
            print("No Segmentation")

        salient_chunks, salient_targets, total_time = \
            train(args, model, tokenizer, train_dataset, eval_dataset, accelerator, text_column, summary_column,
                  run_name, min_length, max_length, no_repeat_ngram_size, segmenter)

        file_time = f"\n\nTIME: {total_time} seconds"
        with open(args.save_times_to + run_name + ".txt", "w") as file:
            file.write(file_time)

        with open(args.save_pairs_to + run_name + ".csv", "w", encoding="UTF8", newline="") as file:
            writer = csv.writer(file, delimiter="|")
            writer.writerow(["Chunk", "Target"])
            for i in range(len(salient_chunks)):
                writer.writerow([salient_chunks[i], salient_targets[i]])

    if args.do_test:
        print("Do test")

        tokenizer = AutoTokenizer.from_pretrained(os.path.join(args.save_to + run_name, "tokenizer"))
        model = AutoModelForSeq2SeqLM.from_pretrained(os.path.join(args.save_to + run_name, "model"))

        if torch.cuda.is_available():
            model = model.cuda()

        segmenter = None
        if not args.no_seg:
            print("Segmentation")
            if args.single_model:
                print("Single Model")
                seg_tokenizer = tokenizer
                seg_model = model.get_encoder()
            else:
                print("Dual Model")
                seg_tokenizer = AutoTokenizer.from_pretrained(os.path.join(args.save_to + run_name, "seg_tokenizer"))
                seg_model = AutoModel.from_pretrained(os.path.join(args.save_to + run_name, "seg_model"))

                if torch.cuda.is_available():
                    seg_model = seg_model.cuda()

            segmenter = Segmenter(seg_model, seg_tokenizer, tokenizer, args)

        predictions, rouge_scores, avg_rouge = test(
            args, model, tokenizer, test_dataset, accelerator, text_column, summary_column, run_name, min_length,
            max_length, no_repeat_ngram_size, segmenter, epoch=None, is_eval=False
        )

        print(f"\n\nROUGE scores: {rouge_scores}\nAVG ROUGE: {avg_rouge}\n")

        file_log_results = f"\n\nROUGE scores: {rouge_scores}\nAVG ROUGE: {avg_rouge}\n\n"
        with open(args.save_results_to + run_name + ".txt", "w") as file:
            file.write(file_log_results)

        with open(args.save_predictions_to + run_name + ".csv", "w", encoding="UTF8", newline="") as file:
            writer = csv.writer(file, delimiter="|")
            writer.writerow(["Gold-Target", "Prediction"])
            for i in range(len(predictions)):
                writer.writerow([test_dataset[summary_column][i], predictions[i]])


def get_bertscore_scores(predictions, references):
    bertscore = load_metric("bertscore", "microsoft/deberta-xlarge-mnli")
    scores = bertscore.compute(predictions=predictions, references=references, lang="en")
    scores = [round(v, 2) for v in scores["f1"]]
    scores = sum(scores)/len(scores)
    return scores


def get_bleurt_scores(predictions, references):
    bleurt = load_metric("bleurt", "bleurt-20", module_type="metric")
    scores = bleurt.compute(predictions=predictions, references=references)
    scores = [round(v, 2) for v in scores["scores"]]
    scores = sum(scores)/len(scores)
    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Athena")
    parser.add_argument("--model_checkpoint", default="facebook/bart-base",
                        help="The Hugging Face model checkpoint for the summarization.")
    parser.add_argument("--segmenter_checkpoint", default="sentence-transformers/all-MiniLM-L6-v2",
                        help="The Hugging Face model checkpoint for the segmentation.")
    parser.add_argument("--chunk_min_len", type=int, default=256, help="The min chunk size for segmentation.")
    parser.add_argument("--save_to", default="./models/", help="The path to save the trained model.")
    parser.add_argument("--dataset_path", default="./datasets/", help="The path to save the datasets.")
    parser.add_argument("--dataset", default="govreport", help="The dataset to use.")
    parser.add_argument("--num_epochs", type=int, default=20, help="Num training epochs.")
    parser.add_argument("--max_train_samples", type=int, default=None, help="Num training instances.")
    parser.add_argument("--max_eval_samples", type=int, default=None, help="Num validation instances.")
    parser.add_argument("--max_test_samples", type=int, default=None, help="Num test instances.")
    parser.add_argument("--seed", type=int, default=42, help="The seed for reproducibility.")
    parser.add_argument("--data_subset", type=int, default=1, choices=[1, 2, 3, 4, 5],
                        help="The subset for experiments.")
    parser.add_argument("--num_beams", type=int, default=4, help="Parameter for generation at test time.")
    parser.add_argument("--num_beams_eval", type=int, default=2, help="Parameter for generation at validation time.")
    parser.add_argument("--lr", type=float, default=3e-5, help="The learning rate.")
    parser.add_argument("--length_penalty", type=float, default=2.0, help="Parameter for generation.")
    parser.add_argument("--repetition_penalty", type=float, default=1.0, help="Parameter for generation.")
    parser.add_argument("--train_batch_size", type=int, default=1, help="The batch size for training.")
    parser.add_argument("--eval_batch_size", type=int, default=1, help="The batch size for validation and test.")
    parser.add_argument("--min_max_len", type=int, default=20, help="The min max length.")
    parser.add_argument("--max_source_length", type=int, default=1024, help="The src max length.")
    parser.add_argument("--min_target_length", type=int, default=None, help="The tgt min length.")
    parser.add_argument("--max_target_length", type=int, default=None, help="The tgt max length.")
    parser.add_argument("--num_warmup_steps", type=int, default=None, help="The number of warmup steps.")
    parser.add_argument("--no_repeat_ngram_size", type=int, default=None, help="no_repeat_ngram_size.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="The gradient accumulation steps.")
    parser.add_argument("--do_train", default=False, action="store_true", help="If do train.")
    parser.add_argument("--do_eval", default=False, action="store_true", help="If do validation.")
    parser.add_argument("--do_test", default=False, action="store_true", help="If do test.")
    parser.add_argument("--no_save", default=False, action="store_true", help="If not save model checkpoints.")
    parser.add_argument("--single_model", default=False, action="store_true", help="If segment with the same model.")
    parser.add_argument("--alignment_loss", default=False, action="store_true", help="If train the segmentation model.")
    parser.add_argument("--alpha", type=float, default=0.5, help="The modulator in the loss.")
    parser.add_argument("--sent_batch", type=int, default=4, help="The sentences batch size for segmentation.")
    parser.add_argument("--save_results_to", default="./results/", help="The path to save the results.")
    parser.add_argument("--save_predictions_to", default="./predictions/", help="The path to save the predictions.")
    parser.add_argument("--save_pairs_to", default="./pairs/", help="The path to save the pairs.")
    parser.add_argument("--save_times_to", default="./times/", help="The path to save the running times.")
    parser.add_argument("--print_stats", default=False, action="store_true", help="If print the chunk statistics.")
    parser.add_argument("--target_rouge", default="rouge1", choices=["rouge1", "rouge2", "rougeL"],
                        help="The type of target assignment.")
    parser.add_argument("--cpu", default=False, action="store_true", help="If use the cpu.")
    parser.add_argument("--no_seg", default=False, action="store_true", help="If do not use text segmentation.")
    parser.add_argument("--bertscore", default=False, action="store_true", help="If evaluate with bertscore.")
    parser.add_argument("--bleurt", default=False, action="store_true", help="If evaluate with bleurt.")
    parser.add_argument("--no_lgen", default=False, action="store_true", help="If use do not use loss generation.")
    parser_args = parser.parse_args()

    main(parser_args)

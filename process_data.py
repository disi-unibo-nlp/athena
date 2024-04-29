import os
import re
from datasets import load_from_disk, load_dataset, DatasetDict

summarization_name_mapping = {
    # DATASET: (INPUT_COLUMN, OUTPUT_COLUMN, MIN_TGT_LENGTH, MAX_TGT_LENGTH, NO_REPEAT_NGRAM_SIZE)
    "pubmed": ("article", "abstract", 100, 300, 3),
    "arxiv": ("article", "abstract", 150, 350, 3),
    "govreport": ("report", "summary", 500, 1000, 5),
    "billsum": ("text", "summary", 100, 300, 3),
    "big_patent": ("description", "abstract", 50, 200, 3),
    "xsum": ("document", "summary", 10, 50, 3),
    "cnn_dailymail": ("article", "highlights", 20, 100, 3),
}


def setup_dataset(args):

    dataset_columns = summarization_name_mapping.get(args.dataset, None)
    text_column = dataset_columns[0]
    summary_column = dataset_columns[1]
    min_length = dataset_columns[2] if args.min_target_length is None else args.min_target_length
    max_length = dataset_columns[3] if args.max_target_length is None else args.max_target_length
    no_repeat_ngram_size = dataset_columns[4] if args.no_repeat_ngram_size is None else args.no_repeat_ngram_size

    path = args.dataset_path + args.dataset

    if not os.path.exists(args.dataset_path + args.dataset):
        if args.dataset == "pubmed":
            dataset = load_dataset("ccdv/pubmed-summarization")
        elif args.dataset == "arxiv":
            dataset = load_dataset("ccdv/arxiv-summarization")
        elif args.dataset == "govreport":
            dataset = load_dataset("ccdv/govreport-summarization")
        elif args.dataset == "cnn_dailymail":
            dataset = load_dataset(args.dataset, "3.0.0")
        else:
            dataset = load_dataset(args.dataset)

        train_dataset = dataset["train"].filter(
            lambda ex: (ex[text_column] is not None and ex[summary_column] is not None) and len(
                ex[text_column]) > 0 and len(ex[summary_column]) > 0).map(
            lambda ex: clean_dataset(ex, text_column, summary_column))

        if args.dataset == "billsum":
            eval_dataset = dataset["test"].filter(
                lambda ex: (ex[text_column] is not None and ex[summary_column] is not None) and len(
                    ex[text_column]) > 0 and len(ex[summary_column]) > 0).map(
                lambda ex: clean_dataset(ex, text_column, summary_column))
        else:
            eval_dataset = dataset["validation"].filter(
                lambda ex: (ex[text_column] is not None and ex[summary_column] is not None) and len(
                    ex[text_column]) > 0 and len(ex[summary_column]) > 0).map(
                lambda ex: clean_dataset(ex, text_column, summary_column))
        test_dataset = dataset["test"].filter(
            lambda ex: (ex[text_column] is not None and ex[summary_column] is not None) and len(
                ex[text_column]) > 0 and len(ex[summary_column]) > 0).map(
            lambda ex: clean_dataset(ex, text_column, summary_column))

        train_dataset = train_dataset.remove_columns(
            list(set(train_dataset.column_names) - {text_column, summary_column}))
        eval_dataset = eval_dataset.remove_columns(
            list(set(eval_dataset.column_names) - {text_column, summary_column}))
        test_dataset = test_dataset.remove_columns(
            list(set(test_dataset.column_names) - {text_column, summary_column}))

        dataset = DatasetDict({"train": train_dataset, "validation": eval_dataset, "test": test_dataset})

        dataset.save_to_disk(path)
    else:
        dataset = load_from_disk(path)
    return dataset, text_column, summary_column, min_length, max_length, no_repeat_ngram_size


def clean_dataset(data, text_column, summary_column):
    data[text_column] = clean_text(data[text_column])
    data[summary_column] = clean_text(data[summary_column])
    return data


def replace_semicolon(text, threshold=100):
    new_text = ""
    for subset in re.split(";", text):
        subset = subset.strip()
        if len(subset.split()) > threshold:
            new_text += ". " + subset[0].upper() + subset[1:]  # Turn the semicolon into a period.
        else:
            new_text += ", " + subset  # Turn the semicolon into a comma.
    return new_text


def clean_text(text):
    # lowering
    text = text.lower()
    # encoding with ascii
    text = text.encode(encoding="ascii", errors="ignore").decode()
    # removing possible extra white space
    text = " ".join([x for x in text.split(" ")])
    # removing urls
    text = re.sub("https?://.*[\r\n]*", "", text)
    text = re.sub("http?://.*[\r\n]*", "", text)
    # removing hashtag and other special characters
    text = re.sub("#", "", text)
    text = re.sub('\\\\', "", text)
    text = re.sub("@", "", text)
    text = re.sub("&", "", text)
    text = re.sub("$", "", text)
    # remove section headers and uniform acronyms
    text = re.compile('SECTION [0-9]{1,2}\.|\nSEC\.* [0-9]{1,2}\.|Sec\.* [0-9]{1,2}\.').sub("", text)
    text = text.replace("U.S.", "US")
    text = text.replace("SEC.", "")
    text = text.replace("Sec.", "")
    text = re.compile('[Uu]\.*[Ss]\.*[Cc]\.]+').sub("USC", text)
    # remove parentheticals
    text = re.compile('\([^(]+[^(]+\)').sub("", text)
    # get rid of enums as bullets or ` as bullets
    text = re.compile('\n[ \t]*`*\([a-zA-Z0-9]*\)').sub(" ", text)
    # clean html.
    text = text.replace("&lt;all&gt;", "")
    # remove annoying punctuation
    text = re.compile(r'([%s])' % re.escape('"#%&\*\+/<=>@[\]^{|}~_'), re.UNICODE).sub("", text)
    # get rid of long sequences of dashes
    text = re.compile('--+').sub(" ", text)
    # remove newlines, tabs, and extra spaces
    text = re.compile('\s+').sub(" ", text)
    # if we ended up with "empty" sentences, get rid of them
    text = re.compile('[,.] *[.,]').sub(".", text)
    # get rid of anything that is not a word from the start of the text
    text = re.compile('^[^A-Za-z]*').sub("", text)
    # get rid of semicolons
    text = replace_semicolon(text)
    # make sure there is a space between periods and the start of the sentence
    text = re.compile('\.([A-Za-z])').sub(". \g<1>", text)
    text = text.replace("SHORT TITLE. ", "")
    # uniform words
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can\'t", "can not", text)
    text = re.sub(r"n\'t", " not", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'s", " is", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'t", " not", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'m", " am", text)
    # fix quotes
    text = text.replace("’", "'")
    text = text.replace("`", "'")
    text = text.replace("\'", "'")
    text = text.replace("\n", "")
    text = text.replace('``', '"')
    text = text.replace('\'\'', '"')
    return text

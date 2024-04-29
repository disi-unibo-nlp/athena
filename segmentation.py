import torch
import torch.nn.functional as F
from rouge_score import rouge_scorer
from nltk.tokenize import sent_tokenize


class Segmenter(torch.nn.Module):
    def __init__(self, model, seg_tokenizer, tokenizer, args):
        super(Segmenter, self).__init__()
        self.model = model
        self.seg_tokenizer = seg_tokenizer
        self.tokenizer = tokenizer
        self.sent_split = sent_tokenize
        self.device = model.device
        self.chunk_min_len = args.chunk_min_len
        self.chunk_max_len = args.max_source_length
        self.sent_batch = args.sent_batch
        self.rouge_metric = args.target_rouge
        self.scorer = rouge_scorer.RougeScorer([self.rouge_metric], use_stemmer=True)
        self.loss_fn = torch.nn.CosineEmbeddingLoss()

    def get_model(self):
        return self.model

    def get_tokenizer(self):
        return self.seg_tokenizer

    def get_token_embeddings(self, batch, max_val):
        input_dict = self.seg_tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
        input_ids = input_dict["input_ids"].to(self.device)
        attention_mask = input_dict["attention_mask"].to(self.device)
        model_output = self.model(input_ids=input_ids, attention_mask=attention_mask)[0]
        shape = model_output.shape
        shared_attention_mask = torch.ones((shape[0], max_val - shape[1], shape[2]), device=self.device)
        embeddings = torch.cat((model_output, shared_attention_mask), 1)
        return embeddings

    def embed_inputs(self, sentences, grad):
        token_embeddings = []
        input_dict = self.seg_tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
        attention_mask_full = input_dict["attention_mask"].to(self.device)
        max_val = len(input_dict["input_ids"][0])
        sentences_batch = [sentences[x:x + self.sent_batch] for x in range(0, len(sentences), self.sent_batch)]

        if grad:
            for batch in sentences_batch:
                embeddings = self.get_token_embeddings(batch, max_val)
                token_embeddings.extend(embeddings)
        else:
            self.model.eval()
            with torch.no_grad():
                for batch in sentences_batch:
                    embeddings = self.get_token_embeddings(batch, max_val)
                    token_embeddings.extend(embeddings)
            self.model.train()

        token_embeddings = torch.stack(token_embeddings)
        input_mask_expanded = attention_mask_full.unsqueeze(-1).expand(token_embeddings.size()).float()
        pool = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return F.normalize(pool, p=2, dim=1)

    def save_pretrained(self, path):
        self.model.save_pretrained(path)

    def set_to_train(self):
        self.model.train()

    def set_to_eval(self):
        self.model.eval()

    def is_smaller(self, chunk_len):
        return chunk_len < self.chunk_min_len

    def will_be_larger(self, chunk_len, sentence_len):
        return (chunk_len + sentence_len) > self.chunk_max_len

    def look_ahead(self, sentences, sentences_len, idx, sentence_len, dictionary):
        chunk_embeddings = []
        chunk = []
        chunk_len = sentence_len
        for j, s in enumerate(sentences[idx + 1:]):
            if self.is_smaller(chunk_len - sentence_len):
                s_len = sentences_len[idx + j + 1]
                if self.will_be_larger(chunk_len, s_len):
                    if len(chunk) == 0:
                        chunk.append(s)
                        chunk_len += s_len
                    else:
                        break
                else:
                    chunk.append(s)
                    chunk_embeddings.append(dictionary.get(s))
                    chunk_len += s_len
            else:
                break
        return chunk, chunk_embeddings, chunk_len

    def get_target(self, chunks, target):
        if len(chunks) == 1:
            targets = [target]
            return chunks, targets, targets
        else:
            targets = [""] * len(chunks)
            summary_sent = self.sent_split(target)
            for sent in summary_sent:
                scores = [self.scorer.score(chunk, str(target + " " + sent).strip())[self.rouge_metric].precision
                          for chunk, target in zip(chunks, targets)]
                indices = [i[0] for i in sorted(enumerate(scores), key=lambda x: x[1], reverse=True)]
                targets[indices[0]] = str(targets[indices[0]] + " " + sent).strip()
            salient_chunks = []
            salient_targets = []
            for i in range(len(chunks)):
                if len(targets[i]) > 0:
                    salient_chunks.append(chunks[i])
                    salient_targets.append(targets[i])
            return salient_chunks, targets, salient_targets

    def get_cosine_embedding_loss(self, chunks, targets):
        chunk_embeddings = torch.stack([self.embed_inputs(self.sent_split(c), grad=True).mean(0) for c in chunks])
        target_embeddings = torch.stack([self.embed_inputs(self.sent_split(t), grad=True).mean(0) for t in targets])
        loss = self.loss_fn(chunk_embeddings, target_embeddings, torch.ones(len(chunk_embeddings), device=self.device))
        return loss

    def forward(
        self,
        document,
        summary,
        train
    ):
        chunks = []
        sentences = self.sent_split(document)
        sentences_len = [len(s) for s in self.tokenizer(sentences, add_special_tokens=False)["input_ids"]]
        if sum(sentences_len) + 2 <= self.chunk_max_len:
            chunks.append(sentences)
        else:
            sent_embeddings = self.embed_inputs(sentences, grad=False)
            dictionary = dict(zip(sentences, sent_embeddings))
            count = -1
            curr_chunk = []
            curr_chunk_len = 0
            for i, (curr_sent, curr_sent_len) in enumerate(zip(sentences[:-1], sentences_len[:-1])):
                if i > count:
                    if self.is_smaller(curr_chunk_len):
                        if self.will_be_larger(curr_chunk_len, curr_sent_len):
                            if curr_chunk_len == 0:
                                curr_chunk.append(curr_sent)
                                curr_chunk_len += curr_sent_len
                            else:
                                chunks.append(curr_chunk)
                                curr_chunk = [curr_sent]
                                curr_chunk_len = curr_sent_len
                        else:
                            curr_chunk.append(curr_sent)
                            curr_chunk_len += curr_sent_len
                    else:
                        if self.will_be_larger(curr_chunk_len, curr_sent_len):
                            chunks.append(curr_chunk)
                            curr_chunk = [curr_sent]
                            curr_chunk_len = curr_sent_len
                        else:
                            next_chunk, next_chunk_embeds, next_chunk_len = self.look_ahead(
                                sentences, sentences_len, i, curr_sent_len, dictionary
                            )
                            if len(next_chunk_embeds) > 0:
                                curr_chunk_embeds = torch.stack([dictionary.get(s) for s in curr_chunk])
                                curr_sent_emb = dictionary.get(curr_sent)
                                curr_sim = F.cosine_similarity(curr_sent_emb, curr_chunk_embeds).mean()
                                next_sim = F.cosine_similarity(curr_sent_emb, torch.stack(next_chunk_embeds)).mean()
                                if curr_sim > next_sim:
                                    curr_chunk.append(curr_sent)
                                    curr_chunk_len += curr_sent_len
                                else:
                                    chunks.append(curr_chunk)
                                    curr_chunk = [curr_sent, *next_chunk]
                                    curr_chunk_len = next_chunk_len
                                    count = i + len(next_chunk)
                            else:
                                curr_chunk.append(curr_sent)
                                curr_chunk_len += curr_sent_len
            if curr_chunk[-1] != sentences[-1]:
                last_sent = sentences[-1]
                last_sent_len = sentences_len[-1]

                if self.will_be_larger(curr_chunk_len, last_sent_len):
                    chunks.append(curr_chunk)
                    chunks.append([last_sent])
                else:
                    curr_chunk.append(last_sent)
                    curr_chunk_len += last_sent_len
                    chunks.append(curr_chunk)
            else:
                chunks.append(curr_chunk)

        chunks = [" ".join(c) for c in chunks]

        targets = None
        salient_chunks = None
        salient_targets = None
        loss = None

        if summary is not None:
            salient_chunks, targets, salient_targets = self.get_target(chunks, summary)
            if train and len(salient_chunks) > 1:
                loss = self.get_cosine_embedding_loss(salient_chunks, salient_targets)
        return chunks, targets, salient_chunks, salient_targets, loss

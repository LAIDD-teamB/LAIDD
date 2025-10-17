"""
If you want to use following Dataset with DataLoader, please specify collate_fn option 
    when you initialize DataLoader. We coded proper collate_fn to be used in
    each Dataset class.
"""

from .vocab import Vocabulary
from torch.utils import data
from torch.nn.utils import rnn
import torch
from typing import List, Callable

class VectorDataset(data.Dataset):
    def __init__(self, vecs):
        self.vecs = vecs
    
    def __len__(self):
        return len(self.vecs)
    
    def __getitem__(self, idx):
        return torch.tensor(self.vecs[idx])
    
    def collate_fn(self, batch):
        return torch.stack(batch)

class StringDataset(data.Dataset):
    """
    Dataset for language modeling based on character or token sequences.

    Each string is tokenized and encoded into indices using a provided vocabulary.
    An <EOS> token is automatically appended to the end of each sequence.

    :param voc: Vocabulary object used for tokenization and encoding.
    :param strings: List of string sequences to be encoded.
    """
    def __init__(self, voc:Vocabulary, strings):
        self.voc = voc
        self.strings = strings
    
    def __len__(self):
        return len(self.strings)
    
    def __getitem__(self, idx):
        toks = self.voc.tokenize(self.strings[idx]) + [self.voc.id2tok[self.voc.get_eosi()]]
        return torch.tensor(self.voc.encode(toks))

    def collate_fn(self, batch):
        """
        Collate a batch of encoded sequences by applying padding.

        :param batch: List of 1D torch.Tensor sequences.
        :return: Padded 2D tensor with shape (batch_size, max_seq_len).
        """        
        string_ids_tensor = rnn.pad_sequence(batch, batch_first=True, 
                                            padding_value=self.voc.get_padi())
        return string_ids_tensor
    
class String2FeatDataset(data.Dataset):
    """
    Dataset that maps string inputs to feature vectors using a custom featurization function.

    :param voc: Vocabulary object (not necessarily used directly here but kept for consistency).
    :param strings: List of string inputs.
    :param featzr: Callable function that transforms a string into a feature vector (e.g., numpy array or list).
    """
    def __init__(self, voc:Vocabulary, strings, featzr:Callable):
        self.voc = voc
        self.strings = strings
        self.featzr = featzr
    
    def __len__(self):
        return len(self.strings)
    
    def __getitem__(self, idx):
        feat_vec = self.featzr(self.strings[idx])
        return torch.tensor(feat_vec)
    
    def collate_fn(self, batch):
        return torch.vstack(batch)

class StringLabelDataset(data.Dataset):
    """
    Dataset for paired sequence and label data.

    This dataset handles string-to-token sequence encoding and stores associated labels.
    It appends an <EOS> token to each input sequence and applies padding during batching.
    For LSTMClassifier, you need to put one_hot encoding as labels.
    If labels are in 1-d, you may use labels=value_list.reshape(-1,1) before initializing this.

    :param voc: Vocabulary object for tokenization and encoding.
    :param strings: List of string inputs.
    :param labels: List of labels, each as a n-d numpy array or list.
    """
    def __init__(self, voc:Vocabulary, strings, labels):
        if len(strings) != len(labels):
            raise ValueError("Please make sure the lengths of string list and label list are same!!")
        self.voc = voc
        self.strings = strings
        self.labels = labels
    
    def __len__(self):
        return len(self.strings)
    
    def __getitem__(self, idx):
        toks = self.voc.tokenize(self.strings[idx]) + [self.voc.id2tok[self.voc.get_eosi()]]
        return torch.tensor(self.voc.encode(toks)), torch.tensor(self.labels[idx])

    def collate_fn(self, batch):
        string_ids_list, label_list = [],[]
        for string_ids, label in batch:
            string_ids_list.append(string_ids)
            label_list.append(label)
        label_tensor = torch.vstack(label_list)
        string_ids_tensor = rnn.pad_sequence(string_ids_list, batch_first=True, 
                                            padding_value=self.voc.get_padi())
        return string_ids_tensor, label_tensor

class String2StringDataset(data.Dataset):
    """
    Dataset for string-to-string sequence modeling (e.g., machine translation, transcription).

    This dataset is initialized with two vocabularies and two lists of paired sequences.
    Each sequence is tokenized and encoded using its respective vocabulary, and an <EOS> token
    is appended at the end of both input and output sequences.

    :param in_voc: Vocabulary used for input sequence encoding.
    :param in_strs: List of input strings.
    :param out_voc: Vocabulary used for output sequence encoding.
    :param out_strs: List of output strings.
    """
    def __init__(self, in_voc:Vocabulary, in_strs, out_voc:Vocabulary, out_strs):
        if len(in_strs) != len(out_strs):
            raise ValueError("Please make sure the lengths of the two string lists are the same!!")
        self.in_voc = in_voc
        self.in_strs = in_strs
        self.out_voc = out_voc
        self.out_strs = out_strs
        
    def __len__(self):
        return len(self.in_strs)
    
    def __getitem__(self, idx):
        in_toks = self.in_voc.tokenize(self.in_strs[idx]) + [self.in_voc.id2tok[self.in_voc.get_eosi()]]
        out_toks = self.out_voc.tokenize(self.out_strs[idx]) + [self.out_voc.id2tok[self.out_voc.get_eosi()]]
        return torch.tensor(self.in_voc.encode(in_toks)), torch.tensor(self.out_voc.encode(out_toks))
    
    def collate_fn(self, batch):
        in_tokids_list, out_tokids_list = [], []
        for in_tokids, out_tokids in batch:
            in_tokids_list.append(in_tokids)
            out_tokids_list.append(out_tokids)
        in_tokids_tensor = rnn.pad_sequence(in_tokids_list, batch_first=True, 
                                            padding_value=self.in_voc.get_padi())
        out_tokids_tensor = rnn.pad_sequence(out_tokids_list, batch_first=True, 
                                            padding_value=self.out_voc.get_padi())
        return in_tokids_tensor, out_tokids_tensor

class SeqMasker:
    """
    Utility class for handling sequence-based preprocessing tasks.

    Assumes that each input sequence contains at least one <EOS> token.
    Tokens appearing after the first <EOS> will be treated as <PAD> during certain operations.
    
    :param voc: Vocabulary object containing token indices.
    :param device: Torch device for tensor computations.
    """
    def __init__(self, voc:Vocabulary, device='cpu'):
        self.voc = voc
        self.device = device

    def find_first_eos(self, batch_seqs:torch.Tensor):
        """
        NOT SURE IF I WILL NEED THIS FUNC

        Find the position of the first <EOS> token for each sequence in the batch.

        :param batch_seqs: Tensor of token indices with shape (batch_size, seq_len).
        :return: List of first <EOS> positions (or seq_len + 1 if not found).
        """
        bsz, slen = batch_seqs.shape
        eosi = self.voc.get_eosi()
        row_col = torch.nonzero(batch_seqs==eosi).cpu().numpy()
        # row_col is in increasing order
        fe_poss = [slen+1 for k in range(bsz)]
        for _rc in row_col[::-1]:
            fe_poss[_rc[0]] = _rc[1]
        return fe_poss

    def pad_after_eos(self, batch_seqs:torch.Tensor):
        """
        Replace all tokens after the first <EOS> in each sequence with <PAD>.

        Note that returned tensor's length could be shorter than the input by truncation.

        :param batch_seqs: Tensor of token sequences with shape (batch_size, seq_len).
        :raises ValueError: If any sequence does not contain an <EOS> token.
        :return: Truncated and padded sequences as a tensor.
        """
        bsz, slen = batch_seqs.shape
        erows, ecols = torch.where(batch_seqs==self.voc.get_eosi())
        # e.g. [[C,c,EOS,Na],[c,EOS,1,EOS]] -> erows=[0,1,1], ecols=[2,1,3]

        trunc_seqs = [None]*bsz  # will collect truncated seqs here
        rbag = set(range(bsz))  # checking if certain row is already met
        for ri, ci in zip(erows.cpu().numpy(), ecols.cpu().numpy()):
            if ri in rbag:
                trunc_seqs[ri] = batch_seqs[ri,:ci+1]  # :ci+1 for including EOS
                rbag.remove(ri)

        if None in trunc_seqs:
            errmsg = "pad_after_eos() expects batch_seqs to have EOS in all its examples,"
            errmsg += "but " + str(rbag) + " examples don't have EOS!!"
            raise ValueError(errmsg)
        
        padded_seqs = rnn.pad_sequence(trunc_seqs, batch_first=True, 
                                       padding_value=self.voc.get_padi())
        return padded_seqs.to(self.device)

    def build_bert_mlm_input(self, batch_seqs:torch.Tensor, pad_mask:torch.Tensor, 
                             p_pos_pred=0.15, p_mask=0.8, p_rand=0.1, p_unch=0.1):
        """
        Create BERT-style masked language model (MLM) inputs.

        This method randomly selects a subset of positions (not padding or CLS) to predict.
        Masking behavior:
        - 80% replaced with <MSK>
        - 10% replaced with random token
        - 10% left unchanged

        :param batch_seqs: Input token sequences (batch_size, seq_len).
        :param pad_mask: Boolean tensor marking padding positions.
                         Shape must match batch_seqs.
        :param p_pos_pred: Probability of predicting a position (default: 0.15).
        :param p_mask: Fraction of predicted positions replaced with <MSK> (default: 0.8).
        :param p_rand: Fraction of predicted positions replaced with random tokens (default: 0.1).
        :param p_unch: Fraction of predicted positions left unchanged (default: 0.1).
        :raises ValueError: If shape mismatch or <CLS> not present at beginning of sequences.
        :return: Tuple of (MLM input tensor, prediction mask).
            - bert_input: (bs, seqlen) encoded tokens tensor 
            - pos_pred_bool: (bs, seqlen) whether BERT should predict the position (of batch_seqs) or not
        """
        vocsz = self.voc.vocab_size
        cls_ind = self.voc.get_clsi()
        bs, seqlen = batch_seqs.shape
        if (bs,seqlen) != pad_mask.shape:
            raise ValueError("shapes of batch_seqs and pad_mask are different!")
        if (p_mask + p_rand + p_unch) != 1.0:
            raise ValueError("p_mask + p_rand + p_unch should equal to 1.0 !!!")
        for seq in batch_seqs:
            if seq[0] != cls_ind:
                raise ValueError("seq in a batch should have <CLS> token at the beginning!")
        cls_mask = torch.zeros((bs,seqlen)).bool().to(self.device)
        cls_mask[:,0] = True

        # random number 0~1 sampling
        pos_pred_pmat = torch.rand((bs, seqlen)).to(self.device)

        # whether the position to be predicted or not
        pos_pred_bool = (pos_pred_pmat < p_pos_pred)
        # We don't predict padding positions
        pos_pred_bool = pos_pred_bool & (~pad_mask)
        # We don't predict CLS positions
        pos_pred_bool = pos_pred_bool & (~cls_mask)

        # rand num for input mask type selection 
        mask_type_pmat = torch.rand((bs, seqlen)).to(self.device)
        # <MSK> bool mat
        mask_bool = (mask_type_pmat < p_mask)
        # p_mask ~ (p_mask + p_rand) range is used for random token change
        rand_tok_bool = (mask_type_pmat >= p_mask) & (mask_type_pmat < (p_mask+p_rand))
        # others are unchanged
        unch_bool = (mask_type_pmat >= (p_mask+p_rand))

        # apply only for prediction positions
        mask_bool = mask_bool & pos_pred_bool
        rand_tok_bool = rand_tok_bool & pos_pred_bool
        unch_bool = unch_bool & pos_pred_bool  # not used
     
        # building MLM input
        bert_input = batch_seqs.clone()  # should it be detached ????
        bert_input = bert_input.masked_fill(mask_bool, self.voc.get_mski())
        bert_input[rand_tok_bool] = torch.randint(low=0, high=vocsz, size=(rand_tok_bool.sum(),)).to(self.device)

        return bert_input, pos_pred_bool

#!  -*-coding: utf-8-*-


def padding(sequences, seq_length):
    seq_zip = zip(sequences, seq_length)
    seq_sorted = sorted(seq_zip, key=lambda x: x[1], reverse=True)
    max_len = seq_sorted[0][1]
    out_seq = []
    out_len = []
    for item in seq_sorted:
        seq, length = item
        pad = [0] * (max_len - len(seq))
        seq += pad
        out_seq.append(seq)
        out_len.append(out_len)
    return out_seq, out_len

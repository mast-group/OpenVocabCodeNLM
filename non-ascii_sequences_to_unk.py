import sys

UNKNOWN_WORD = "-UNK-"

def non_ascii_seq_to_unk(source_file, destination_file):
    with open(source_file, 'r') as rf:
        with open(destination_file, 'w') as wf:
            for line in rf:
                in_non_ascii_seq = False
                for char in line:
                    if ord(char) < 128:
                        if in_non_ascii_seq:
                            wf.write(UNKNOWN_WORD)
                            in_non_ascii_seq = False
                        wf.write(char)
                    else:
                        in_non_ascii_seq = True




if __name__=="__main__":
    if len(sys.argv) != 3:
        print 'Usage non-ascii_sequences_to_unk.py source_file destination_file'
        sys.exit(1)
    non_ascii_seq_to_unk( sys.argv[1], sys.argv[2] )

import sys
JAVA_KEYWORDS = set([
    'abstract',
    'assert',
    'boolean',
    'break',
    'byte',
    'case',
    'catch',
    'char',
    'class',
    'const',
    'continue',
    'default',
    'do',
    'double',
    'else',
    'enum',
    'extends',
    'final',
    'finally',
    'float',
    'for',
    'goto',
    'if',
    'implements',
    'import',
    'instanceof',
    'int',
    'interface',
    'long',
    'native',
    'new',
    'package',
    'private',
    'protected',
    'public',
    'return',
    'short',
    'static',
    'strictfp',
    'super',
    'switch',
    'synchronized',
    'this',
    'throw',
    'throws',
    'transient',
    'try',
    'void',
    'volatile',
    'while',

    'byte',
    'short',
    'int',
    'long',
    'float',
    'double',
    'char',
    # 'String',
    'boolean',
    
    'true',
    'false'
])


def get_Java_identifier_mapping(code):
    """[summary]
    
    Arguments:
        code {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """
    identifier_mapping = []

    in_string = False
    for token in code:
        if not in_string:
            if token in JAVA_KEYWORDS:
                identifier_mapping.append(0)
            elif token[0] == '"':
                in_string = True
                identifier_mapping.append(0)
                if token[-1] == '"':
                    if len(token) > 1:
                        if token[-2] != '\\':
                            in_string = False
                    else:
                        in_string = False
            elif token[0].isalpha() or token[0] == '_': # should I add '$'? 
                identifier_mapping.append(1)
            else:
                identifier_mapping.append(0)
        elif token[-1] == '"':
            in_string = False;
            identifier_mapping.append( 0 )
        else:
            identifier_mapping.append( 0 )
    
    return identifier_mapping


def token_to_subtoken_map(id_map, subtokenized_code, code=None):
    subtoken_id_map = []

    token_index = 0
    i = 0
    while i < len(subtokenized_code):
        subtoken_id_map.append(id_map[token_index])
        # print('put', id_map[token_index])
        if not subtokenized_code[i].endswith('@@'):
            # print('token_index', token_index, len(subtokenized_code) - i, code[token_index], subtokenized_code[i])
            token_index += 1
        i += 1
    return subtoken_id_map


if __name__ == "__main__":
    file = "/mnt/datastore/inf/groups/cdt_ds/mpatsis/PhD/rafaelository/data/Miltos/tokenized/test/java_test_slp_pre";
    heur_file = "/mnt/datastore/inf/groups/cdt_ds/mpatsis/PhD/rafaelository/data/Miltos/tokenized/test/java_test_slp_pre_sub"
    bpe2000_file = "/mnt/datastore/inf/groups/cdt_ds/mpatsis/PhD/rafaelository/data/Miltos/tokenized/test/java_test_slp_pre_enc_bpe_2000"
    bpe5000_file = "/mnt/datastore/inf/groups/cdt_ds/mpatsis/PhD/rafaelository/data/Miltos/tokenized/test/java_test_slp_pre_enc_bpe_5000"
    bpe10000_file = "/mnt/datastore/inf/groups/cdt_ds/mpatsis/PhD/rafaelository/data/Miltos/tokenized/test/java_test_slp_pre_enc_bpe_10000"

    id_map_file = "/mnt/datastore/inf/groups/cdt_ds/mpatsis/PhD/rafaelository/data/Miltos/tokenized/test/id_map_java_test_slp_pre"
    heur_map_file = "/mnt/datastore/inf/groups/cdt_ds/mpatsis/PhD/rafaelository/data/Miltos/tokenized/test/id_map_java_test_slp_pre_sub"
    bpe2000_map_file = "/mnt/datastore/inf/groups/cdt_ds/mpatsis/PhD/rafaelository/data/Miltos/tokenized/test/id_map_java_test_slp_pre_bpe_2000"
    bpe5000_map_file = "/mnt/datastore/inf/groups/cdt_ds/mpatsis/PhD/rafaelository/data/Miltos/tokenized/test/id_map_java_test_slp_pre_bpe_5000"
    bpe10000_map_file = "/mnt/datastore/inf/groups/cdt_ds/mpatsis/PhD/rafaelository/data/Miltos/tokenized/test/id_map_java_test_slp_pre_bpe_10000"
    
    with open(file, 'r') as fr, open(heur_file, 'r') as f_heur, \
        open(bpe2000_file, 'r') as f_bpe2000, open(bpe5000_file, 'r') as f_bpe5000, \
            open(bpe10000_file, 'r') as f_bpe10000, open(id_map_file, 'w') as fw, \
                open(heur_map_file, 'w') as fw_heur, open(bpe2000_map_file, 'w') as fw_bpe2000, \
                    open(bpe5000_map_file, 'w') as fw_bpe5000, open(bpe10000_map_file, 'w') as fw_bpe10000:
        
        for line, heur_line, bpe2000_line, bpe5000_line, bpe10000_line in zip(fr, f_heur, f_bpe2000, f_bpe5000, f_bpe10000):
            code = line.rstrip('\n')[4: -5].split()
            code_heur = heur_line.rstrip('\n')[4: -5].split()
            code_bpe2000 = bpe2000_line.rstrip('\n')[4: -5].split()
            code_bpe5000 = bpe5000_line.rstrip('\n')[4: -5].split()
            code_bpe10000 = bpe10000_line.rstrip('\n')[4: -5].split()
            
            identifier_mapping = get_Java_identifier_mapping(code)
            assert(len(identifier_mapping) == len(code))
            print(len(identifier_mapping))
            # heur_id_mapping = token_to_subtoken_map(identifier_mapping, code_heur, code)
            bpe2000_id_mapping = token_to_subtoken_map(identifier_mapping, code_bpe2000)
            bpe5000_id_mapping = token_to_subtoken_map(identifier_mapping, code_bpe5000)
            bpe10000_id_mapping = token_to_subtoken_map(identifier_mapping, code_bpe10000)

            fw.write(str(identifier_mapping))
            fw.write('\n')
            
            # fw_heur.write(str(heur_id_mapping))
            # fw_heur.write('\n')
            
            fw_bpe2000.write(str(bpe2000_id_mapping))
            fw_bpe2000.write('\n')
            
            fw_bpe5000.write(str(bpe5000_id_mapping))
            fw_bpe5000.write('\n')

            fw_bpe10000.write(str(bpe10000_id_mapping))
            fw_bpe10000.write('\n')

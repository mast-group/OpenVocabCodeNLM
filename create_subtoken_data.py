import re
import sys

from datetime import date

def subtoken_instance_generator(token_instance_generator):
    for token_instance in token_instance_generator:
        subtokens = []
        for token in token_instance:
            if re.search('[a-zA-Z]', token) is None:
                subtokens.append(token)
            else:
                stokens = split_to_subtokens(token)
                # if len(stokens) > 1:
                    # stokens.insert(0, SUBTOKENS_START)
                    # stokens.append(SUBTOKENS_END)
                    # print 'subtokens', stokens
                    # sys.exit(0)
                subtokens.extend(stokens)
        # print subtokens:
        # if len(subtokens) >= 2924689:
        #     half = 2924689/2
        #     yield subtokens[0:half]
        #     yield subtokens[half:]
        # else:
        #     yield subtokens
        yield subtokens


def instance_generator(data_file):
    with open(data_file, 'rb') as instances:
        for instance in instances:
            tokens = []
            for wh_token in instance.split():
                if wh_token == '.':
                    tokens.append(wh_token)
                else:
                    for token in re.split('(\.)', wh_token):
                        tokens.append(token)
            # print 'tokens', tokens
            yield tokens


def split_to_subtokens(identifier):
    subtokens = []
    # CamelCase split
    matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', identifier)
    camel_subtokens = [m.group(0) for m in matches]

    # if len(camel_subtokenssubtokens) > 1:
    #     print 'st:', camel_subtokens
    # underscore split
    for camel_subtoken in camel_subtokens:
        for subtoken in re.split('(_)', camel_subtoken):
            subtokens.append(subtoken + '@@')
    subtokens[-1] = subtokens[-1][:-2]
    if len(subtokens) == 0:
        print('error:', identifier)
    return subtokens


def export_to_file(subtoken_instance_generator, export_file):
    with open(export_file, 'w') as f:
        for subtoken_instance in subtoken_instance_generator:
            f.write(' '.join(subtoken_instance))
            f.write('\n')


if __name__ == "__main__":

    # Create Subtoken data for Java
    # Validation
    datapath = '/mnt/datastore/inf/groups/cdt_ds/mpatsis/PhD/rafaelository/data/Miltos/tokenized/'
    # dataset = 'validation'
    tokens_file = datapath + 'validation/java_validation_slp_pre'
    export_file = datapath + 'validation/java_validation_slp_pre_sub'
    export_to_file(subtoken_instance_generator(instance_generator(tokens_file)), export_file)

    # Test
    tokens_file = datapath + 'test/java_test_slp_pre'
    export_file = datapath + 'test/java_test_slp_pre_sub'
    export_to_file(subtoken_instance_generator(instance_generator(tokens_file)), export_file)

    # Training
    tokens_file = datapath + 'training/java_training_slp_pre'
    export_file = datapath + 'training/java_training_slp_pre_sub'
    export_to_file(subtoken_instance_generator(instance_generator(tokens_file)), export_file)

    # Training
    tokens_file = datapath + 'training/java_training_slp_huge_pre'
    export_file = datapath + 'training/java_training_slp_huge_pre_sub'
    export_to_file(subtoken_instance_generator(instance_generator(tokens_file)), export_file)


    # Create Subtoken data for Python
    # Validation
    datapath = '/mnt/datastore/inf/groups/cdt_ds/mpatsis/PhD/rafaelository/data/codeCorpora/python/tokenized/'
    # dataset = 'validation'
    tokens_file = datapath + 'validation_set_pre'
    export_file = datapath + 'validation_set_pre_sub'
    export_to_file(subtoken_instance_generator(instance_generator(tokens_file)), export_file)

    # Test
    tokens_file = datapath + 'test_set_pre'
    export_file = datapath + 'test_set_pre_sub'
    export_to_file(subtoken_instance_generator(instance_generator(tokens_file)), export_file)

    # Training
    tokens_file = datapath + 'small_training_set_pre'
    export_file = datapath + 'small_training_set_pre_sub'
    export_to_file(subtoken_instance_generator(instance_generator(tokens_file)), export_file)

    # Training
    tokens_file = datapath + 'full_training_set_pre'
    export_file = datapath + 'full_training_set_pre_sub'
    export_to_file(subtoken_instance_generator(instance_generator(tokens_file)), export_file)


    # Create Subtoken data for C
    # Validation
    datapath = '/mnt/datastore/inf/groups/cdt_ds/mpatsis/PhD/rafaelository/data/codeCorpora/c/tokenized/'
    # dataset = 'validation'
    tokens_file = datapath + 'validation_set_pre'
    export_file = datapath + 'validation_set_pre_sub'
    export_to_file(subtoken_instance_generator(instance_generator(tokens_file)), export_file)

    # Test
    tokens_file = datapath + 'test_set_pre'
    export_file = datapath + 'test_set_pre_sub'
    export_to_file(subtoken_instance_generator(instance_generator(tokens_file)), export_file)

    # Training
    tokens_file = datapath + 'small_training_set_pre'
    export_file = datapath + 'small_training_set_pre_sub'
    export_to_file(subtoken_instance_generator(instance_generator(tokens_file)), export_file)

    # Training
    tokens_file = datapath + 'full_training_set_pre'
    export_file = datapath + 'full_training_set_pre_sub'
    export_to_file(subtoken_instance_generator(instance_generator(tokens_file)), export_file)

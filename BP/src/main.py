from data.data_processing import create_clean_data, save_train_test, remove_file
from data.word2vec import word2vec
from initialize.generator import save_network
from initialize.validation import train_test, load_random_sentences, explain_model

nn_types = ['mlp', 'rnn', 'cnn', 'lstm']
w2v_types = ['mlp_w2v', 'rnn_w2v', 'cnn_w2v', 'lstm_w2v']
# nn_types = []
# w2v_types = ['cnn_w2v']


def do_everything():
    # Clean the reports file
    remove_file("reports.txt")

    # Create models of each network and save them to files, and test the networks using an 80/20 train/test
    # split
    print("\nTRAIN TEST SPLIT\n\n")
    file_path = 'reports.txt'
    with open(file_path, 'a') as file:
        file.write("TRAIN TEST SPLIT\n\n\n")
    file.close()
    sentences_set = False
    tokenizer = None
    for nn_type in nn_types:
        print("Evaluating " + nn_type)
        tokenizer, length, voc_size = save_network(nn_type)
        x_test, y_test = train_test(nn_type)
        if not sentences_set:
            load_random_sentences(x_test, y_test, tokenizer)
            sentences_set = True
        explain_model(nn_type)

    for nn_type in w2v_types:
        print("Evaluating " + nn_type)
        tokenizer, length, voc_size = save_network(nn_type)
        x_test, y_test = train_test(nn_type)
        if not sentences_set:
            load_random_sentences(x_test, y_test, tokenizer)
            sentences_set = True
        explain_model(nn_type)

    # print("\nKFOLD\n\n")
    # with open(file_path, 'a') as file:
    #     file.write("\nKFOLD\n\n\n")
    # file.close()
    # for nn_type in nn_types:
    #     print("Evaluating " + nn_type)
    #     kfold(nn_type)
    #
    # for nn_type in w2v_types:
    #     print("Evaluating " + nn_type)
    #     kfold(nn_type)
    # save_network("cnn")
    # kfold("cnn")


def train_test_one_model(nn_types):
    # Clean the tokens in the text data, save cleaned dataset
    create_clean_data()
    # Split cleaned data into train/test
    save_train_test()
    # Generate word2vec embedding file
    word2vec()
    # Clean the reports file
    remove_file("reports.txt")
    remove_file("explanations.txt")

    print("\nTRAIN TEST SPLIT\n\n")
    file_path = 'reports.txt'
    with open(file_path, 'a') as file:
        file.write("TRAIN TEST SPLIT\n\n\n")
    file.close()
    for nn_type in nn_types:
        print("Evaluating " + nn_type)
        tokenizer, length, voc_size = save_network(nn_type)
        train_test(nn_type, tokenizer)


if __name__ == '__main__':
    do_everything()
    # train_test_one_model(["lstm_w2v", "lstm"])

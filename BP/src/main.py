from data.data_processing import create_clean_data, save_train_test, remove_file
from data.word2vec import word2vec
from initialize.generator import save_network
from initialize.validation import train_test, test
from initialize.kfold import kfold

nn_types = ['cnn', 'rnn', 'mlp', 'lstm']
w2v_types = ['cnn_w2v', 'rnn_w2v', 'mlp_w2v', 'lstm_w2v']


def do_everything():
    # Clean the tokens in the text data, save cleaned dataset
    create_clean_data()
    # Split cleaned data into train/test
    save_train_test()
    # Generate word2vec embedding file
    word2vec()
    # Clean the reports file
    remove_file("reports.txt")
    remove_file("explanations.txt")

    # Create models of each network and save them to files, and test the networks using an 80/20 train/test
    # split
    print("\nTRAIN TEST SPLIT\n\n")
    file_path = 'reports.txt'
    with open(file_path, 'a') as file:
        file.write("TRAIN TEST SPLIT\n\n\n")
    file.close()
    for nn_type in nn_types:
        print("Evaluating " + nn_type)
        save_network(nn_type)
        train_test(nn_type)

    for nn_type in w2v_types:
        print("Evaluating " + nn_type)
        save_network(nn_type)
        train_test(nn_type)

    print("\nKFOLD\n\n")
    with open(file_path, 'a') as file:
        file.write("\nKFOLD\n\n\n")
    file.close()
    for nn_type in nn_types:
        print("Evaluating " + nn_type)
        kfold(nn_type)

    for nn_type in w2v_types:
        print("Evaluating " + nn_type)
        kfold(nn_type)
    save_network("cnn")
    kfold("cnn")


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


def test_one_model(nn_types):
    test(nn_types[0])


if __name__ == '__main__':
    # do_everything()
    # test_one_model(["mlp", "mlp_w2v"])
    train_test_one_model(["mlp_w2v", "mlp"])

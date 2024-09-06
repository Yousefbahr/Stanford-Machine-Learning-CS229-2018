import collections
from audioop import reverse
from plistlib import writePlist

import numpy as np

import util
import svm
from svm import train_and_predict_svm


def get_words(message):
    """Get the normalized list of words from a message string.

    This function should split a message into words, normalize them, and return
    the resulting list. For splitting, you should split on spaces. For normalization,
    you should convert everything to lowercase.

    Args:
        message: A string containing an SMS message

    Returns:
       The list of normalized words from the message.
    """

    # *** START CODE HERE ***
    words = message.split()
    normalized = [word.lower() for word in words]
    return np.array(normalized)
    # *** END CODE HERE ***


def create_dictionary(messages):
    """Create a dictionary mapping words to integer indices.

    This function should create a dictionary of word to indices using the provided
    training messages. Use get_words to process each message. 

    Rare words are often not useful for modeling. Please only add words to the dictionary
    if they occur in at least five messages.

    Args:
        messages: A list of strings containing SMS messages

    Returns:
        A python dict mapping words to integers.
    """
    # *** START CODE HERE ***
    the_dict = dict()
    all_words = dict()
    index = 0
    for message in messages:
        words = get_words(message)
        for word in words:
            # if word found before
            if all_words.get(word) is not None:
                count = all_words.get(word)
                all_words.update({word: count + 1})
            # first time seeing the word
            else:
                all_words.update({word: 0})

    # get only words with freq greater than 5
    for word, count in all_words.items():
        if count >= 5:
            the_dict.update({word: index})
            index += 1

    return the_dict
    # *** END CODE HERE ***


def transform_text(messages, word_dictionary):
    """Transform a list of text messages into a numpy array for further processing.

    This function should create a numpy array that contains the number of times each word
    appears in each message. Each row in the resulting array should correspond to each 
    message and each column should correspond to a word.

    Use the provided word dictionary to map words to column indices. Ignore words that 
    are not present in the dictionary. Use get_words to get the words for a message.

    Args:
        messages: A list of strings where each string is an SMS message.
        word_dictionary: A python dict mapping words to integers.

    Returns:
        A numpy array marking the words present in each message.
    """
    # *** START CODE HERE ***
    result = np.zeros((len(messages), len(word_dictionary)))
    for i, message in enumerate(messages):
        words = get_words(message)
        for word in words:
            if word_dictionary.get(word) is not None:
                index = word_dictionary.get(word)
                result[i][index] += 1

    return result
    # *** END CODE HERE ***


def fit_naive_bayes_model(matrix, labels):
    """Fit a naive bayes model.

    This function should fit a Naive Bayes model given a training matrix and labels.

    The function should return the state of that model.

    Feel free to use whatever datatype you wish for the state of the model.

    Args:
        matrix: A numpy array containing word counts for the training data
        labels: The binary (0 or 1) labels for that training data

    Returns: The trained model
    """

    # *** START CODE HERE ***

    # num for numerator
    # denom for denominator
    phi_num_y_1 = []
    phi_num_y_0 = []

    # numerator
    for feature in range(matrix.shape[1]):
        my_sum_1 = 0
        my_sum_0 = 0
        for row in range(matrix.shape[0]):
            # for y == 1
            if labels[row] == 1:
                my_sum_1 += matrix[row][feature]

            else:
                my_sum_0 += matrix[row][feature]

        phi_num_y_1.append(my_sum_1 + 1)
        phi_num_y_0.append(my_sum_0 + 1)

    # denominator
    denom_sum_1 = 1 * matrix.shape[1]
    denom_sum_0 = 1 * matrix.shape[1]
    for row in range(matrix.shape[0]):
        for feature in range(matrix.shape[1]):
            if labels[row] == 1:
                denom_sum_1 += matrix[row][feature]

            else:
                denom_sum_0 += matrix[row][feature]

    phi_y_1 = np.array([x / denom_sum_1 for x in phi_num_y_1])
    phi_y_0 = np.array([x / denom_sum_0 for x in phi_num_y_0])

    prior_1 = np.sum(labels == 1) / len(labels)
    prior_0 = np.sum(labels == 0) / len(labels)

    return {"phi_1":phi_y_1,
            "phi_0": phi_y_0,
            "prior_1": prior_1,
            "prior_0": prior_0}

    # *** END CODE HERE ***


def predict_from_naive_bayes_model(model, matrix):
    """Use a Naive Bayes model to compute predictions for a target matrix.

    This function should be able to predict on the models that fit_naive_bayes_model
    outputs.

    Args:
        model: A trained model from fit_naive_bayes_model
        matrix: A numpy array containing word counts

    Returns: A numpy array containg the predictions from the model
    """
    # *** START CODE HERE ***

    prior_1 = np.log(model["prior_1"])
    prior_0 = np.log(model["prior_0"])

    preds = np.zeros((matrix.shape[0]))
    for row in range(matrix.shape[0]):
        y_1 = prior_1
        y_0 = prior_0
        for feature in range(matrix.shape[1]):
            y_1 += matrix[row][feature] * np.log(model["phi_1"][feature])
            y_0 += matrix[row][feature] * np.log(model["phi_0"][feature])

        if y_1 > y_0:
            preds[row] = 1
        else:
            preds[row] = 0

    return preds
    # *** END CODE HERE ***


def get_top_five_naive_bayes_words(model, dictionary):
    """Compute the top five words that are most indicative of the spam (i.e positive) class.

    Ues the metric given in 6c as a measure of how indicative a word is.
    Return the words in sorted form, with the most indicative word first.

    Args:
        model: The Naive Bayes model returned from fit_naive_bayes_model
        dictionary: A mapping of word to integer ids

    Returns: The top five most indicative words in sorted order with the most indicative first
    """
    # *** START CODE HERE ***
    result = {}
    for word, feature in dictionary.items():
        result.update({word: np.log( model["phi_1"][feature] / model["phi_0"][feature] )})

    return list(dict(sorted(result.items(), key=lambda item: item[1], reverse= True)).keys())[:5]
    # *** END CODE HERE ***


def compute_best_svm_radius(train_matrix, train_labels, val_matrix, val_labels, radius_to_consider):
    """Compute the optimal SVM radius using the provided training and evaluation datasets.

    You should only consider radius values within the radius_to_consider list.
    You should use accuracy as a metric for comparing the different radius values.

    Args:
        train_matrix: The word counts for the training data
        train_labels: The spma or not spam labels for the training data
        val_matrix: The word counts for the validation data
        val_labels: The spam or not spam labels for the validation data
        radius_to_consider: The radius values to consider
    
    Returns:
        The best radius which maximizes SVM accuracy.
    """
    # *** START CODE HERE ***
    results = {}
    for radius in radius_to_consider:
        preds = train_and_predict_svm(train_matrix, train_labels, val_matrix, radius)
        accuracy = np.mean(preds == val_labels)
        results.update({accuracy: radius})

    return sorted(results.items(), reverse=True)[0][1]
    # *** END CODE HERE ***


def main():
    train_messages, train_labels = util.load_spam_dataset('data/ds6_train.tsv')
    val_messages, val_labels = util.load_spam_dataset('data/ds6_val.tsv')
    test_messages, test_labels = util.load_spam_dataset('data/ds6_test.tsv')
    
    dictionary = create_dictionary(train_messages)

    util.write_json('output/p06_dictionary', dictionary)

    train_matrix = transform_text(train_messages, dictionary)

    np.savetxt('output/p06_sample_train_matrix', train_matrix[:100,:])

    val_matrix = transform_text(val_messages, dictionary)
    test_matrix = transform_text(test_messages, dictionary)

    naive_bayes_model = fit_naive_bayes_model(train_matrix, train_labels)

    naive_bayes_predictions = predict_from_naive_bayes_model(naive_bayes_model, test_matrix)

    np.savetxt('output/p06_naive_bayes_predictions', naive_bayes_predictions)

    naive_bayes_accuracy = np.mean(naive_bayes_predictions == test_labels)

    print('Naive Bayes had an accuracy of {} on the testing set'.format(naive_bayes_accuracy))

    top_5_words = get_top_five_naive_bayes_words(naive_bayes_model, dictionary)

    print('The top 5 indicative words for Naive Bayes are: ', top_5_words)


    util.write_json('output/p06_top_indicative_words', top_5_words)

    optimal_radius = compute_best_svm_radius(train_matrix, train_labels, val_matrix, val_labels, [0.01, 0.1, 1, 10])

    util.write_json('output/p06_optimal_radius', optimal_radius)

    print('The optimal SVM radius was {}'.format(optimal_radius))

    svm_predictions = svm.train_and_predict_svm(train_matrix, train_labels, test_matrix, optimal_radius)

    svm_accuracy = np.mean(svm_predictions == test_labels)

    print('The SVM model had an accuracy of {} on the testing set'.format(svm_accuracy, optimal_radius))


if __name__ == "__main__":
    main()

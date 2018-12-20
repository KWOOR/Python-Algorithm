import pickle

# Input features (x_train, x_test)
# - ret2: previous 2-month return
# - ret3: previous 3-month return
# - ret6: previous 6-month return
# - ret9: previous 9-month return
# - ret12: previous 12-month return
#
# Output (y_train, y_test)
# - the class of a stock: 0, 1, ..., 9
#
# r_train, r_test
# - the next month return. need these to calculate financial performance of portfolios.

def get_sample():
    train = pickle.load(open("data/train.pickle", "rb"))
    test = pickle.load(open("data/test.pickle", "rb"))

    x_train = {}
    y_train = {}
    r_train = {}
    for m in range(36):
        x_train[m] = train[m].loc[:, ['ret2', 'ret3', 'ret6', 'ret9', 'ret12']]
        y_train[m] = train[m].loc[:, 'label']
        r_train[m] = train[m].loc[:, 'target_ret_1']

    x_test = {}
    y_test = {}
    r_test = {}
    for m in range(36):
        x_test[m] = test[m].loc[:, ['ret2', 'ret3', 'ret6', 'ret9', 'ret12']]
        y_test[m] = test[m].loc[:, 'label']
        r_test[m] = test[m].loc[:, 'target_ret_1']

    return x_train, y_train, r_train, x_test, y_test, r_test

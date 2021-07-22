
def train_test_divide(train_test, train_length):
    train = train_test[:train_length]
    test = train_test[train_length:]
    return train, test

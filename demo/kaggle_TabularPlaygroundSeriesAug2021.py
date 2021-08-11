from autox import AutoX
path = '../input/tabular-playground-series-aug-2021'
autox = AutoX(target = 'loss', train_name = 'train.csv', test_name = 'test.csv',
               id = ['id'], path = path)
sub = autox.get_submit()
sub.to_csv("submission.csv", index = False)

import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf

from tensorflow.keras.layers import Embedding, LSTM, Dense


class lstm:
    def fileWalker(file, dirName):
        dataframe = pandas.read_csv(dirName + "/1000.csv", low_memory=False, memory_map=True, header=None)
        dataset = dataframe.values
        # split into input (X) and output (Y) variables

        data = {"Frequency": 25,
                "BlockFrequency": 100,
                "Runs": 100,
                "LongestRunOfOnes": 128,
                "Rank": 38912,
                "DiscreteFourierTransform": 1000,
                "NonOverlappingTemplateMatchings": 1048576,
                "OverlappingTemplateMatching": 10000,
                "Universal": 387840,
                "LinearComplexity": 1000000,
                "Serial": 32,
                "ApproximateEntropy": 127,
                "CumulativeSums": 100,
                "RandomExcursions": 1000000,
                "RandomExcursionsVariant": 1000000}

        dataWidth = data[file];

        X = dataset[:, 0:dataWidth].astype(bool)
        Y = dataset[:, dataWidth]

        encoder = LabelEncoder()
        encoder.fit(Y)
        encoded_Y = encoder.transform(Y)

        tf.experimental_relax_shapes=True;

        # estimator = KerasClassifier(build_fn=create_baseline, epochs=10, batch_size=50, verbose=0)
        estimator = KerasClassifier(build_fn=lstm.create_baseline, input_dim=dataWidth, epochs=10, batch_size=50, verbose=0)
        kfold = StratifiedKFold(n_splits=5, shuffle=True)
        # kfold = StratifiedKFold(n_splits=10, shuffle=True)
        results = cross_val_score(estimator, X, encoded_Y, cv=kfold)

        f = open("./Results/" + file + ".txt", "w")
        f.write("%s %s Baseline: %.4f%% (%.4f%%)" % (file, file, results.mean() * 100, results.std() * 100))
        print("%s Baseline: %.4f%% (%.4f%%)" %  (file, results.mean() * 100, results.std() * 100))

    def create_baseline(input_dim=None):
        # create model
        model = Sequential()
        model.add(Embedding(input_dim=input_dim, output_dim=1))
        model.add(LSTM(units=1))
        model.add(Dense(units=1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model



""" Rubikloud take home problem """
import luigi


class CleanDataTask(luigi.Task):
    """ Cleans the input CSV file by removing any rows without valid geo-coordinates.

        Output file should contain just the rows that have geo-coordinates and
        non-(0.0, 0.0) files.
    """
    tweet_file = luigi.Parameter()
    output_file = luigi.Parameter(default='clean_data.csv')

    def output(self):
        return luigi.LocalTarget(self.output_file)

    def run(self):
        import pandas as pd

        data = pd.read_csv(self.tweet_file, encoding='ANSI', index_col='_unit_id')
        data = data.dropna(subset=['tweet_coord'])
        data = data[(data['tweet_coord']!='[0.0, 0.0]')]

        data.to_csv(self.output_file)


class TrainingDataTask(luigi.Task):
    """ Extracts features/outcome variable in preparation for training a model.

        Output file should have columns corresponding to the training data:
        - y = airline_sentiment (coded as 0=negative, 1=neutral, 2=positive)
        - X = a one-hot coded column for each city in "cities.csv"
    """
    tweet_file = luigi.Parameter()
    cities_file = luigi.Parameter(default='cities.csv')
    output_file = luigi.Parameter(default='features.csv')

    # TODO...

    def requires(self):
        return CleanDataTask(self.tweet_file)

    def output(self):
        return luigi.LocalTarget(self.output_file)

    def run(self):
        import numpy as np
        import pandas as pd

        clean_data = pd.read_csv('clean_data.csv', usecols=['_unit_id', 'airline_sentiment', 'tweet_coord'], index_col='_unit_id')
        cities_data = pd.read_csv(self.cities_file, usecols=['name', 'latitude', 'longitude'])

        # convert all coordinates to numpy arrays
        cities_data['coord'] = cities_data.apply(lambda x : np.array((x['latitude'], x['longitude'])), axis=1)
        clean_data['coord'] = self._convert_tweet_coord(clean_data['tweet_coord'])

        # find closest city to each tweet
        clean_data['closest_city'] = clean_data['coord'].apply(lambda tweet_coord : self._find_closest_city(tweet_coord, cities_data))

        # one hot encode
        features = pd.get_dummies(clean_data['closest_city'])
        # add target back
        features['target'] = clean_data['airline_sentiment']
        # write to output_file
        features.to_csv(self.output_file)

    def _convert_tweet_coord(self, coord_series):
        import numpy as np

        coord = coord_series.str.replace('[','').str.replace(']','') \
                .apply(lambda x : np.fromstring(x, sep=','))
        return coord

    def _find_closest_city(self, location, cities_data):
        import numpy as np

        distances = cities_data['coord'].apply(lambda city_coord : np.linalg.norm(location - city_coord))
        min_distance_id = distances.idxmin()
        closest_city = cities_data['name'][min_distance_id]
        return closest_city


class TrainModelTask(luigi.Task):
    """ Trains a classifier to predict negative, neutral, positive
        based only on the input city.

        Output file should be the pickle'd model.
    """
    tweet_file = luigi.Parameter()
    output_file = luigi.Parameter(default='model.pkl')

    def requires(self):
        return TrainingDataTask(self.tweet_file)

    def output(self):
        return luigi.LocalTarget(self.output_file)

    def run(self):
        import pandas as pd
        import pickle
        from sklearn.tree import DecisionTreeClassifier

        features = pd.read_csv('features.csv')
        # with the current simple features, nothing more complex than
        # a decision tree is expected to output increased accuracy
        tweets_model = DecisionTreeClassifier()
        tweets_model.fit(features.drop('target', axis=1), features['target'])

        pickle.dump(tweets_model, open(self.output_file, 'wb'))


class ScoreTask(luigi.Task):
    """ Uses the scored model to compute the sentiment for each city.

        Output file should be a four column CSV with columns:
        - city name
        - negative probability
        - neutral probability
        - positive probability
    """
    tweet_file = luigi.Parameter()
    output_file = luigi.Parameter(default='scores.csv')

    # TODO...
    def requires(self):
        return TrainModelTask(self.tweet_file)

    def output(self):
        return luigi.LocalTarget(self.output_file)

    def run(self):
        with self.output().open('w') as out_file:
            out_file.write('score not completed yet!')


if __name__ == "__main__":
    luigi.run()

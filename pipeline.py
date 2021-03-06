
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

    def requires(self):
        return CleanDataTask(self.tweet_file)

    def output(self):
        return luigi.LocalTarget(self.output_file)

    def run(self):
        import numpy as np
        import pandas as pd

        clean_data = pd.read_csv(self.input().path, usecols=['_unit_id', 'airline_sentiment', 'tweet_coord'], index_col='_unit_id')
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

        # Due to the nature of this exercise
        # there will be no train-test split

        # write to output_file
        features.to_csv(self.output_file, index_label='_unit_id')

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

        features = pd.read_csv(self.input().path, index_col='_unit_id')
        features = features.replace(to_replace={'target':{'negative':0, 'neutral':1, 'positive':2}})
        # with the current simple features, there seems to be no compelling
        # reason to try any model more complex than a decision tree
        cities_model = DecisionTreeClassifier()
        cities_model.fit(features.drop('target', axis=1), features['target'])

        pickle.dump(cities_model, open(self.output_file, 'wb'))


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

    def requires(self):
        return {'features': TrainingDataTask(self.tweet_file),
                'model': TrainModelTask(self.tweet_file)}

    def output(self):
        return luigi.LocalTarget(self.output_file)

    def run(self):
        import pandas as pd
        import pickle
        from sklearn.tree import DecisionTreeClassifier

        # Produce a DataFrame with one-hot encoded vectors for
        # all cities in the training dataset
        score_data = pd.read_csv(self.input()['features'].path, index_col='_unit_id')
        score_data = score_data.drop(labels=['target'], axis=1)
        score_data = score_data.drop_duplicates()

        # load model and predict probabilities
        cities_model = pickle.load(open(self.input()['model'].path, 'rb'))
        results = cities_model.predict_proba(score_data)
        results = pd.DataFrame(results)

        # replace indexes with city names
        city_list = score_data[score_data==1].stack().reset_index().set_index('_unit_id').rename(columns={'level_1':'city_name'})
        results = results.set_index(city_list['city_name'])

        results = results.sort_values(by=[0, 1], ascending=False)

        results.to_csv(self.output_file)


if __name__ == "__main__":
    luigi.run()

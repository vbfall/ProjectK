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
        import pandas

        data = pandas.read_csv(self.tweet_file, encoding='ANSI', index_col='_unit_id')
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


class TrainModelTask(luigi.Task):
    """ Trains a classifier to predict negative, neutral, positive
        based only on the input city.

        Output file should be the pickle'd model.
    """
    tweet_file = luigi.Parameter()
    output_file = luigi.Parameter(default='model.pkl')

    # TODO...


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
        return CleanDataTask(self.tweet_file)

    def output(self):
        return luigi.LocalTarget(self.output_file)

    def run(self):
        with self.output().open('w') as out_file:
            out_file.write('score not completed yet!')


if __name__ == "__main__":
    luigi.run()

from . import dists
import csv

class Searcher:
    def __init__(self, db_path):
        self.db_path = db_path

    def search(self, queryFeatures, numResults=10):
        results = {}

        # open database for reading
        with open(self.db_path) as f:
            reader = csv.reader(f)

            # loop over rows of the index
            for row in reader:
                # parse out the image ID and features, then compute the chi-squared
                # distance between the features in our database and the query features
                features = [float(x) for x in row[1:]]
                d = dists.chi2_distance(features, queryFeatures)

                # now that we have the distance between the two feature vectors, we
                # can update the results dictionary -- they key is the current image
                # ID in the database and the value is the distance we just computed,
                # representing how 'similar' the image in the database is to our query
                results[row[0]] = d
            
        # sort our results, so that teh smaller distances (i.e. the more relevant images)
        # are at the front of the list
        results = sorted([(v, k) for (k, v) in results.items()])

        return results[:numResults]
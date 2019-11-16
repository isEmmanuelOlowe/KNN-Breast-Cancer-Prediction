class KNearestNeighbours:

    def __init__(self, df):
        """
        Creates new KNN Object.

        :param df: the dataframe being used
        """
        self.df = df

    def classify(self, data, k):
        """
        Predicts the classification of data given proximity to other k nearest points

        :param data: data point being predicted
        :return: class prediction
        """
        selection = self.df.copy()

        def Euclidean_Distance(row):
            """
            Determines the distance between a row and the data point being classified.
            """
            total = 0
            for (columnName, columnData) in data.iteritems():
                total += (columnData - row[columnName]) ** 2
            return total ** (1/2)
        # runs the Euclidean Distance method on every
        # Creates new column for results
        selection["distance"] = selection.apply(Euclidean_Distance, axis=1)
        # Sorts the dataframe to by proximity to the datapoint
        sorted_selection = selection.sort_values("distance")
        # choose only the k closest to select from as prediction
        k_sorted_selection = sorted_selection.head(k)
        # Automatically sorted in ascending order
        # choose category with highest occurence as prediction
        choices = k_sorted_selection["diagnosis"].value_counts()
        choice_list = list(choices.iteritems())
        # returns the choice
        return choice_list[0][0]

    def test(self, test_data):
        """
        Tests the model against a set of testing data to see the Accuracy of model in predictions

        :param test_data: the data being used to test model
        """
        success = 0
        total = 0
         # Determines the number of correct predictions
        for n in range(len(test_data)):
            total += 1
            prediction = self.classify(test_data.iloc[n, 1:], 3)
            if prediction == test_data.iloc[n, 0]:
                success += 1
        print("Accuracy " + str(success/total))

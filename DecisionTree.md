## Decision Tree in Machine Learning

A decision tree is a flowchart-like structure in which each internal node represents a `test` on a feature (e.g. whether a coin flip comes up heads or tails) , each leaf node represents a `class label` (decision taken after computing all features) and branches represent conjunctions of features that lead to those class labels. The paths from root to leaf represent `classification rules`. Below diagram illustrate the basic diagram of decision tree decision making with labels (Rain(Yes), No Rain(No)).

![FlowDiagram](flowdiagram.png)

Decision tree is one of the predictive modelling approaches used in `statistics`, `data mining` and `machine learning`.

Basically decision tree is divided into two parts **classification and regression tree**.

Tree models where the target variable can take a discrete set of values are called **classification trees**. Decision trees where the target variable can take continuous values (typically real numbers) are called **regression trees**. Classification And Regression Tree (CART) is general term for this.

Throughout this post i will try to explain using the examples.

#### Data Format

Data comes in records of forms.

    (x,Y)=(x1,x2,x3,....,xk,Y)

The dependent variable, Y, is the target variable that we are trying to understand, classify or generalize. The vector x is composed of the features, x1, x2, x3 etc., that are used for that task.

**Example**
      
      training_data = [
                      ['Green', 3, 'Apple'],
                      ['Yellow', 3, 'Apple'],
                      ['Red', 1, 'Grape'],
                      ['Red', 1, 'Grape'],
                      ['Yellow', 3, 'Lemon'],
                      ]
     # Header = ["Color", "diameter", "Label"]
     # The last column is the label.
     # The first two columns are features.

#### Approach to make decision tree

While making decision tree, at each node of tree we ask different type of questions. Based on the asked question we will calculate the information gain corresponding to it.

Information gain is used to decide which feature to split on at each step in building the tree. Simplicity is best, so we want to keep our tree small. To do so, at each step we should choose the split that results in the purest daughter nodes. A commonly used measure of purity is called information. For each node of the tree, the information value **measures how much `information` a feature gives us about the class. The split with the highest information gain will be taken as the first split and the process will continue until all children nodes are pure, or until the information gain is 0.**

### Asking Question
 
    class Question:
      """A Question is used to partition a dataset.

      This class just records a 'column number' (e.g., 0 for Color) and a
      'column value' (e.g., Green). The 'match' method is used to compare
      the feature value in an example to the feature value stored in the
      question. See the demo below.
      """

      def __init__(self, column, value):
          self.column = column
          self.value = value

      def match(self, example):
          # Compare the feature value in an example to the
          # feature value in this question.
          val = example[self.column]
          if is_numeric(val):
              return val >= self.value
          else:
              return val == self.value

      def __repr__(self):
          # This is just a helper method to print
          # the question in a readable format.
          condition = "=="
          if is_numeric(self.value):
              condition = ">="
          return "Is %s %s %s?" % (
              header[self.column], condition, str(self.value))
            

Lets try querying questions and its outputs.

    Question(1, 3) ## Is diameter >= 3?
    Question(0, "Green") ## Is color == Green?
    
Now we will try to Partition the dataset based on asked question. Data will be divided into two classes at each steps.

      def partition(rows, question):
        """Partitions a dataset.

        For each row in the dataset, check if it matches the question. If
        so, add it to 'true rows', otherwise, add it to 'false rows'.
        """
        true_rows, false_rows = [], []
        for row in rows:
            if question.match(row):
                true_rows.append(row)
            else:
                false_rows.append(row)
        return true_rows, false_rows
        
       # Let's partition the training data based on whether rows are Red.
       true_rows, false_rows = partition(training_data, Question(0, 'Red'))
       # This will contain all the 'Red' rows.
       true_rows ## [['Red', 1, 'Grape'], ['Red', 1, 'Grape']]
       false_rows ## [['Green', 3, 'Apple'], ['Yellow', 3, 'Apple'], ['Yellow', 3, 'Lemon']]
       
Algorithm for constructing decision tree usually works top-down, by choosing a variable at each step that `best` splits the set of items. Different algorithms use different metrices for measuring `best`.

### Gini Impurity

First let's understand the meaning of **Pure** and **Impure**.

#### Pure
Pure means, in a selected sample of dataset all data belongs to same class (PURE).

#### Impure
Impure means, data is mixture of different classes.

#### Definition of Gini Impurity

Gini Impurity is a measurement of the likelihood of an incorrect classification of a new instance of a random variable, if that new instance were randomly classified according to the distribution of class labels from the data set.

If our dataset is `Pure` then likelihood of incorrect classification is 0. If our sample is mixture of different classes then likelihood of incorrect classification will be high.

**Example**

        # Demo 1:
        # Let's look at some example to understand how Gini Impurity works.
        #
        # First, we'll look at a dataset with no mixing.
        no_mixing = [['Apple'],
                     ['Apple']]
        # this will return 0
        gini(no_mixing) ## output=0
       
       ## Demo 2:
       # Now, we'll look at dataset with a 50:50 apples:oranges ratio
        some_mixing = [['Apple'],
                       ['Orange']]
        # this will return 0.5 - meaning, there's a 50% chance of misclassifying
        # a random example we draw from the dataset.
        gini(some_mixing) ##output=0.5
        
        ## Demo 3:
        # Now, we'll look at a dataset with many different labels
        lots_of_mixing = [['Apple'],
                          ['Orange'],
                          ['Grape'],
                          ['Grapefruit'],
                          ['Blueberry']]
        # This will return 0.8
        gini(lots_of_mixing) ##output=0.8
        #######
           

 ## Steps
 
 * Get list of rows which are taken into consideration for making decision tree (recursively at each nodes).
 * Calculate uncertanity of our dataset or Gini impurity or how much our data is mixed up etc.
     * Generate list of all question which needs to be asked at that node.
     * Partition rows into `True rows` and `False rows` based on each question asked.
     * Calculate information gain based gini impurity and partition of data.
     * Update highest information gain based on each question asked.
     * Update best question based on information gain (higher information gain).
 * Divide the node on best question.

**Code for Above Steps**

     def find_best_split(rows):
        """Find the best question to ask by iterating over every feature / value
        and calculating the information gain."""
        best_gain = 0  # keep track of the best information gain
        best_question = None  # keep train of the feature / value that produced it
        current_uncertainty = gini(rows)
        n_features = len(rows[0]) - 1  # number of columns

        for col in range(n_features):  # for each feature

            values = set([row[col] for row in rows])  # unique values in the column

            for val in values:  # for each value

                question = Question(col, val)

                # try splitting the dataset
                true_rows, false_rows = partition(rows, question)

                # Skip this split if it doesn't divide the
                # dataset.
                if len(true_rows) == 0 or len(false_rows) == 0:
                    continue

                # Calculate the information gain from this split
                gain = info_gain(true_rows, false_rows, current_uncertainty)

                # You actually can use '>' instead of '>=' here
                # but I wanted the tree to look a certain way for our
                # toy dataset.
                if gain >= best_gain:
                    best_gain, best_question = gain, question

        return best_gain, best_question
        
        #######
        # Demo:
        # Find the best question to ask first for our toy dataset.
        best_gain, best_question = find_best_split(training_data)
        best_question
        ## output - Is diameter >= 3?
 
 
 Now build the Decision tree based on step discussed above recursively at each node.
 
     def build_tree(rows):
        """Builds the tree.

        Rules of recursion: 1) Believe that it works. 2) Start by checking
        for the base case (no further information gain). 3) Prepare for
        giant stack traces.
        """

        # Try partitioing the dataset on each of the unique attribute,
        # calculate the information gain,
        # and return the question that produces the highest gain.
        gain, question = find_best_split(rows)

        # Base case: no further info gain
        # Since we can ask no further questions,
        # we'll return a leaf.
        if gain == 0:
            return Leaf(rows)

        # If we reach here, we have found a useful feature / value
        # to partition on.
        true_rows, false_rows = partition(rows, question)

        # Recursively build the true branch.
        true_branch = build_tree(true_rows)

        # Recursively build the false branch.
        false_branch = build_tree(false_rows)

        # Return a Question node.
        # This records the best feature / value to ask at this point,
        # as well as the branches to follow
        # dependingo on the answer.
        return Decision_Node(question, true_branch, false_branch)
        
 

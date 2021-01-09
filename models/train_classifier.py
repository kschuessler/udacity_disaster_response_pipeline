import nltk, sys, pickle
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sqlalchemy import create_engine
nltk.download(['punkt', 'wordnet'])


def load_data(database_filepath, table='messages'):
    """
    INPUT:
        database_filepath - path to the database
        table - name of table (default is 'messages')
    OUTPUT:
        X - data frame containing the feature
        Y - data frame containing the targets
    """
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table(table, engine)
    X = df['message']
    Y = df.drop(['message', 'id', 'genre', 'original', 'child_alone'], axis=1)

    return X, Y


def tokenize(text):
    """
    INPUT:
        text - raw text
    OUTPUT:
        clean_tokens - clean, tokenized text
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """
    INPUT:
        n/a
    OUTPUT:
        cv - model
    """
    # build ML pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfdif', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    # use GridSearchCV to improve model performance
    parameters = {'tfdif__use_idf': (True, False),
                  'clf__estimator__n_estimators': [10, 20]}
    cv = GridSearchCV(pipeline, parameters, verbose=1)

    return cv


def evaluate_model(model, X_test, Y_test):
    """
    INPUT:
        model - built model
        X_test - data frame containing test data for feature
        Y_test - data frame containing test data for targets
    OUTPUT:
        results - data frame containing weighted average of precision, recall, and f-score for each target
    """
    Y_pred = model.predict(X_test)
    results = pd.DataFrame(columns=['category', 'precision', 'recall', 'f_score'])
    for i, column in enumerate(Y_test):
        precision, recall, f_score, support = precision_recall_fscore_support(Y_test[column], Y_pred[:, i],
                                                                              average='weighted')
        results.at[i, 'category'] = column
        results.at[i, 'precision'] = precision
        results.at[i, 'recall'] = recall
        results.at[i, 'f_score'] = f_score
    print('Aggregated f_score:', results['f_score'].mean())
    print('Aggregated precision:', results['precision'].mean())
    print('Aggregated recall:', results['recall'].mean())

    return results


def save_model(model, model_filepath):
    """
    INPUT:
        model - the model to be saved
        model_filepath - filepath where the model should be saved
    OUTPUT:
        n/a
    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y = load_data(database_filepath, table='messages')
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
import pickle

def load():
	return pickle.load(open('digitsData/database.p','rb'))

def save(database):
	pickle.dump(database,open('digitsData/database.p','wb'))

def updateErrors(database, trueClass, prediction):
	database[trueClass]['errors'] += 1
	database[trueClass][prediction] += 1

def initData():
	digitsDict = {
        '0': {
            'errors':0,
            '0': 0, '1': 0, '2': 0, '3': 0, '4': 0, 
            '5': 0, '6': 0, '7': 0, '8': 0, '9': 0,
        },
        '1': {
            'errors':0,
            '0': 0, '1': 0, '2': 0, '3': 0, '4': 0, 
            '5': 0, '6': 0, '7': 0, '8': 0, '9': 0,
        },
        '2': {
            'errors':0,
            '0': 0, '1': 0, '2': 0, '3': 0, '4': 0, 
            '5': 0, '6': 0, '7': 0, '8': 0, '9': 0,
        },
        '3': {
            'errors':0,
            '0': 0, '1': 0, '2': 0, '3': 0, '4': 0, 
            '5': 0, '6': 0, '7': 0, '8': 0, '9': 0,
        },
        '4': {
            'errors':0,
            '0': 0, '1': 0, '2': 0, '3': 0, '4': 0, 
            '5': 0, '6': 0, '7': 0, '8': 0, '9': 0,
        },
        '5': {
            'errors':0,
            '0': 0, '1': 0, '2': 0, '3': 0, '4': 0, 
            '5': 0, '6': 0, '7': 0, '8': 0, '9': 0,
        },
        '6': {
            'errors':0,
            '0': 0, '1': 0, '2': 0, '3': 0, '4': 0, 
            '5': 0, '6': 0, '7': 0, '8': 0, '9': 0,
        },
        '7': {
            'errors':0,
            '0': 0, '1': 0, '2': 0, '3': 0, '4': 0, 
            '5': 0, '6': 0, '7': 0, '8': 0, '9': 0,
        },
        '8': {
            'errors':0,
            '0': 0, '1': 0, '2': 0, '3': 0, '4': 0, 
            '5': 0, '6': 0, '7': 0, '8': 0, '9': 0,
        },
        '9': {
            'errors':0,
            '0': 0, '1': 0, '2': 0, '3': 0, '4': 0, 
            '5': 0, '6': 0, '7': 0, '8': 0, '9': 0,
        }
    }
	return digitsDict
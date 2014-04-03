import csv as csv
import numpy as np
#import pylab as pl
from sklearn.ensemble import RandomForestClassifier


csvObject = csv.reader(open('../data/train.csv', 'rt'))

#skip header
trainHeader = next(csvObject)
print(trainHeader)

data = []

for row in csvObject:
    data.append(row)

data = np.array(data)
print(data[0])
#Percentage of survival
numPassengers = np.size(data[0::,0].astype(np.float))
numSurvived = np.sum(data[0::,1].astype(np.float))
percentSurvived = (numSurvived / numPassengers)*100

#Women and Men passengers
womenPassengers = data[0::, 4] == "female"
menPassengers = data[0::, 4] == "male"

#row numbers of women and men onboard
womenOnboard = data[womenPassengers, 0].astype(np.float)
menOnboard = data[menPassengers, 0].astype(np.float)

#proportion of women that survived
#womenSurvived/totalWomen

totalWomen = np.size(womenOnboard)
womenSurvived = data[womenPassengers, 1].astype(np.float)
womenSurvivedCount = np.sum(womenSurvived)

percentWomenSurvived = (womenSurvivedCount / totalWomen)*100


totalMen = np.size(menOnboard)
menSurvived = data[menPassengers, 1].astype(np.float)
menSurvivedCount = np.sum(menSurvived)

percentMenSurvived = (menSurvivedCount / totalMen)*100

print('Proportion of women who survived is %s' % percentWomenSurvived)
print('Proportion of men who survived is %s' % percentMenSurvived)

#open a new file and if female predict that they will survive and if male predict that they won't
testFileObj = csv.reader(open('../data/test.csv', 'rt'))
header = next(testFileObj)

fileToWrite = csv.writer(open('../data/genderbasedmodelpy.csv', 'wt'))
fileToWrite.writerow(trainHeader[0:2])

for row in testFileObj:
    if row[3] == 'female':
        row.insert(1, '1')
        fileToWrite.writerow(row[0:2])
    else:
        row.insert(1, '0')
        fileToWrite.writerow(row[0:2])

# survival reference table - [male or female], [1st, 2nd, 3rd class], [4 bins of prices]
# our 4 bins: 0-9,10-19,20-29,30-39

fareCeiling = 40
data[data[0::,9].astype(np.float) >= fareCeiling, 9] = fareCeiling-1.0
fareBracketSize = 10

numberOfPriceBrackets = int(fareCeiling / fareBracketSize)
numberOfClasses = 3

#Multi dimensional array (numDimensions, numRows, numCols)
survivalTable = np.zeros((2, numberOfClasses, numberOfPriceBrackets))

#1st class, 0-9
#1st class 10-19
#1st class 20-29

for row in range(numberOfClasses):
    for col in range(numberOfPriceBrackets):
        womenSurvived = data[
            (data[0::, 4] == "female")
            & (data[0::, 1].astype(np.float) == 1)
            & (data[0::, 2].astype(np.float)
               == row + 1)
            & (data[0:, 9].astype(np.float)
               >= col * fareBracketSize)
            & (data[0:, 9].astype(np.float)
               < (col + 1) * fareBracketSize)
            , 0]

        womenAll = data[
            (data[0::, 4] == "female")
            & (data[0::, 2].astype(np.float)
               == row + 1)
            & (data[0:, 9].astype(np.float)
               >= col * fareBracketSize)
            & (data[0:, 9].astype(np.float)
               < (col + 1) * fareBracketSize)
            , 0]

        if np.size(womenAll) != 0:
            survivalTable[0, row, col] = float(np.size(womenSurvived))/float(np.size(womenAll))

        menSurvived = data[
            (data[0::, 4] != "female")
            & (data[0::, 1].astype(np.float) == 1)
            & (data[0::, 2].astype(np.float)
               == row + 1)
            & (data[0:, 9].astype(np.float)
               >= col * fareBracketSize)
            & (data[0:, 9].astype(np.float)
               < (col + 1) * fareBracketSize)
            , 0]

        menAll = data[
            (data[0::, 4] != "female")
            & (data[0::, 2].astype(np.float)
               == row + 1)
            & (data[0:, 9].astype(np.float)
               >= col * fareBracketSize)
            & (data[0:, 9].astype(np.float)
               < (col + 1) * fareBracketSize)
            , 0]

        if np.size(menAll) != 0:
            survivalTable[1, row, col] = float(np.size(menSurvived))/float(np.size(menAll))

print(survivalTable)

survivalTable[survivalTable < 0.6] = 0
survivalTable[survivalTable >= 0.6] = 1

print(survivalTable)

#open a new file and predict if they will survive from our survival table
testFileObj = csv.reader(open('../data/test.csv', 'rt'))
header = next(testFileObj)
print(header)
fileToWrite = csv.writer(open('../data/genderbasedmodel2py.csv', 'wt'))
fileToWrite.writerow(trainHeader[0:2])


for row in testFileObj:
    for price in range(numberOfPriceBrackets):
        try:
            #make sure there is data
            row[9] = float(row[9])
        except:
            binFare = 3 - float(row[1])
            break
        if row[9] > fareCeiling:
            binFare = numberOfPriceBrackets - 1
            break

    if row[3] == "female":
        row.insert(1, int(survivalTable[0, float(row[1])-1, binFare]))
    else:
        row.insert(1, int(survivalTable[1, float(row[1])-1, binFare]))

    fileToWrite.writerow(row[0:2])

#converting to float
csvObject = csv.reader(open('../data/train.csv', 'rt'))

#gender - female more likely to survive
#class - higher the class the more likely you are to survive
#fare - higher the fare the more likely you are to survive
#SibSp - if you have more than 0 more likely you are to survive

numberOfClasses = 3

ageBracketSize = 10
numberOfAgeBrackets = int(110/ageBracketSize)
#Multi dimensional array (numDimensions, numRows, numCols)
survivalTable2 = np.zeros((2, numberOfClasses, numberOfAgeBrackets))
#1st class, 0-9
#1st class 10-19
#1st class 20-29
age = data[0::,5]
age[age==''] = 0
medianAge = np.median(age.astype(np.float))

for fileRow in csvObject:
    try:
        fileRow[5] = float(fileRow[5])
    except:
        fileRow[5] = medianAge

for row in range(numberOfClasses):
    for col in range(numberOfAgeBrackets):
        womenSurvived = data[
            (data[0::, 4] == "female")
            & (data[0::, 1].astype(np.float) == 1)
            & (data[0::, 2].astype(np.float)
               == row + 1)
            & (data[0:, 5].astype(np.float)
               >= col * ageBracketSize)
            & (data[0::, 5].astype(np.float)
               < (col + 1) * ageBracketSize)
            , 0]

        womenAll = data[
            (data[0::, 4] == "female")
            & (data[0::, 2].astype(np.float)
               == row + 1)
            & (data[0::, 5].astype(np.float)
               >= col * ageBracketSize)
            & (data[0::, 5].astype(np.float)
               < (col + 1) * ageBracketSize)
            , 0]

        if np.size(womenAll) != 0:
            survivalTable2[0, row, col] = float(np.size(womenSurvived))/float(np.size(womenAll))

        menSurvived = data[
            (data[0::, 4] != "female")
            & (data[0::, 1].astype(np.float) == 1)
            & (data[0::, 2].astype(np.float)
               == row + 1)
            & (data[0::, 5].astype(np.float)
               >= col * ageBracketSize)
            & (data[0::, 5].astype(np.float)
               < (col + 1) * ageBracketSize)
            , 0]

        menAll = data[
            (data[0::, 4] != "female")
            & (data[0::, 2].astype(np.float)
               == row + 1)
            & (data[0::, 5].astype(np.float)
               >= col * ageBracketSize)
            & (data[0::, 5].astype(np.float)
               < (col + 1) * ageBracketSize)
            , 0]

        if np.size(menAll) != 0:
            survivalTable2[1, row, col] = float(np.size(menSurvived))/float(np.size(menAll))


print(survivalTable2)
survivalTable2[survivalTable2 < 0.5] = 0
survivalTable2[survivalTable2 >= 0.5] = 1
print(survivalTable2)


fileToWrite = csv.writer(open('../data/genderbasedmodel3py.csv', 'wt'))
fileToWrite.writerow(trainHeader[0:2])
testFileObj = csv.reader(open('../data/test.csv', 'rt'))
header = next(testFileObj)


for row in testFileObj:
    for age in range(numberOfAgeBrackets):
        try:
            #make sure there is data
            row[4] = float(row[4])
            passAge = float(row[4])
        except:
            passAge = medianAge

        if (passAge >= float(age * ageBracketSize)) & (passAge < float((age+1) * ageBracketSize)):
            if row[3] == "female":
                row.insert(1, int(survivalTable2[0, float(row[1])-1, age]))
            else:
                row.insert(1, int(survivalTable2[1, float(row[1])-1, age]))
    fileToWrite.writerow(row[0:2])
#skip header

age = data[0::, 5]
age[age == ''] = 0
medianAge = np.median(age.astype(np.float))

cleanData = []
target = []

csvObject = csv.reader(open('../data/train.csv', 'rt'))
header = next(csvObject)
#PassengerId, Survived, PClass, Sex, Age, Sibsp, Fare
i = 0
for row in csvObject:
    i = i+1
    newRow = [0]*3
    #newRow[0] = float(row[0])
    target.append(float(row[1]))
    newRow[0] = float(row[2])
    if row[4] == "female":
        newRow[1] = 0
    else:
        newRow[1] = 1
    try:
        newRow[2] = float(row[5])
    except:
        newRow[2] = medianAge

    cleanData.append(newRow)
print(i)
print(np.size(target))
cleanTest = []
testFileObj = csv.reader(open('../data/test.csv', 'rt'))
header = next(testFileObj)
#PassengerId, Survived, PClass, Sex, Age, Sibsp, Fare
print(header)
female = 0
for row in testFileObj:
    print(row)
    newRow = [0]*3
   # newRow[0] = float(row[0])
    newRow[0] = float(row[1])
    if row[3] == "female":
        female = female + 1
        newRow[1] = 0
    else:
        newRow[1] = 1
    try:
        newRow[2] = float(row[4])
    except:
        newRow[2] = medianAge

    cleanTest.append(newRow)

print(female)
#sklearn
# Create the random forest object which will include all the parameters
# for the fit
Forest = RandomForestClassifier(n_estimators=100)

# Fit the training data to the training output and create the decision
# trees
Forest = Forest.fit(cleanData, target)

# Take the same decision trees and run on the test data
Output = Forest.predict(cleanTest)
print(Output)
print(np.size(Output))
print(sum(Output))
print(sum(Output)/np.size(Output))


fileToWrite = csv.writer(open('../data/genderbasedmodel4py.csv', 'wt'))
fileToWrite.writerow(trainHeader[0:2])
testFileObj = csv.reader(open('../data/test.csv', 'rt'))
header = next(testFileObj)

rowNum = 0
for row in testFileObj:
    row.insert(1, int(Output[rowNum]))
    try:
        age = float(row[4])
    except:
        age = 100
    if age <= 10:
        row.insert(1, 1)

    if row[3] == 'female':
        row.insert(1,1)
    rowNum = rowNum+1
    fileToWrite.writerow(row[0:2])


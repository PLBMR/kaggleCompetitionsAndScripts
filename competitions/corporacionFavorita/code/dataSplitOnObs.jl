#dataSplitOnObs.jl
#includes my code for splitting the training data into a training set and a 
#validation set; this is based on a split of observations

#imports

using DataFrames

#helpers

function performSplit(mainFrame,propTrain)
    #helper that performs our train-validation split
    rows = range(1,nrow(mainFrame));
    sampleSize = floor(Int,length(rows) * propTrain);
    trainRows = sample(rows,sampleSize,replace = false);
    validationRows = collect(setdiff(Set(rows),Set(trainRows)));
    trainFrame = mainFrame[trainRows,:];
    println(size(trainFrame));
    validationFrame = mainFrame[validationRows,:];
    println(size(validationFrame));
    return trainFrame, validationFrame
end

#main process

#get proportion train
propTrain = float(ARGS[1]);
#load in data
mainFrame = readtable("../data/preprocessed/train.csv");
#then perform split
trainFrame, validationFrame = performSplit(mainFrame,propTrain);
#then write our tables
writetable("../data/preprocessed/train_splitObs.csv",trainFrame)
writetable("../data/preprocessed/validation_splitObs.csv",validationFrame)


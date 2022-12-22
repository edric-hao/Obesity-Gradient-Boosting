%% Loading the Data and storing them in a Table
train = import_train('data.csv');

%% Accessing Feature elements of the Table

% Accessing features and preprocessing the training dataset
train.Gender = categorical(train.Gender);
train.family_history = categorical(train.family_history);
train.FCHCF = categorical(train.FCHCF);
train.CFBM = categorical(train.CFBM, {'no', 'Sometimes', 'Frequently', 'Always'},'Ordinal',true);
%train.CFBM = categorical(train.CFBM);
train.Smoke = categorical(train.Smoke);
train.CA = categorical(train.CA, {'no', 'Sometimes', 'Frequently', 'Always'},'Ordinal',true);
%train.CA = categorical(train.CA);
train.CCM = categorical(train.CCM);
train.Transportation = categorical(train.Transportation, {'Walking', 'Bike', 'Motorbike', 'Automobile', 'Public_Transportation'},'Ordinal',true);
%train.Transportation = categorical(train.Transportation);

%Feature engineering
train.BMI = train.Weight./train.Height.^2;

%Save to Table
rowNames = ["Height"; "Weight"; "Gender"; "Age"; "BMI"; "family_history"; "FCHCF"; "FCV"; "NMM"; "CFBM"; "Smoke"; "CW"; "CCM"; "PAF"; "TUT"; "CA"; "Transportation"];
X = table(train.Height, train.Weight, train.Gender, train.Age, train.BMI, train.family_history, train.FCHCF, train.FCV, train.NMM, train.CFBM, train.Smoke, train.CW, train.CCM, train.PAF, train.TUT, train.CA, train.Transportation, 'VariableNames', rowNames);
Y = train.Obesity;

%Train Validation Split
x_train = X([1:1477], :);
x_validate = X([1478:2111], :);
y_train = Y([1:1477], :);
y_validate = Y([1478:2111], :);

%% Model Creation and Hyperparameter Tuning

%current best reproducible model
%rng('default') % For reproducibility
c = cvpartition(Y,'KFold',10);
t = templateTree('Reproducible',true,'MinLeafSize',1,'SplitCriterion','gdi','NumVariablesToSample',3,'MaxNumSplits',1008);
model = fitcensemble(X, Y, 'Learners', t, 'Method', 'AdaBoostM2','NumLearningCycles',146,'LearnRate',0.096439, 'CVPartition',c);
treeRate = kfoldLoss(model);
display(treeRate);

%experimental models
%rng('default') % For reproducibility
%c = cvpartition(Y,'KFold',10);
%t = templateTree('Reproducible',true,'MinLeafSize',1,'SplitCriterion','gdi','NumVariablesToSample',3);
%model = fitcensemble(X, Y, "OptimizeHyperparameters", {'NumLearningCycles','MaxNumSplits','LearnRate'}, 'Learners', t, 'Method', 'AdaBoostM2','HyperparameterOptimizationOptions',struct('MaxObjectiveEvaluations',500,'CVPartition',c));
%model = fitcensemble(X, Y, "OptimizeHyperparameters", {'MinLeafSize','SplitCriterion','NumVariablesToSample'},'HyperparameterOptimizationOptions',struct('MaxObjectiveEvaluations',50));
%treeRate = kfoldLoss(model);
%display(treeRate);

%train test split models
%t = templateTree('Reproducible',true,'MinLeafSize',1,'SplitCriterion','gdi','NumVariablesToSample',3,'MaxNumSplits',1008);
%model = fitcensemble(x_train, y_train, 'Learners', t, 'Method', 'AdaBoostM2','NumLearningCycles',146,'LearnRate',0.096439);

%label_train = predict(model, x_train);
%label_validate = predict(model, x_validate);
%y_train = categorical(y_train);
%y_validate = categorical(y_validate);
%label_train = categorical(label_train);
%label_validate = categorical(label_validate);

%figure(1)
%hold on
%plotconfusion(y_train, label_train)

%figure(2)
%hold on
%plotconfusion(y_validate, label_validate)

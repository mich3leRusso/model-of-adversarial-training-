clear all
close all force
warning off

siz=[224 224];

load('DatasColor_37.mat','DATA');%to load the dataset used in this example
%the information to split data between training and test set
DIV=DATA{3};
DIM1=DATA{4};%per datas color_65 mettere 835
DIM2=DATA{5};
lab=DATA{2};%label
NX=DATA{1};%cell array that stores the image

fold=1;

trainPattern=(DIV(fold,1:DIM1));%id of the training patterns
testPattern=(DIV(fold,DIM1+1:DIM2));%id of the test patterns
y=lab(DIV(fold,1:DIM1));%label of the training set
labelTE=lab(DIV(fold,DIM1+1:DIM2));%label of the test set
classes = categories(categorical(y));
numClasses = max(y);
miniBatchSize=60;

clear nome trainingImages
for pattern=1:DIM1%for all the images 
    IM=NX{DIV(fold,pattern)};
    IM=imresize(IM,[siz(1) siz(2)]);
    %se si vogliono usare immagini non rgb usare quest
    %IM=rgb2gray(IM);
    %if IM(:,:,1)==1
     %   IM(:,:,2)=IM;
     %   IM(:,:,3)=IM;
    %end
    trainingImages(:,:,:,pattern)=IM;
end 
DIM=length(y);
approccio=1;%inserire data augumentation, altrimenti qualsiasi altro valore per non averla
if approccio==1%method App1 of the paper
    App1;
end
net=resnet18;
lgraph = layerGraph(net);
%sistemazione della rete 
%inserisco i layer di normalizzazione
lgraph = removeLayers(lgraph, {'ClassificationLayer_predictions','prob','fc1000'});
%layer per la shallow finale
newLayers = [
    fullyConnectedLayer(numClasses,'Name','fc','WeightLearnRateFactor',20,'BiasLearnRateFactor', 20)
    softmaxLayer('Name','softmax')
    classificationLayer('Name','classoutput');
    ];
lgraph = addLayers(lgraph,newLayers);
lgraph = connectLayers(lgraph,'pool5','fc');
%numero epoch
numEpoches=15;
numObservations = numel(y);
numIterationsPerEpoch = floor(numObservations./miniBatchSize);

%CNN training options
metodoOptim='sgdm';
options = trainingOptions(metodoOptim,...
    'MiniBatchSize',60,...
    'MaxEpochs',1,...
    'InitialLearnRate',0.001,...
    'Verbose',false,...
    'Plots','training-progress');   
%batch per la normalizzazione degli adversarial pattern
      batch_adv=batchNormalizationLayer('Name',"bn_conv1",'TrainedMean',ones([1 1 64]),'TrainedVariance',ones([1 1 64]),'Offset',ones([1 1 64]),'Scale',ones([1 1 64]));
      %inizio addestramento
      for j=1:numEpoches            
            idx=randperm(DIM1);
            train1=trainingImages(:,:,:,idx);  
            label1=y(idx);
        for i=1:numIterationsPerEpoch
            %shuffle data 
            close all
            idx=randperm(DIM1,120);
            train=train1(:,:,:,idx);
            label=label1(idx);
            %creo minibatch con adv
            [train_adv,label_adv]=create_adversarial(train,label,120,lgraph);
            train = augmentedImageDatastore(siz,train,categorical(label));
            netTransfer = trainNetwork(train,lgraph,options);
            lgraph=layerGraph(netTransfer);
            batch=lgraph.Layers(3);
            %chiudo le finestre
            set(groot,'ShowHiddenHandles','on')
            c = get(groot,'Children');
            delete(c)
            %sostituisco con il layer di normalizzazione pre gli adv
            lgraph = replaceLayer(lgraph,lgraph.Layers(3).Name,batch_adv);
            train = augmentedImageDatastore(siz,train_adv,categorical(label_adv));
            %alleno la rete con gli adversarial examples
            netTransfer1 = trainNetwork(train,lgraph,options);
            lgraph=layerGraph(netTransfer1);
            batch_adv=lgraph.Layers(3);
            lgraph = replaceLayer(lgraph,lgraph.Layers(3).Name,batch);
            %chiudo le finestre
            set(groot,'ShowHiddenHandles','on')
            c = get(groot,'Children');
            delete(c)
        end
      end
      %test della rete
        for pattern=ceil(DIM1)+1:ceil(DIM2)
             IM=NX{DIV(fold,pattern)};
             IM=imresize(IM,[siz(1) siz(2)]);
             %se si vogliono usare immagini non rgb usare quest
             %IM=rgb2gray(IM);
             %if IM(:,:,1)==1
             %   IM(:,:,2)=IM;
             %   IM(:,:,3)=IM;
             %end
            testImages(:,:,:,pattern-ceil(DIM1))=uint8(IM);
        end      
[Ypredict,score{fold}] =  classify(netTransfer,testImages);
[a,b]=max(score{fold}');
accuracy = sum(labelTE == b)./numel(Ypredict);
save netTransfer